import torch
import torch.nn as nn
from ibvs_rl.rl.optimizer import ga_argmax_action


class Trainer:
    """
    Reinforcement Learning Trainer for Continous Action Q-learning with optional Actor-Critic extension.

    This class manages:
        - Sampling from the replay buffer
        - Updating the Q-network (critic)
        - Updating the actor (if present)
        - Performing soft updates of target networks
        - Logging training progress via ROS node
    """

    def __init__(self, node, replay, qnet, qnet_live, q_target, optimizer,
                 batch_size, gamma, lambda_bounds, ga_iters, ga_lr, device,
                 lock, lock_copy, lock_actor, tau=0.001,
                 actor=None, actor_live=None, actor_optimizer=None,
                 min_replay_size=500):
        # ROS node for logging
        self.node = node

        # RL components
        self.replay = replay
        self.qnet = qnet
        self.qnet_live = qnet_live
        self.q_target = q_target
        self.optimizer = optimizer
        self.actor = actor
        self.actor_live = actor_live
        self.actor_optimizer = actor_optimizer

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_bounds = lambda_bounds
        self.ga_iters = ga_iters
        self.ga_lr = ga_lr
        self.device = device
        self.tau = tau
        self.min_replay_size = min_replay_size

        # Locks for thread/process safety
        self.lock = lock
        self.lock_copy = lock_copy
        self.lock_actor = lock_actor

        # Training step counter
        self.train_step_count = 0

        # Parameters for dynamic tolerance in GA optimization
        self.k1 = 0.1
        self.k2 = 0.1

    def train_timer_cb(self):
        """
        Periodic callback for performing a training step.
        Checks replay buffer size, executes training, logs results,
        and performs soft update of target networks.
        """
        if len(self.replay) < self.min_replay_size:
            self.node.get_logger().info(
                f"Replay buffer size: {len(self.replay)}/{self.min_replay_size}"
            )
            return

        loss, actor_loss = self.train_step()
        if loss is not None:
            self.node.get_logger().info(
                f"[Train {self.train_step_count}] Q-loss={loss:.6f}, Actor-loss={actor_loss}"
            )

        # Soft update of target critic
        self.soft_update(self.qnet, self.q_target, tau=self.tau)

        self.train_step_count += 1

    def train_step(self):
        """
        Execute a single training step:
            - Sample batch from replay
            - Compute Q-targets
            - Update Q-network
            - Update actor (if present)
            - Update live copies of networks
        """
        if len(self.replay) < self.batch_size:
            return None, None

        # Sample batch
        s, a, r, sn, done = self.replay.sample(self.batch_size)

        # Convert to tensors
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
        if a_t.dim() == 1:
            a_t = a_t.unsqueeze(-1)
        elif a_t.dim() > 2:
            a_t = a_t.view(a_t.size(0), -1)

        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
        sn_t = torch.tensor(sn, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device)

        # === Dynamic tolerance estimation (if actor available) ===
        if self.actor is not None:
            with torch.no_grad():
                with self.lock:
                    a_pred_next = self.actor(sn_t)
                q_next_actor = self.q_target(sn_t, a_pred_next)
            td_errors = torch.abs(
                r_t + self.gamma * (1.0 - done_t) * q_next_actor - self.qnet_live(s_t, a_t)
            )
            td_error_mean = td_errors.mean().item()
            tau_dynamic = td_error_mean * (self.k1 * (self.k2 ** self.train_step_count))
        else:
            tau_dynamic = None

        # === Compute next actions with GA optimization ===
        a_primes = []
        for i in range(self.batch_size):
            warm = float(a[i])
            try:
                with self.lock:
                    a_star = ga_argmax_action(
                        self.qnet, sn[i],
                        a_bounds=self.lambda_bounds,
                        iters=self.ga_iters, lr=self.ga_lr,
                        device=self.device, warmstart=warm, tau_dynamic=tau_dynamic
                    )
            except Exception:
                # Fallback: midpoint of action bounds
                a_star = (self.lambda_bounds[0] + self.lambda_bounds[1]) / 2.0
            a_primes.append([a_star])
        a_primes_t = torch.tensor(a_primes, dtype=torch.float32, device=self.device)

        # === Compute Q-targets ===
        with torch.no_grad():
            q_next = self.q_target(sn_t, a_primes_t)
            target = r_t + self.gamma * (1.0 - done_t) * q_next

        # === Q-network update ===
        with self.lock:
            pred = self.qnet(s_t, a_t)
            loss = nn.MSELoss()(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # === Actor update (if available) ===
        actor_loss = None
        if self.actor is not None and self.actor_optimizer is not None:
            with self.lock:
                a_pred = self.actor(sn_t)
                q_pred = self.qnet(sn_t, a_pred)
                q_star = self.qnet(sn_t, a_primes_t).detach()

                # 1. Imitation loss: actor(s) ≈ a_star
                imitation_loss = nn.MSELoss()(a_pred, a_primes_t)

                # 2. Value consistency: Q(s, actor(s)) ≈ Q(s, a_star)
                value_loss = nn.MSELoss()(q_pred, q_star)

                # 3. Total actor loss (weighted sum)
                alpha = 0.1
                actor_loss = imitation_loss + alpha * value_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # === Update live copies ===
        with self.lock_copy:
            self.qnet_live.load_state_dict(self.qnet.state_dict())

        if self.actor is not None:
            with self.lock_actor:
                self.actor_live.load_state_dict(self.actor.state_dict())

        return float(loss.item()), float(actor_loss.item()) if actor_loss is not None else None

    @staticmethod
    def soft_update(local_net, target_net, tau=0.001):
        """
        Polyak averaging (soft update):
        target ← (1 - tau) * target + tau * local
        """
        for p, tp in zip(local_net.parameters(), target_net.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)