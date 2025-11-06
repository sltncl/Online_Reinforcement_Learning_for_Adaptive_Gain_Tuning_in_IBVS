import os
import threading
import torch
import torch.optim as optim
import numpy as np
import time
import rclpy

from ibvs_rl.rl.models import QNetwork, ActionFunction
from ibvs_rl.rl.replay_buffer import ReplayBuffer
from ibvs_rl.rl.optimizer import ga_argmax_action
from ibvs_rl.rl.trainer import Trainer 
from ibvs_rl.utils.normalizer import Normalizer 


class CAQLAgent:
    """
    Continuous Action Q-Learning Agent (CAQL)
    =========================================

    Implements a hybrid reinforcement learning agent combining 
    **Q-learning with continuous actions** and optional **actor approximation**.  
    The agent alternates between:
      - Gradient Ascent (GA)-based exploration phase
      - Policy (Actor)-based exploitation phase

    It supports asynchronous background training using a replay buffer 
    and can operate as a ROS2 node component (with integrated logging).

    Attributes:
        node (rclpy.node.Node): ROS2 node for logging and parameter handling.
        state_dim (int): Dimension of state input.
        action_dim (int): Dimension of continuous action space.
        device (torch.device): Computational device (CPU/GPU).
        qnet, q_target, qnet_live (QNetwork): Neural networks for Q-learning.
        actor, actor_live (ActionFunction): Optional actor networks for policy-based control.
        replay (ReplayBuffer): Experience replay buffer.
        trainer (Trainer): Trainer handling optimization and soft updates.
    """
    def __init__(self, node, state_dim=6, action_dim=1, model_dir="~/ros_workspace/src/utils/caql_models"):
        """
        Initialize the CAQLAgent.

        Args:
            node (rclpy.node.Node): ROS2 Node instance (used for logging, timers, etc.).
            state_dim (int): Dimension of input state vector.
            action_dim (int): Dimension of action space.
            model_dir (str): Directory path to save/load model parameters.
        """
        # === Core setup ===
        self.node = node
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State normalization utility
        self.normalizer = Normalizer(state_dim)

        # === Exploration–Exploitation scheduling ===
        self.mode_cycle_length = 3000   # Full cycle length (samples)
        self.ga_phase_length = 1500     # GA phase duration
        self.actor_phase_length = 1500  # Actor phase duration
        self.sample_counter = 0         # Global sample counter

        self.node.get_logger().info(f"[CAQLAgent] Device: {self.device}")

        # === Hyperparameters ===
        self.gamma = 0.99                           # Discount factor for future rewards
        self.batch_size = 64                        # Mini-batch size for training
        self.train_interval = 3.0                   # Interval (in seconds) between training updates
        self.lambda_bounds = (0.05, 5.0)            # Action bounds for continuous parameter (e.g., servo gain)
        self.lambda_bounds_normalized = (-1, 1)     # Action bounds normalized 
        self.ga_iters = 50                          # Iterations for genetic algorithm optimizer
        self.ga_lr = 1e-4                           # Learning rate for GA optimization
        self.tau = 0.005                            # Soft update parameter for target network
        self.min_replay_size = 250                  # Minimum replay buffer size before training starts

        # === Noise parameters for exploration ===
        self.noise_scale = 0.2
        self.noise_max = 0.2
        self.noise_decay = 0.999
        self.noise_cycle_length = 1000
        self.train_step_count = 0
        self.noise_min = 0.05
        
        # === Neural Network setup ===
        # Online and target Q-networks for Bellman updates
        self.qnet = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.qnet_live = QNetwork(state_dim, action_dim).to(self.device)

        # Sync initialization between networks
        self.qnet_live.load_state_dict(self.qnet.state_dict())
        self.q_target.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-4)

        # === Optional Actor Function ===
        # Used for smooth action prediction during exploitation phase
        self.node.declare_parameter("use_action_function", False)
        self.use_action_function = self.node.get_parameter("use_action_function").get_parameter_value().bool_value
        
        if self.use_action_function:
            self.actor = ActionFunction(state_dim, action_dim).to(self.device)
            self.actor_live = ActionFunction(state_dim, action_dim).to(self.device)
            self.actor_live.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        else:
            self.actor = None
            self.actor_live = None
            self.actor_optimizer = None

        # === Experience Replay ===
        self.replay = ReplayBuffer(capacity=200000)

        # Thread locks to prevent concurrent model access
        self.lock = threading.Lock()
        self.lock_copy = threading.Lock()
        self.lock_actor = threading.Lock()

        # === Model persistence configuration ===
        self.model_dir = os.path.expanduser(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "qnet.pth")
        self.actor_path = os.path.join(self.model_dir, "actor.pth")

        # === Trainer initialization ===
        self.trainer = Trainer(
            self.node, self.replay, self.qnet, self.qnet_live, 
            self.q_target, self.optimizer, self.batch_size, self.gamma, self.lambda_bounds_normalized,
            self.ga_iters, self.ga_lr, self.device, self.lock, self.lock_copy, self.lock_actor, self.tau, self.actor,
            self.actor_live, self.actor_optimizer, self.min_replay_size
        )

        # === Background training thread ===
        self.training_thread = threading.Thread(
            target=self._training_loop, daemon=True
        )
        self.training_thread.start()

        # Tracking previous state/action
        self.last_state = None
        self.last_action = None


    # ============================================================
    # BACKGROUND TRAINING LOOP
    # ============================================================
    def _training_loop(self):
        """
        Asynchronous background training loop.

        Periodically triggers `Trainer.train_timer_cb()` to perform 
        off-policy updates from the replay buffer.
        """
        while rclpy.ok():
            try:
                self.trainer.train_timer_cb()
            except Exception as e:
                self.node.get_logger().warn(f"[CAQLAgent] Training error: {e}")
            time.sleep(self.train_interval)

    # ============================================================
    # POLICY ACTION SELECTION
    # ============================================================
    def act(self, state_vec, warmstart=None):
        """
        Compute the next action given the current state.

        Depending on replay size and current cycle position, the agent:
          - Uses **random exploration** (startup phase)
          - Uses **GA-based Q optimization** (exploration)
          - Uses **actor-based prediction** (exploitation)

        Args:
            state_vec (np.ndarray): Current environment state.
            warmstart (float, optional): Previous or reference action for initialization.

        Returns:
            Tuple[float, float]: Selected action and additive exploration noise.
        """
        # Normalize state and update statistics
        self.normalizer.update(np.array([state_vec], dtype=np.float32))
        state_norm = self.normalizer.normalize(state_vec)

        # Determine current mode within exploration-exploitation cycle
        cycle_pos = self.sample_counter % self.mode_cycle_length
        use_ga = True # default mode

        if len(self.replay) < self.min_replay_size:
            # Initial random exploration
            action = np.random.uniform(*self.lambda_bounds)
            noise = 0.0
        else:
            # Alternate between GA and actor-based inference
            use_ga = cycle_pos < self.ga_phase_length

            if not use_ga and self.actor_live is not None:
                # Exploitation using trained actor
                state_norm_tensor = torch.tensor(state_norm, dtype=torch.float32, device=self.device)
                with self.lock_copy:
                    with torch.no_grad():
                        action_norm = self.actor_live(state_norm_tensor).cpu().numpy().item()
                        action = self.normalizer.denormalize_action(action_norm, self.lambda_bounds)
            else:
                # Exploration or fallback using GA optimization
                try:
                    with self.lock_copy:
                        warmstart_norm = (
                            self.normalizer.normalize_action(warmstart, self.lambda_bounds)
                            if warmstart is not None else None
                        )
                        action_norm = ga_argmax_action(
                            self.qnet_live, state_norm,
                            a_bounds= self.lambda_bounds_normalized,
                            iters=self.ga_iters,
                            lr=self.ga_lr,
                            device=self.device,
                            warmstart=warmstart_norm
                        )
                        action = self.normalizer.denormalize_action(action_norm, self.lambda_bounds)
                except Exception as ex:
                    # Recovery fallback to warmstart or mean action
                    self.node.get_logger().warn(f"[CAQLAgent] GA argmax failed: {ex}, fallback to warmstart")
                    action = warmstart if warmstart is not None else np.mean(self.lambda_bounds)

        # Add adaptive Gaussian noise for continuous exploration
        if len(self.replay)>=self.min_replay_size:
            self.update_noise_scale()
            noise = np.random.normal(0, self.noise_scale)
            action += noise
            action = float(np.clip(action, *self.lambda_bounds))

        # Bookkeeping
        self.last_action = action
        return action, noise

    # ============================================================
    # MEMORY MANAGEMENT
    # ============================================================
    def remember(self, reward, next_state, done=False):
        """
        Store a transition (s, a, r, s') in the replay buffer.

        Args:
            reward (float): Observed scalar reward.
            next_state (np.ndarray): Successor state vector.
            done (bool): Terminal flag.
        """
        if self.last_state is not None and self.last_action is not None:
            self.normalizer.update(np.array([self.last_state, next_state], dtype=np.float32))
            last_state_norm = self.normalizer.normalize(self.last_state)
            next_state_norm = self.normalizer.normalize(next_state)
            last_action_norm = self.normalizer.normalize_action(self.last_action, self.lambda_bounds)
            self.replay.push(last_state_norm, [last_action_norm], reward, next_state_norm, done)
        self.last_state = next_state.copy()
        self.sample_counter += 1

    
    # ============================================================
    # MODEL PERSISTENCE
    # ============================================================
    def save(self):
        """
        Save Q-network and (optionally) Actor to disk.
        Creates or overwrites existing model files.
        """
        try:
            torch.save(self.qnet.state_dict(), self.model_path)
            if self.actor is not None:
                torch.save(self.actor.state_dict(), self.actor_path)
                self.node.get_logger().info(f"[CAQLAgent] Actor saved at {self.actor_path}")
            self.node.get_logger().info(f"[CAQLAgent] Model saved at {self.model_path}")
        except Exception as e:
            self.node.get_logger().warn(f"[CAQLAgent] Failed to save model: {e}")

    def load(self):
        """
        Load model weights for Q-network and optional Actor from disk.
        Automatically syncs Q-target after loading.
        """
        try: 
            if os.path.exists(self.model_path):
                self.qnet.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.q_target.load_state_dict(self.qnet.state_dict())
                self.node.get_logger().info(f"[CAQLAgent] Model loaded from {self.model_path}")
            else:
                self.node.get_logger().warn(f"[CAQLAgent] Q-network file not found: {self.model_path}")
            if self.actor is not None:
                if os.path.exists(self.actor_path):
                    self.actor.load_state_dict(torch.load(self.actor_path, map_location=self.device))
                    self.node.get_logger().info(f"[CAQLAgent] Actor loaded from {self.actor_path}")
                else:
                    self.node.get_logger().warn(f"[CAQLAgent] No saved actor found")
        except Exception as e:
             self.node.get_logger().warn(f"[CAQLAgent] Failed to load model: {e}")

    
    # ============================================================
    # EXPLORATION NOISE MANAGEMENT
    # ============================================================
    def update_noise_scale(self):
        """
        Adaptive noise scheduling across training cycles.
        Noise decays exponentially after each step and resets at cycle start.
        """
        if self.train_step_count % self.noise_cycle_length == 0:
            self.noise_scale = self.noise_max
        else:
            self.noise_scale = max(self.noise_scale * self.noise_decay, self.noise_min)
