import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for Reinforcement Learning.

    Stores transitions of the form (state, action, reward, next_state, done)
    and allows uniform random sampling of mini-batches for training.

    Attributes:
        buf (deque): Fixed-size buffer storing experience tuples.
    """

    def __init__(self, capacity=200000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
                            Oldest entries are discarded when capacity is exceeded.
        """
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        """
        Store a transition in the buffer.

        Args:
            s (array-like): Current state.
            a (array-like): Action taken.
            r (float): Reward received.
            s_next (array-like): Next state.
            done (bool): Terminal flag indicating episode termination.
        """
        self.buf.append((
            np.array(s, dtype=np.float32),                  # State
            np.array(a, dtype=np.float32).reshape(-1,),     # Action (flattened)
            float(r),                                       # Reward
            np.array(s_next, dtype=np.float32),             # Next state
            bool(done)                                      # Done flag
        ))

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
                   Each element is a NumPy array of shape (batch_size, ...).
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, sn, d = zip(*batch)
        return (
            np.stack(s),                          # States
            np.stack(a),                          # Actions
            np.array(r, dtype=np.float32),        # Rewards
            np.stack(sn),                         # Next states
            np.array(d, dtype=np.float32)         # Done flags
        )

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buf)