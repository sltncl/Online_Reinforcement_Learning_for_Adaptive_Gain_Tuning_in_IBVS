import numpy as np
import os
import pickle
from collections import defaultdict
import random

class QlearningAgent:
    def __init__(self, node, q_table_file="~/ros_workspace/src/utils/qlearning_model/q_table_ibvs.pkl"):
        """
        Q-learning agent for selecting optimal servo gain values 
        based on reinforcement learning feedback.
        """
        self.node = node
        # === Hyperparameters ===
        self.gamma = 0.99                                   # Discount factor: how much future rewards are valued
        self.servo_gain_range = np.linspace(0.05, 5, 20)    # Discretized range of possible servo gains
        self.alpha = 0.1                                    # Learning rate: weight of new updates vs old values
        self.epsilon = 0.7                                  # Exploration rate: probability of taking a random action
        self.epsilon_decay = 0.9995                         # Multiplicative decay applied to epsilon after each action
        self.epsilon_min = 0.2                              # Minimum exploration threshold

        # Variables to keep track of previous step
        self.last_state = None
        self.last_action = None

        # === Q-table management ===
        # Q-table stores expected rewards for each state-action pair
        self.q_table_file = os.path.expanduser(q_table_file)
        os.makedirs(os.path.dirname(self.q_table_file), exist_ok=True)
        self.Q = self.load()  # Load existing Q-table if available

    def discretize_state(self, state_vec, decimals=1):
        """
        Convert a continuous state vector into a discrete.
        Rounding ensures similar states are grouped together in the Q-table.
        """
        return tuple(np.round(state_vec.flatten(), decimals))
    
    def select_action(self, state):
        """
        Epsilon-greedy policy:
        - With probability epsilon, choose a random action (exploration).
        - Otherwise, select the action with the highest Q-value (exploitation).
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(self.servo_gain_range) - 1)
        return int(np.argmax(self.Q[state]))
    
    def update_Q(self, reward, state_vec):
        """
        Update the Q-table using the Bellman equation:
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        """
        if self.last_state is not None:
            state = self.discretize_state(state_vec)
            best_next = np.max(self.Q[state])  # Estimate of optimal future value
            old_value = self.Q[self.last_state][self.last_action]
            # Apply Q-learning update rule
            self.Q[self.last_state][self.last_action] = \
                old_value + self.alpha * (reward + self.gamma * best_next - old_value)

    def act(self, state_vec):
        """
        Select an action given the current state:
        - Discretize state
        - Choose action using epsilon-greedy policy
        - Save state-action for learning updates
        - Occasionally save Q-table
        - Decay epsilon
        """
        state = self.discretize_state(state_vec)

        # Action selection
        action_idx = self.select_action(state)
        action = self.servo_gain_range[action_idx]

        # Store current step for next update
        self.last_state = state
        self.last_action = action_idx

        # Occasionally save Q-table to file
        if random.random() < 0.1:
            self.save()

        # Apply epsilon decay (exploration decreases over time)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action

    def save(self):
        """
        Save the Q-table to disk as a pickle file.
        Ensures persistence of learned policy across sessions.
        """
        try:
            with open(self.q_table_file, "wb") as f:
                pickle.dump(dict(self.Q), f)
            self.node.get_logger().info(f"Q-table saved at {self.q_table_file}")
        except Exception as e:
            self.node.get_logger().warn(f"[QlearningAgent] Failed to save Q-table: {e}")

    def load(self):
        """
        Load the Q-table from disk if available.
        If not found, initialize an empty defaultdict with zeroed action values.
        """
        try:
            with open(self.q_table_file, "rb") as f:
                loaded = pickle.load(f)
                return defaultdict(lambda: np.zeros(len(self.servo_gain_range)), loaded)
        except FileNotFoundError:
            # Return a fresh Q-table with default zero values
            return defaultdict(lambda: np.zeros(len(self.servo_gain_range)))