import numpy as np


class Normalizer:
    """
    Online normalizer for state and action vectors.

    Maintains running estimates of mean and variance using
    a numerically stable incremental update rule. Provides
    normalization/denormalization utilities for both states
    and actions.
    """

    def __init__(self, dim, eps=1e-8):
        """
        Initialize the normalizer.

        Args:
            dim (int): Dimensionality of the input vectors.
            eps (float): Small constant to avoid division by zero.
        """
        self.mean = np.zeros(dim, dtype=np.float32)  # Running mean
        self.var = np.ones(dim, dtype=np.float32)    # Running variance
        self.count = eps                             # Effective sample count

    def update(self, x):
        """
        Update running mean and variance with a new batch.

        Args:
            x (np.ndarray): Batch of samples with shape (N, dim).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count

        # Update variance using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        """
        Normalize input using running mean and variance.

        Args:
            x (np.ndarray): Input vector or batch.

        Returns:
            np.ndarray: Normalized input.
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    # -----------------------------
    # Action normalization utilities
    # -----------------------------
    @staticmethod
    def normalize_action(a, a_bounds):
        """
        Normalize action to [-1, 1] given bounds.

        Args:
            a (float or np.ndarray): Action value(s).
            a_bounds (tuple): (a_min, a_max).

        Returns:
            float or np.ndarray: Normalized action(s).
        """
        a_min, a_max = a_bounds
        return 2.0 * (a - a_min) / (a_max - a_min) - 1.0

    @staticmethod
    def denormalize_action(a_norm, a_bounds):
        """
        Denormalize action from [-1, 1] back to original bounds.

        Args:
            a_norm (float or np.ndarray): Normalized action(s).
            a_bounds (tuple): (a_min, a_max).

        Returns:
            float or np.ndarray: Denormalized action(s).
        """
        a_min, a_max = a_bounds
        return a_min + (a_norm + 1.0) * 0.5 * (a_max - a_min)