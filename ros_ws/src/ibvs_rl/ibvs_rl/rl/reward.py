import numpy as np

def compute_reward(e, e_dot, fx, fy):
    """
    Compute the reward signal for visual servoing or control tasks.

    The reward is based on the image-plane error (e) and its time derivative (e_dot).
    A sparse success reward is given when the error is within tolerance,
    otherwise a quadratic penalty is applied.

    Args:
        e (np.ndarray): Feature error vector.
        e_dot (np.ndarray): Time derivative of the error.
        fx (float): Camera focal length in x-direction (pixels).
        fy (float): Camera focal length in y-direction (pixels).

    Returns:
        float: Reward value.  
               +1.0 if the error is within tolerance,  
               -1.0 if invalid input (NaN or None),  
               otherwise a negative quadratic penalty.
    """

    # === Input validation ===
    if e is None or np.any(np.isnan(e)):
        return -1.0

    # Reshape error into (N_points, 2)
    e_uv = e.reshape(-1, 2)

    # Success condition: all errors within tolerance
    ok_u = np.all(np.abs(e_uv[:, 0]) < 5.0 / fx)
    ok_v = np.all(np.abs(e_uv[:, 1]) < 5.0 / fy)
    if ok_u and ok_v:
        return 1.0

    # === Quadratic penalty ===
    # Weight matrix for error term
    W_e = 0.6 * np.eye(e.shape[0])

    # Weight matrix for error derivative term
    W_edot = 0.1 * np.eye(e_dot.shape[0])

    # Negative quadratic cost as reward
    return float(-(e.T @ W_e @ e + e_dot.T @ W_edot @ e_dot))