import numpy as np


def build_target_square(u_star, v_star, side):
    """
    Build a square target centered at (u_star, v_star).

    Args:
        u_star (float): Horizontal center coordinate.
        v_star (float): Vertical center coordinate.
        side (float): Side length of the square.

    Returns:
        np.ndarray: Array of shape (4, 2) containing the square's corner points
                    in the following order:
                        top-left, top-right, bottom-right, bottom-left
    """
    return np.array([
        [u_star - side / 2.0, v_star - side / 2.0],  # Top-left
        [u_star + side / 2.0, v_star - side / 2.0],  # Top-right
        [u_star + side / 2.0, v_star + side / 2.0],  # Bottom-right
        [u_star - side / 2.0, v_star + side / 2.0],  # Bottom-left
    ])


def order_box_points(pts):
    """
    Order four corner points of a quadrilateral into a consistent order.

    Args:
        pts (np.ndarray): Array of shape (4, 2) containing four (x, y) points.

    Returns:
        np.ndarray: Array of shape (4, 2) with points ordered as:
                    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates → smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates → smallest = top-right, largest = bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect