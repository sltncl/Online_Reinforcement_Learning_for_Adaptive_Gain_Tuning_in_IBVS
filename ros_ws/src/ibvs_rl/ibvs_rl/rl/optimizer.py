import torch
import numpy as np


def ga_argmax_action(qnet, state_np, a_bounds=(-1, 1), iters=40, lr=1e-2,
                     tol=1e-6, device='cpu', warmstart=None, tau_dynamic=None):
    """
    Gradient-ascent based action selection:
    Finds the action a* that maximizes Q(s, a) for a given state.

    Args:
        qnet (nn.Module): Q-network mapping (state, action) → scalar Q-value.
        state_np (np.ndarray): State vector (numpy array).
        a_bounds (tuple): Action bounds (min, max).
        iters (int): Maximum number of gradient ascent iterations.
        lr (float): Learning rate for gradient ascent.
        tol (float): Convergence tolerance on Q-value improvement.
        device (str): Torch device ('cpu' or 'cuda').
        warmstart (float, optional): Initial action guess. Defaults to midpoint if None.
        tau_dynamic (float, optional): Overrides tolerance if provided.

    Returns:
        float: Action value a* that maximizes Q(s, a) within bounds.
    """

    # Convert state to torch tensor with batch dimension
    s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    a_min, a_max = a_bounds

    # === Warmstart initialization ===
    if warmstart is None:
        warmstart = (a_min + a_max) / 2.0  # Default: midpoint of action range

    # Generate additional candidate starting points in different segments
    seg_points = [
        np.random.uniform(0.1, 0.4),  # First third of the interval
        np.random.uniform(0.6, 0.9)   # Last third of the interval
    ]

    # Map normalized points into [a_min, a_max]
    seg_points = [a_min + p * (a_max - a_min) for p in seg_points]

    # Combine warmstart with segment-based initializations
    warmstarts = [warmstart] + seg_points

    # Override tolerance if dynamic threshold is provided
    if tau_dynamic is not None:
        tol = tau_dynamic

    best_a = None
    best_q = -float("inf")

    # === Gradient ascent from multiple initializations ===
    for ws in warmstarts:
        # Initialize action variable with gradient tracking
        a = torch.tensor([ws], dtype=torch.float32, device=device, requires_grad=True)
        prev_q = None

        for i in range(iters):
            # Forward pass: evaluate Q(s, a)
            q = qnet(s, a.unsqueeze(0))
            q_val = q.squeeze()

            # Reset gradients
            if a.grad is not None:
                a.grad.zero_()

            # Backpropagate to compute ∂Q/∂a
            q_val.backward()
            grad = a.grad
            if grad is None:
                break

            # Gradient ascent update with clamping to action bounds
            with torch.no_grad():
                a += lr * grad
                a.clamp_(a_min, a_max)

            # Convergence check based on Q-value improvement
            q_now = q_val.item()
            if prev_q is not None and abs(q_now - prev_q) < tol:
                break
            prev_q = q_now

            # Reset gradient for next iteration
            if a.grad is not None:
                a.grad.zero_()

        # Evaluate final Q-value for this initialization
        final_q = qnet(s, a.unsqueeze(0)).item()
        if final_q > best_q:
            best_q = final_q
            best_a = float(a.detach().cpu().numpy()[0])

    return best_a