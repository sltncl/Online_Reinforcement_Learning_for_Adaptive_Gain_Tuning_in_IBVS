import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-network approximator: Q_theta(s, a).
    Maps a state-action pair to a scalar Q-value.

    Input:
        - state vector s ∈ ℝ^(state_dim)
        - action vector a ∈ ℝ^(action_dim)

    Output:
        - scalar Q-value (expected return estimate)
    """

    def __init__(self, state_dim=6, action_dim=1, hidden=[64, 32]):
        super().__init__()
        inp = state_dim + action_dim  # Concatenated input dimension
        layers = []
        prev = inp

        # Hidden layers with ReLU activations
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        # Final linear layer → scalar output
        layers.append(nn.Linear(prev, 1))

        # Wrap layers in a sequential container
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        """
        Forward pass through the Q-network.

        Args:
            s (Tensor): state tensor of shape (batch, state_dim) or (state_dim,)
            a (Tensor): action tensor of shape (batch, action_dim) or (action_dim,)

        Returns:
            Tensor: scalar Q-value(s), shape (batch,) or scalar
        """
        # Ensure batch dimension
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)

        # Flatten higher-dimensional inputs if necessary
        if s.dim() > 2:
            s = s.view(s.size(0), -1)
        if a.dim() > 2:
            a = a.view(a.size(0), -1)

        # Concatenate state and action
        x = torch.cat([s, a], dim=-1)

        # Forward through network and squeeze output to scalar
        return self.net(x).squeeze(-1)


class ActionFunction(nn.Module):
    """
    Policy network (actor): π_phi(s).
    Maps a state vector to a continuous action.

    Input:
        - state vector s ∈ ℝ^(state_dim)

    Output:
        - action vector a ∈ ℝ^(action_dim), bounded in [-1, 1] via Tanh
    """

    def __init__(self, state_dim=6, action_dim=1, hidden=[32, 32]):
        super().__init__()
        layers = []
        prev = state_dim

        # Hidden layers with ReLU activations
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        # Final layer → action_dim outputs
        layers.append(nn.Linear(prev, action_dim))

        # Tanh activation ensures bounded action space
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, s):
        """
        Forward pass through the policy network.

        Args:
            s (Tensor): state tensor of shape (batch, state_dim) or (state_dim,)

        Returns:
            Tensor: action tensor of shape (batch, action_dim) or (action_dim,)
        """
        # Ensure batch dimension
        if s.dim() == 1:
            s = s.unsqueeze(0)

        # Flatten higher-dimensional inputs if necessary
        if s.dim() > 2:
            s = s.view(s.size(0), -1)

        return self.net(s)