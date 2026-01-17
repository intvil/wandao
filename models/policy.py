"""
Neural network policy for draft agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from draft.config import device, TOTAL_STEPS

# Draft sequence: (side, action) where side 0=A, 1=B; action 'b' ban, 'p' pick
_SEQ = [
    (0, "b"),
    (1, "b"),
    (1, "b"),
    (0, "b"),
    (1, "b"),
    (1, "b"),
    (0, "b"),
    (0, "p"),
    (1, "p"),
    (0, "b"),
    (0, "b"),
    (1, "b"),
    (1, "p"),
    (0, "p"),
    (0, "p"),
    (1, "p"),
    (1, "p"),
    (0, "p"),
    (0, "b"),
    (1, "b"),
    (1, "b"),
    (0, "b"),
    (0, "p"),
    (1, "p"),
]
if len(_SEQ) != TOTAL_STEPS:
    raise ValueError(
        f"TOTAL_STEPS ({TOTAL_STEPS}) must match draft sequence length ({len(_SEQ)})"
    )


def side_to_move(t):
    """Return which side moves at timestep t (0=A, 1=B)."""
    return _SEQ[t][0]


def is_ban_step(t):
    """Return True if timestep t is a ban phase."""
    return _SEQ[t][1] == "b"


def is_pick_step(t):
    """Return True if timestep t is a pick phase."""
    return _SEQ[t][1] == "p"


def multi_hot(indices, n):
    """Create multi-hot encoding from list of indices."""
    x = torch.zeros(n, device=device)
    x[indices] = 1.0
    return x


def masked_categorical_sample(logits, valid_mask):
    """Sample from categorical distribution with masking for invalid actions.

    Args:
        logits: Tensor of shape (n_actions,)
        valid_mask: Boolean tensor of shape (n_actions,)

    Returns:
        Tuple of (action_index, log_probability)
    """
    masked_logits = logits.clone()
    masked_logits[~valid_mask] = -1e9
    probs = F.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    a = dist.sample()
    return a.item(), dist.log_prob(a)


class DraftPolicy(nn.Module):
    """Neural network policy for draft decisions."""

    def __init__(self, n_champs, total_steps):
        """Initialize policy network.

        Args:
            n_champs: Number of champions/heroes
            total_steps: Total number of draft steps
        """
        super().__init__()
        # Input: available heroes + team A picks + team B picks + step one-hot + side one-hot
        inp = 3 * n_champs + total_steps + 2
        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_champs),
        )

    def forward(self, state_vec):
        """Forward pass through policy network.

        Args:
            state_vec: State representation tensor

        Returns:
            Logits for each possible action
        """
        return self.net(state_vec)


def build_state(available, pickA, pickB, t, side):
    """Build state representation for the policy network.

    Args:
        available: Tensor of available heroes (float)
        pickA: Tensor of team A picks (one-hot)
        pickB: Tensor of team B picks (one-hot)
        t: Current timestep
        side: Current side to move (0 or 1)

    Returns:
        Concatenated state vector
    """
    step_oh = torch.zeros(TOTAL_STEPS, device=device)
    step_oh[t] = 1.0
    side_oh = (
        torch.tensor([1.0, 0.0], device=device)
        if side == 0
        else torch.tensor([0.0, 1.0], device=device)
    )
    return torch.cat([available, pickA, pickB, step_oh, side_oh], dim=0)
