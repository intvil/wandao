"""
Draft step sequencing helpers.
"""

from wandao.config import TOTAL_STEPS

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
