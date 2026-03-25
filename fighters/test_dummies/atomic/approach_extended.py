"""
Slowly approaches while in extended stance.
Always extended, always approaching -- a simple aggressive dummy.
Used for Level 2: Basic Skills training.
"""


def decide(state):
    """Approach opponent slowly in extended stance."""
    direction = state["opponent"]["direction"]

    if direction == 0:
        accel = 0.0  # Already on top of opponent
    else:
        accel = direction * 1.5  # Slow approach

    return {
        "acceleration": accel,
        "stance": "extended",
    }
