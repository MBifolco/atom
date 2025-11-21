"""
Always circles left.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Always moves left, bouncing off walls."""
    my_pos = state["you"]["position"]

    # Bounce off left wall
    if my_pos < 1.0:
        accel = 2.0  # Bounce right
    else:
        accel = -2.0  # Default left

    return {
        "acceleration": accel,
        "stance": "neutral"
    }