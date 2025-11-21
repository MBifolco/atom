"""
Always circles right.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Always moves right, bouncing off walls."""
    my_pos = state["you"]["position"]
    arena_width = state["arena"]["width"]

    # Bounce off right wall
    if my_pos > arena_width - 1.0:
        accel = -2.0  # Bounce left
    else:
        accel = 2.0  # Default right

    return {
        "acceleration": accel,
        "stance": "neutral"
    }