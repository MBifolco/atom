"""
Shuttles back and forth slowly.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Moves back and forth at slow speed."""
    my_pos = state["you"]["position"]
    my_vel = state["you"]["velocity"]
    arena_width = state["arena"]["width"]

    # Bounce off walls
    if my_pos < 2.0:
        accel = 1.0  # Near left wall, go right
    elif my_pos > arena_width - 2.0:
        accel = -1.0  # Near right wall, go left
    elif my_vel > 0:
        accel = 1.0  # Continue right
    else:
        accel = -1.0  # Continue left

    return {
        "acceleration": accel,
        "stance": "neutral"
    }