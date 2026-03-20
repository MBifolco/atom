"""
Shuttles back and forth at medium speed.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Moves back and forth at medium speed."""
    my_pos = state["you"]["position"]
    my_vel = state["you"]["velocity"]
    arena_width = state["arena"]["width"]

    # Bounce off walls at higher speed
    if my_pos < 2.0:
        accel = 1.8  # Near left wall, go right
    elif my_pos > arena_width - 2.0:
        accel = -1.8  # Near right wall, go left
    elif my_vel > 0:
        accel = 1.8  # Continue right
    else:
        accel = -1.8  # Continue left

    return {
        "acceleration": accel,
        "stance": "neutral"
    }