"""
Always flees from the opponent.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Moves away from opponent."""
    my_pos = state["you"]["position"]
    arena_width = state["arena"]["width"]
    direction = state["opponent"]["direction"]

    # Flee away from opponent
    if direction > 0:
        accel = -1.5  # Opponent to right, flee left
    elif direction < 0:
        accel = 1.5   # Opponent to left, flee right
    else:
        accel = 1.5   # Overlap, default flee right

    # Don't flee into walls
    if my_pos < 1.0:
        accel = 1.5  # Near left wall, go right
    elif my_pos > arena_width - 1.0:
        accel = -1.5  # Near right wall, go left

    return {
        "acceleration": accel,
        "stance": "neutral"
    }