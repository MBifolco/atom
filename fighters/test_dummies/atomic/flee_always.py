"""
Always flees from the opponent.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Moves away from opponent."""
    my_pos = state["you"]["position"]
    arena_width = state["arena"]["width"]

    # Use distance to determine flee direction
    distance = state["opponent"]["distance"]

    # If opponent is to the right (positive distance), flee left
    # If opponent is to the left (negative distance), flee right
    if distance > 0:
        # Opponent is to right, flee left
        accel = -1.5
    else:
        # Opponent is to left, flee right
        accel = 1.5

    # Don't flee into walls
    if my_pos < 1.0:
        accel = 1.5  # Near left wall, go right
    elif my_pos > arena_width - 1.0:
        accel = -1.5  # Near right wall, go left

    return {
        "acceleration": accel,
        "stance": "neutral"
    }