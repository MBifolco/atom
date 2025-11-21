"""
Slowly approaches the opponent.
Used for Level 2: Basic Skills training.
"""

def decide(state):
    """Moves toward opponent at slow speed."""
    my_pos = state["you"]["position"]
    opp_pos = state["opponent"]["position"] if "position" in state["opponent"] else None

    if opp_pos is None:
        # Use distance to estimate opponent position
        distance = state["opponent"]["distance"]
        my_vel = state["you"]["velocity"]

        # If distance is positive, opponent is to the right
        if distance > 0:
            accel = 1.5  # Move right
        else:
            accel = -1.5  # Move left
    else:
        # Direct position available
        if opp_pos > my_pos:
            accel = 1.5  # Move right toward opponent
        else:
            accel = -1.5  # Move left toward opponent

    return {
        "acceleration": accel,
        "stance": "neutral"
    }