"""
Maintains 3m distance from opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Maintains 3m distance for defensive play."""
    distance = abs(state["opponent"]["distance"])
    target_distance = 3.0
    tolerance = 0.3

    if distance > target_distance + tolerance:
        # Too far, approach slowly
        if state["opponent"]["distance"] > 0:
            accel = 1.0  # Opponent is to right
        else:
            accel = -1.0  # Opponent is to left
    elif distance < target_distance - tolerance:
        # Too close, back away
        if state["opponent"]["distance"] > 0:
            accel = -2.0  # Back away left
        else:
            accel = 2.0  # Back away right
    else:
        # Good distance
        accel = 0.0

    # Use neutral stance at this range
    stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }