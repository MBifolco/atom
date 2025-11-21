"""
Maintains 1m distance from opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Maintains optimal striking distance."""
    distance = abs(state["opponent"]["distance"])
    target_distance = 1.0
    tolerance = 0.2

    if distance > target_distance + tolerance:
        # Too far, approach
        if state["opponent"]["distance"] > 0:
            accel = 2.0  # Opponent is to right
        else:
            accel = -2.0  # Opponent is to left
    elif distance < target_distance - tolerance:
        # Too close, back away
        if state["opponent"]["distance"] > 0:
            accel = -2.0  # Back away left
        else:
            accel = 2.0  # Back away right
    else:
        # Good distance
        accel = 0.0

    # Use extended stance at optimal range
    stance = "extended" if abs(distance - target_distance) < tolerance else "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }