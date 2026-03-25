"""
Maintains 1m distance from opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Maintains optimal striking distance."""
    distance = state["opponent"]["distance"]
    direction = state["opponent"]["direction"]
    target_distance = 1.0
    tolerance = 0.2

    if distance > target_distance + tolerance:
        # Too far, approach
        accel = 2.0 * direction
    elif distance < target_distance - tolerance:
        # Too close, back away
        accel = -2.0 * direction
    else:
        # Good distance
        accel = 0.0

    # Use extended stance at optimal range
    stance = "extended" if abs(distance - target_distance) < tolerance else "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }