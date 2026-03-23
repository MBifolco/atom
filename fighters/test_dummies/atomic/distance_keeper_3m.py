"""
Maintains 3m distance from opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Maintains 3m distance for defensive play."""
    distance = state["opponent"]["distance"]
    direction = state["opponent"]["direction"]
    target_distance = 3.0
    tolerance = 0.3

    if distance > target_distance + tolerance:
        # Too far, approach slowly
        accel = 1.0 * direction
    elif distance < target_distance - tolerance:
        # Too close, back away
        accel = -2.0 * direction
    else:
        # Good distance
        accel = 0.0

    # Use neutral stance at this range
    stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }