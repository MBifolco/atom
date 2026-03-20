"""
Strategic retreating behavior with stance management.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Strategically retreats while managing stamina and defense."""
    distance = abs(state["opponent"]["distance"])

    # Strategic retreat thresholds
    danger_zone = 1.0
    safe_zone = 3.0

    if distance < danger_zone:
        # Too close - retreat quickly
        if state["opponent"]["distance"] > 0:
            accel = -3.0  # Quick retreat left
        else:
            accel = 3.0  # Quick retreat right
        stance = "defending"  # Defensive when close

    elif distance < safe_zone:
        # Medium range - controlled retreat
        if state["opponent"]["distance"] > 0:
            accel = -1.5  # Moderate retreat left
        else:
            accel = 1.5  # Moderate retreat right
        stance = "neutral"  # Neutral at medium range

    else:
        # Safe distance - can stop or slowly approach
        accel = 0.0
        stance = "extended"  # Can be aggressive at safe distance

    return {
        "acceleration": accel,
        "stance": stance
    }