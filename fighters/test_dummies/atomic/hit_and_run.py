"""
Hit and Run — simplified Out-Fighter.
Approaches until within striking distance, attacks, then retreats once
too close.  Alternates between approach and retreat based purely on distance.
Combines approach/retreat transitions and range-based stance switching.
Used for Level 6: Bridge (simplified Expert) training.
"""


def decide(state):
    """Approach to strike, then retreat when too close."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    if distance > 1.5:
        # Far — approach opponent
        accel = direction * 3.0
        stance = "neutral"
    elif distance > 1.0:
        # Striking range — attack
        accel = direction * 0.5
        stance = "extended"
    else:
        # Too close — retreat
        accel = -direction * 3.0
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
