"""
HP Adaptive — three stateless bands based on HP comparison.
Presses advantage when ahead, retreats and defends when behind,
stays cautious in the dead zone.  No wall awareness.
Used for Level 5: Advanced training.
"""


def decide(state):
    """Adapt strategy based on relative HP advantage."""
    you = state["you"]
    opponent = state["opponent"]
    direction = opponent["direction"]
    distance = opponent["distance"]

    hp_diff = (you["hp"] / you["max_hp"]) - (opponent["hp"] / opponent["max_hp"])

    if hp_diff > 0.1:
        # Pressing advantage — attack
        accel = direction * 2.5
        stance = "extended"
    elif hp_diff < -0.1:
        # Protecting deficit — retreat and defend
        accel = -direction * 2.5
        stance = "defending"
    else:
        # Dead zone — cautious, maintain ~2m distance (target=2.0, tolerance=0.3)
        if distance > 2.3:
            accel = direction * 1.0
        elif distance < 1.7:
            accel = -direction * 2.0
        else:
            accel = 0.0
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
