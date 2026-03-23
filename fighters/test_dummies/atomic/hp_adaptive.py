"""
Three stateless bands based on HP comparison.
Presses advantage when ahead, retreats and defends when behind,
stays cautious in the dead zone. Wall-aware when retreating.
Used for Level 5: Advanced training.
"""


def decide(state):
    """Adapt strategy based on relative HP advantage."""
    you = state["you"]
    opponent = state["opponent"]
    direction = opponent["direction"]
    distance = opponent["distance"]
    position = you["position"]
    arena_width = state["arena"]["width"]

    hp_diff = (you["hp"] / you["max_hp"]) - (opponent["hp"] / opponent["max_hp"])

    if direction == 0:
        direction = 1.0  # Default direction if overlapping

    if hp_diff > 0.1:
        # Pressing advantage -- attack
        accel = direction * 3.0
        stance = "extended"
    elif hp_diff < -0.1:
        # Protecting deficit -- retreat and defend
        accel = -direction * 2.0
        stance = "defending"
        # Wall awareness
        if position < 1.0:
            accel = 2.0
        elif position > arena_width - 1.0:
            accel = -2.0
    else:
        # Dead zone -- cautious, maintain ~2m distance
        if distance > 2.5:
            accel = direction * 1.0
        elif distance < 1.5:
            accel = -direction * 1.0
        else:
            accel = 0.0
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
