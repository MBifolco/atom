"""
Pressure Fighter — simplified Slugger.
Always pushes toward the opponent at moderate speed, attacks when within
range, and only defends at critically low stamina.  Combines constant
pressure, distance-based attacking, and emergency stamina defense.
Used for Level 6: Bridge (simplified Expert) training.
"""


def decide(state):
    """Constant forward pressure, attack in range, defend when gassed."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]

    if stamina_pct < 0.15:
        # Critical stamina — defend briefly
        accel = direction * 0.6
        stance = "defending"
    elif distance < 2.0:
        # In range — attack while pressing
        accel = direction * 0.6
        stance = "extended"
    else:
        # Far — close distance in neutral
        accel = direction * 0.6
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
