"""
Strategic Retreater — retreats from close range, holds at mid range,
and slowly approaches when far.  Three distance zones with stance
management.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Strategically retreat with distance-based zones."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    if distance < 1.0:
        # Danger zone — retreat quickly
        accel = -direction * 3.0
        stance = "defending"
    elif distance < 3.0:
        # Mid range — controlled retreat
        accel = -direction * 1.5
        stance = "neutral"
    else:
        # Far — slowly approach
        accel = direction * 0.5
        stance = "extended"

    return {
        "acceleration": accel,
        "stance": stance,
    }
