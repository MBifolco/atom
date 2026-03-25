"""
Backward Mover — continuously moves away from opponent at moderate speed.
Uses direction for movement, not absolute distance.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Always moves away from the opponent."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    accel = -direction * 2.0
    stance = "defending" if distance < 2.0 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
