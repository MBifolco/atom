"""
Forward Mover — continuously moves toward opponent at moderate speed.
Uses direction for movement, not absolute distance.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Always moves toward the opponent."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    accel = direction * 2.0
    stance = "extended" if distance < 1.5 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
