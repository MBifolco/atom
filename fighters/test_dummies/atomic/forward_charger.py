"""
Forward Charger — always charges toward opponent at maximum acceleration.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Always charge forward at max acceleration."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    accel = direction * 4.0
    stance = "extended" if distance < 1.5 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
