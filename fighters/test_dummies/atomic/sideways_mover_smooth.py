"""
Sideways Mover (Smooth) — position-based sine oscillation.
Acceleration is a smooth function of the fighter's own position,
creating natural back-and-forth movement.
Used for Level 3: Intermediate training.
"""

import math


def decide(state):
    """Smooth oscillating movement driven by own position."""
    position = state["you"]["position"]
    distance = state["opponent"]["distance"]

    accel = 2.0 * math.sin(position * 1.5)

    if distance < 1.0:
        stance = "defending"
    elif distance < 2.0:
        stance = "extended"
    else:
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
