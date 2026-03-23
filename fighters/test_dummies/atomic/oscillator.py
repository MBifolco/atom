"""
Oscillator — moves back and forth using a sine wave driven by tick count.
Completely stateless: uses state["tick"] for the oscillation phase.
Used for Level 3: Intermediate training.
"""

import math


def decide(state):
    """Oscillate position using sine wave."""
    tick = state["tick"]
    distance = state["opponent"]["distance"]

    accel = 4.0 * math.sin(tick * 0.15)

    if distance < 0.8:
        stance = "defending"
    elif distance < 1.5:
        stance = "extended"
    else:
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
