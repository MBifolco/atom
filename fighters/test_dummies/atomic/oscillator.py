"""
Oscillator test dummy - Moves back and forth in a pattern
"""

import math

# Track state between calls
_tick_count = 0

def decide(state):
    """Oscillate position using sine wave."""
    global _tick_count
    you = state["you"]
    opponent = state["opponent"]

    # Increment tick counter
    _tick_count += 1

    # Oscillate using sine wave
    acceleration = 0.7 * math.sin(_tick_count * 0.1)

    # Vary stance based on distance
    if opponent["distance"] < 0.8:
        stance = "defending"
    elif opponent["distance"] < 1.5:
        stance = "extended"
    else:
        stance = "neutral"

    return {
        "acceleration": acceleration,
        "stance": stance
    }