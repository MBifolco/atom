"""
Defensive Stance Switcher - Focuses on defense with occasional counters
"""

# Track state
_tick_count = 0

def decide(state):
    """Defensively switch stances, focusing on blocking."""
    global _tick_count
    you = state["you"]
    opponent = state["opponent"]

    _tick_count += 1

    # Maintain distance defensively
    if opponent["distance"] < 1.0:
        acceleration = -0.5 * opponent["direction"]  # Back away
    else:
        acceleration = 0.0  # Hold position

    # Mostly defend, occasionally counter
    if _tick_count % 20 < 15:
        stance = "defending"
    elif opponent["distance"] < 1.2:
        stance = "extended"  # Quick counter
    else:
        stance = "neutral"

    return {
        "acceleration": acceleration,
        "stance": stance
    }