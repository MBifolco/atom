"""
Balanced Stance Switcher - Balanced approach between offense and defense
"""

# Track state
_tick_count = 0

def decide(state):
    """Balanced stance switching based on situation."""
    global _tick_count
    you = state["you"]
    opponent = state["opponent"]

    _tick_count += 1

    # Balanced movement - close when far, maintain when near
    if opponent["distance"] > 2.0:
        acceleration = 0.6 * opponent["direction"]
    elif opponent["distance"] < 0.8:
        acceleration = -0.4 * opponent["direction"]
    else:
        acceleration = 0.2 * opponent["direction"]

    # Balanced stance pattern based on stamina and distance
    if you["stamina"] < 3.0:
        stance = "defending"  # Recover stamina
    elif opponent["distance"] < 1.0 and you["stamina"] > 5.0:
        stance = "extended"  # Attack when close and have stamina
    elif opponent["distance"] < 1.5 and _tick_count % 8 < 3:
        stance = "extended"  # Occasional jabs
    else:
        stance = "neutral"  # Default neutral

    return {
        "acceleration": acceleration,
        "stance": stance
    }