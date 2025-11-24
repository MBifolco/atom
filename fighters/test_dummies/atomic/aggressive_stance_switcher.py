"""
Aggressive Stance Switcher - Rapidly switches between attacking stances
"""

# Track state
_tick_count = 0

def decide(state):
    """Aggressively switch stances while closing distance."""
    global _tick_count
    you = state["you"]
    opponent = state["opponent"]

    _tick_count += 1

    # Move toward opponent aggressively
    acceleration = 0.9 * opponent["direction"]

    # Switch between aggressive stances rapidly
    if _tick_count % 10 < 5:
        stance = "extended"
    else:
        stance = "neutral"  # Reset to neutral to prepare next attack

    # Override to defend only if very low stamina
    if you["stamina"] < 2.0:
        stance = "defending"

    return {
        "acceleration": acceleration,
        "stance": stance
    }