"""
Forward Charger test dummy - Always moves forward aggressively
"""

def decide(state):
    """Always charge forward in extended stance."""
    you = state["you"]
    opponent = state["opponent"]

    # Always move toward opponent aggressively
    acceleration = 1.0 * opponent["direction"]

    # Always attack when in range
    stance = "extended" if opponent["distance"] < 1.5 else "neutral"

    return {
        "acceleration": acceleration,
        "stance": stance
    }