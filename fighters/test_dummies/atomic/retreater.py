"""
Retreater test dummy - Always moves away from opponent
"""

def decide(state):
    """Always move away from opponent."""
    you = state["you"]
    opponent = state["opponent"]

    # Always move away from opponent
    acceleration = -0.8 * opponent["direction"]

    # Defend when too close
    if opponent["distance"] < 1.0:
        stance = "defending"
    else:
        stance = "neutral"

    return {
        "acceleration": acceleration,
        "stance": stance
    }