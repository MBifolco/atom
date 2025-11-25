"""
Continuously moves backward away from opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Always moves away from the opponent."""
    # Determine direction away from opponent
    if state["opponent"]["distance"] > 0:
        # Opponent is to the right, move left
        accel = -2.0
    else:
        # Opponent is to the left, move right
        accel = 2.0

    # Defensive stance when backing away
    stance = "defending" if abs(state["opponent"]["distance"]) < 2.0 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }