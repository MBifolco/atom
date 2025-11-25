"""
Continuously moves forward toward opponent.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Always moves toward the opponent."""
    # Determine direction to opponent
    if state["opponent"]["distance"] > 0:
        # Opponent is to the right, move right
        accel = 2.0
    else:
        # Opponent is to the left, move left
        accel = -2.0

    # Aggressive stance when moving forward
    stance = "extended" if abs(state["opponent"]["distance"]) < 1.5 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }