"""
Charges when opponent approaches.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Uses extended stance when opponent is close."""
    distance = abs(state["opponent"]["distance"])

    # Charge (extended stance) when close
    if distance < 2.0:
        stance = "extended"
    else:
        stance = "neutral"

    return {
        "acceleration": 0.0,  # Stationary
        "stance": stance
    }