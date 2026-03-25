"""
Stationary opponent that defends when you approach.
Used for Level 3: Intermediate training.

Teaches the AI that approaching isn't always free — sometimes the opponent
will guard, and the AI needs to time attacks or outlast the defense.
Simpler than charge_on_approach (which punishes with damage).
"""


def decide(state):
    """Switches to defending stance when opponent is close."""
    distance = abs(state["opponent"]["distance"])

    if distance < 1.5:
        stance = "defending"
    else:
        stance = "neutral"

    return {
        "acceleration": 0.0,
        "stance": stance,
    }
