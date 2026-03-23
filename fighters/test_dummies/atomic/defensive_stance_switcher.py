"""
Defensive Stance Switcher — focuses on defense with occasional counters.
Mostly defending on a 20-tick cycle (15 defend, 5 counter).
Approaches only when opponent is close; holds position otherwise.
Completely stateless: uses state["tick"] for the cycle.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Defensively switch stances, focusing on blocking."""
    tick = state["tick"]
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    # Approach when close, hold otherwise
    if distance < 2.0:
        accel = direction * 1.5
    else:
        accel = 0.0

    # Mostly defend, occasionally counter
    if tick % 20 < 15:
        stance = "defending"
    else:
        if distance < 2.0:
            stance = "extended"
        else:
            stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
