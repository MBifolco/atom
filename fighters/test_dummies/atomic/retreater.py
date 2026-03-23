"""
Retreater — always moves away from opponent with wall awareness.
Bounces off walls to avoid getting cornered.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Always move away from opponent, with wall bounce."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    position = state["you"]["position"]
    arena_width = state["arena"]["width"]

    accel = -direction * 0.8

    # Wall bounce — reverse if near arena edges
    if position < 1.0:
        accel = abs(accel)
    elif position > arena_width - 1.0:
        accel = -abs(accel)

    stance = "defending" if distance < 1.0 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
