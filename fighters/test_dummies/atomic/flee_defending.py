"""
Always flees in defending stance with wall awareness.
Teaches the agent to chase down and pressure a retreating defender.
Used for Level 2: Basic Skills training.
"""


def decide(state):
    """Flee from opponent in defending stance, bouncing off walls."""
    direction = state["opponent"]["direction"]
    position = state["you"]["position"]
    arena_width = state["arena"]["width"]

    # Flee away from opponent
    if direction == 0:
        accel = 1.0  # Pick a direction if overlapping
    else:
        accel = -direction * 2.0

    # Wall awareness -- reverse if about to hit a wall
    if position < 1.0:
        accel = 2.0
    elif position > arena_width - 1.0:
        accel = -2.0

    return {
        "acceleration": accel,
        "stance": "defending",
    }
