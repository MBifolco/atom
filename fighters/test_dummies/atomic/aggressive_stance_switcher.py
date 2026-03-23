"""
Aggressive Stance Switcher — rapidly switches between extended and
neutral stances on a 10-tick cycle while closing distance.
Completely stateless: uses state["tick"] for the cycle.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Aggressively switch stances while closing distance."""
    tick = state["tick"]
    direction = state["opponent"]["direction"]
    stamina = state["you"]["stamina"]

    # Move toward opponent at near-max acceleration
    accel = direction * 0.9 * 4.0

    # Switch between aggressive stances rapidly
    if tick % 10 < 5:
        stance = "extended"
    else:
        stance = "neutral"

    # Override to defend only if very low stamina
    if stamina < 2.0:
        stance = "defending"

    return {
        "acceleration": accel,
        "stance": stance,
    }
