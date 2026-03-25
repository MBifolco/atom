"""
Wait and Counter — simplified Counter Puncher.
Holds a defending stance by default and counters with a short step forward
when the opponent enters close range.  Stops countering when stamina is low.
Combines defending, reactive attacking, and stamina gating.
Used for Level 6: Bridge (simplified Expert) training.
"""


def decide(state):
    """Defend by default, counter when opponent is close."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]

    if stamina_pct < 0.20:
        # Too tired to counter — turtle up
        accel = 0.0
        stance = "defending"
    elif distance < 1.0:
        # Opponent in counter range — strike and step in
        accel = direction * 1.0
        stance = "extended"
    else:
        # Default — hold ground and defend
        accel = 0.0
        stance = "defending"

    return {
        "acceleration": accel,
        "stance": stance,
    }
