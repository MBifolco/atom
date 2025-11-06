"""
Flee Always

Test dummy that always flees from the opponent.
Constant retreat, neutral stance.

Purpose: Test evasion behavior, extended matches,
and how fighters handle opponents that won't engage.
"""


def decide(snapshot):
    """
    Always flee test dummy.

    Always moves away from opponent at speed 3.0.
    """
    my_position = snapshot["you"]["position"]
    opp_position = snapshot["opponent"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Always move away from opponent
    if my_position < opp_position:
        acceleration = -3.0  # Move left away from opponent
    elif my_position > opp_position:
        acceleration = 3.0  # Move right away from opponent
    else:
        acceleration = 3.0  # Emergency escape

    # Don't flee into walls
    if my_position < 2.0 and acceleration < 0:
        acceleration = 3.0  # Don't flee into left wall
    elif my_position > arena_width - 2.0 and acceleration > 0:
        acceleration = -3.0  # Don't flee into right wall

    return {"acceleration": acceleration, "stance": "neutral"}