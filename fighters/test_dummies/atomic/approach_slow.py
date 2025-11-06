"""
Approach Slow

Test dummy that always moves slowly toward the opponent.
Constant slow pursuit, neutral stance.

Purpose: Test pursuit behavior, collision initiation at low speed,
and how fighters handle constant approach.
"""


def decide(snapshot):
    """
    Slow approach test dummy.

    Always moves toward opponent at speed 1.5.
    """
    my_position = snapshot["you"]["position"]
    opp_position = snapshot["opponent"]["position"]

    # Always move toward opponent
    if my_position < opp_position:
        acceleration = 1.5  # Move right toward opponent
    elif my_position > opp_position:
        acceleration = -1.5  # Move left toward opponent
    else:
        acceleration = 0.0  # At opponent position

    return {"acceleration": acceleration, "stance": "neutral"}