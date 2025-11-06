"""
Approach Fast

Test dummy that always moves quickly toward the opponent.
Constant fast pursuit, neutral stance.

Purpose: Test aggressive pursuit, high-speed collision initiation,
and how fighters handle rapid approach.
"""


def decide(snapshot):
    """
    Fast approach test dummy.

    Always moves toward opponent at speed 4.0.
    """
    my_position = snapshot["you"]["position"]
    opp_position = snapshot["opponent"]["position"]

    # Always move toward opponent aggressively
    if my_position < opp_position:
        acceleration = 4.0  # Move right toward opponent
    elif my_position > opp_position:
        acceleration = -4.0  # Move left toward opponent
    else:
        acceleration = 0.5  # Slight forward pressure when at opponent

    return {"acceleration": acceleration, "stance": "neutral"}