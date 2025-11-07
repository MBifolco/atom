"""
Counter Movement

Test dummy that moves opposite to the opponent's movement.
If opponent moves left, this dummy moves right. If opponent moves right,
this dummy moves left.

Purpose: Test against evasive movement, flanking attempts,
and how fighters handle opponents that actively avoid alignment.
"""


def decide(snapshot):
    """
    Counter movement test dummy.

    Moves in opposite direction to opponent's movement.
    """
    opponent_velocity = snapshot["opponent"]["velocity"]

    # Counter the opponent's velocity
    if opponent_velocity > 0.5:
        # Opponent moving right: Move left
        acceleration = -3.0
    elif opponent_velocity < -0.5:
        # Opponent moving left: Move right
        acceleration = 3.0
    else:
        # Opponent stationary: Stay still
        acceleration = 0.0

    # Neutral stance for mobility
    return {"acceleration": acceleration, "stance": "neutral"}