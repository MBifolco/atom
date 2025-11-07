"""
Mirror Movement

Test dummy that mirrors the opponent's movement direction.
If opponent moves left, this dummy moves left. If opponent moves right,
this dummy moves right.

Purpose: Test against synchronized movement, formation fighting,
and how fighters handle opponents that match their positioning.
"""


def decide(snapshot):
    """
    Mirror movement test dummy.

    Copies opponent's movement direction with matching intensity.
    """
    opponent_velocity = snapshot["opponent"]["velocity"]

    # Mirror the opponent's velocity
    if opponent_velocity > 0.5:
        # Opponent moving right: Move right too
        acceleration = 3.0
    elif opponent_velocity < -0.5:
        # Opponent moving left: Move left too
        acceleration = -3.0
    else:
        # Opponent stationary: Stay still
        acceleration = 0.0

    # Neutral stance for mobility
    return {"acceleration": acceleration, "stance": "neutral"}