"""
Wall Hugger Left

Test dummy that moves to the left wall and stays there.
Uses defending stance to minimize wall damage.

Purpose: Test wall damage mechanics, wall physics,
and how fighters handle wall-stuck opponents.
"""


def decide(snapshot):
    """
    Left wall hugger test dummy.

    Moves to left wall and stays there in defending stance.
    """
    my_position = snapshot["you"]["position"]

    # Move to left wall
    if my_position > 0.5:
        acceleration = -3.0  # Move to wall
    else:
        acceleration = -0.5  # Gentle pressure against wall

    # Use defending stance to minimize wall damage
    return {"acceleration": acceleration, "stance": "defending"}