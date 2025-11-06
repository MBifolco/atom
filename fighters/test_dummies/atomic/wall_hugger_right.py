"""
Wall Hugger Right

Test dummy that moves to the right wall and stays there.
Uses defending stance to minimize wall damage.

Purpose: Test wall damage mechanics, wall physics,
and how fighters handle wall-stuck opponents.
"""


def decide(snapshot):
    """
    Right wall hugger test dummy.

    Moves to right wall and stays there in defending stance.
    """
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Move to right wall
    if my_position < arena_width - 0.5:
        acceleration = 3.0  # Move to wall
    else:
        acceleration = 0.5  # Gentle pressure against wall

    # Use defending stance to minimize wall damage
    return {"acceleration": acceleration, "stance": "defending"}