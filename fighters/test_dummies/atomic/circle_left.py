"""
Circle Left

Test dummy that always moves left at constant speed.
Will bounce off left wall and continue circling.

Purpose: Test wall collision mechanics on left side,
constant velocity movement, and wall damage.
"""


def decide(snapshot):
    """
    Circle left test dummy.

    Always accelerates left at speed 2.0, bounces off walls.
    """
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Default: move left
    acceleration = -2.0

    # Bounce off left wall
    if my_position < 1.0:
        acceleration = 2.0  # Bounce right

    return {"acceleration": acceleration, "stance": "neutral"}