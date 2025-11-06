"""
Circle Right

Test dummy that always moves right at constant speed.
Will bounce off right wall and continue circling.

Purpose: Test wall collision mechanics on right side,
constant velocity movement, and wall damage.
"""


def decide(snapshot):
    """
    Circle right test dummy.

    Always accelerates right at speed 2.0, bounces off walls.
    """
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Default: move right
    acceleration = 2.0

    # Bounce off right wall
    if my_position > arena_width - 1.0:
        acceleration = -2.0  # Bounce left

    return {"acceleration": acceleration, "stance": "neutral"}