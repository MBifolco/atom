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
    my_velocity = snapshot["you"]["velocity"]
    opponent_distance = snapshot["opponent"]["distance"]
    opponent_rel_velocity = snapshot["opponent"]["velocity"]
    arena_width = snapshot["arena"]["width"]

    # Determine opponent direction based on our position
    # If we're close to left wall, opponent is probably to the right
    # If we're close to right wall, opponent is probably to the left
    # Otherwise, use velocity hints

    if opponent_distance < 0.1:
        # Very close, just stand still
        acceleration = 0.0
    elif my_position < arena_width * 0.3:
        # We're on left side, opponent likely to the right
        acceleration = 1.5
    elif my_position > arena_width * 0.7:
        # We're on right side, opponent likely to the left
        acceleration = -1.5
    else:
        # In middle - use relative velocity to guess direction
        # If rel_velocity is negative, we're approaching (moving toward each other)
        # If we're moving right and approaching, opponent is to the right
        # If we're moving left and approaching, opponent is to the left
        if my_velocity > 0:
            acceleration = 1.5  # Continue right
        elif my_velocity < 0:
            acceleration = -1.5  # Continue left
        else:
            # Stopped - default to moving right
            acceleration = 1.5

    return {"acceleration": acceleration, "stance": "neutral"}