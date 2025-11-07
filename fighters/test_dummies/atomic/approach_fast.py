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
    my_velocity = snapshot["you"]["velocity"]
    opponent_distance = snapshot["opponent"]["distance"]
    opponent_rel_velocity = snapshot["opponent"]["velocity"]
    arena_width = snapshot["arena"]["width"]

    # Determine opponent direction based on our position
    # If we're close to left wall, opponent is probably to the right
    # If we're close to right wall, opponent is probably to the left
    # Otherwise, use velocity hints

    if opponent_distance < 0.1:
        # Very close, slight forward pressure
        acceleration = 0.5 if my_velocity >= 0 else -0.5
    elif my_position < arena_width * 0.3:
        # We're on left side, opponent likely to the right
        acceleration = 4.0
    elif my_position > arena_width * 0.7:
        # We're on right side, opponent likely to the left
        acceleration = -4.0
    else:
        # In middle - use relative velocity to guess direction
        if my_velocity > 0:
            acceleration = 4.0  # Continue right aggressively
        elif my_velocity < 0:
            acceleration = -4.0  # Continue left aggressively
        else:
            # Stopped - default to moving right
            acceleration = 4.0

    return {"acceleration": acceleration, "stance": "neutral"}