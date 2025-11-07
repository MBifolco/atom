"""
Flee Always

Test dummy that always flees from the opponent.
Constant retreat, neutral stance.

Purpose: Test evasion behavior, extended matches,
and how fighters handle opponents that won't engage.
"""


def decide(snapshot):
    """
    Always flee test dummy.

    Always moves away from opponent at speed 3.0.
    """
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]
    opponent_distance = snapshot["opponent"]["distance"]
    arena_width = snapshot["arena"]["width"]

    # Determine opponent direction and flee opposite
    # If we're close to left wall, opponent is probably to the right - flee left
    # If we're close to right wall, opponent is probably to the left - flee right
    # Otherwise, use velocity hints

    if my_position < arena_width * 0.3:
        # We're on left side, opponent likely to the right - flee left
        acceleration = -3.0
    elif my_position > arena_width * 0.7:
        # We're on right side, opponent likely to the left - flee right
        acceleration = 3.0
    else:
        # In middle - flee opposite to current velocity direction
        # (reverses direction to get away)
        if my_velocity > 0:
            acceleration = -3.0  # Was moving right, flee left
        elif my_velocity < 0:
            acceleration = 3.0  # Was moving left, flee right
        else:
            # Stopped - default to fleeing right
            acceleration = 3.0

    # Don't flee into walls
    if my_position < 2.0 and acceleration < 0:
        acceleration = 3.0  # Don't flee into left wall
    elif my_position > arena_width - 2.0 and acceleration > 0:
        acceleration = -3.0  # Don't flee into right wall

    return {"acceleration": acceleration, "stance": "neutral"}