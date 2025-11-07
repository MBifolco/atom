"""
Shuttle Slow

Test dummy that moves back and forth slowly between positions 3m and 9m.
Constant slow movement pattern, neutral stance.

Purpose: Test predictable movement, low-speed physics,
and basic position tracking.
"""


def decide(snapshot):
    """
    Slow shuttle movement test dummy.

    Moves back and forth at speed 1.0 between bounds.
    """
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]

    # Shuttle bounds
    left_bound = 3.0
    right_bound = 9.0
    speed = 1.0

    # Determine acceleration
    if my_position <= left_bound:
        acceleration = speed  # Move right
    elif my_position >= right_bound:
        acceleration = -speed  # Move left
    elif my_velocity > 0:  # Moving right
        if my_position >= right_bound - 0.5:
            acceleration = -speed  # Start turning
        else:
            acceleration = speed  # Continue right
    else:  # Moving left
        if my_position <= left_bound + 0.5:
            acceleration = speed  # Start turning
        else:
            acceleration = -speed  # Continue left

    return {"acceleration": acceleration, "stance": "neutral"}