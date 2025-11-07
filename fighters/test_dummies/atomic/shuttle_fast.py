"""
Shuttle Fast

Test dummy that moves back and forth at maximum speed between positions 3m and 9m.
Constant fast movement pattern, neutral stance.

Purpose: Test maximum velocity, acceleration limits,
physics at high speed, and rapid direction changes.
"""


def decide(snapshot):
    """
    Fast shuttle movement test dummy.

    Moves back and forth at speed 4.0 between bounds.
    """
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]

    # Shuttle bounds
    left_bound = 3.0
    right_bound = 9.0
    speed = 4.0

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