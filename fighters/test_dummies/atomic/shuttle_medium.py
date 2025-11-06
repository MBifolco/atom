"""
Shuttle Medium

Test dummy that moves back and forth at medium speed between positions 3m and 9m.
Constant medium-speed movement pattern, neutral stance.

Purpose: Test predictable movement at medium speed,
velocity limits, and turn mechanics.
"""


def decide(snapshot):
    """
    Medium shuttle movement test dummy.

    Moves back and forth at speed 2.5 between bounds.
    """
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]

    # Shuttle bounds
    left_bound = 3.0
    right_bound = 9.0
    speed = 2.5

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