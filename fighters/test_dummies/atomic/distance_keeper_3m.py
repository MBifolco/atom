"""
Distance Keeper 3m

Test dummy that maintains exactly 3 meter distance from opponent.
Uses moderate acceleration to stay at mid-range.

Purpose: Test mid-range combat, spacing control,
and transition between engagement ranges.
"""


def decide(snapshot):
    """
    3m distance keeper test dummy.

    Maintains 3 meter distance from opponent using controlled movement.
    """
    my_position = snapshot["you"]["position"]
    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)

    target_distance = 3.0
    tolerance = 0.3

    if distance > target_distance + tolerance:
        # Too far: Approach
        if opponent_position > my_position:
            acceleration = 2.5  # Move right
        else:
            acceleration = -2.5  # Move left
    elif distance < target_distance - tolerance:
        # Too close: Back away
        if opponent_position > my_position:
            acceleration = -2.5  # Move left
        else:
            acceleration = 2.5  # Move right
    else:
        # Perfect distance: Maintain
        acceleration = 0.0

    # Use neutral stance at mid-range
    stance = "neutral"

    return {"acceleration": acceleration, "stance": stance}