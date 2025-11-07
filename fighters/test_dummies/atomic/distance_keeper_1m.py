"""
Distance Keeper 1m

Test dummy that maintains exactly 1 meter distance from opponent.
Uses precise acceleration control to stay at this range.

Purpose: Test close-range combat, in-fighting mechanics,
and how fighters handle persistent close pressure.
"""


def decide(snapshot):
    """
    1m distance keeper test dummy.

    Maintains 1 meter distance from opponent using precise movement.
    """
    my_position = snapshot["you"]["position"]
    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)

    target_distance = 1.0
    tolerance = 0.2

    if distance > target_distance + tolerance:
        # Too far: Approach
        if opponent_position > my_position:
            acceleration = 2.0  # Move right
        else:
            acceleration = -2.0  # Move left
    elif distance < target_distance - tolerance:
        # Too close: Back away
        if opponent_position > my_position:
            acceleration = -2.0  # Move left
        else:
            acceleration = 2.0  # Move right
    else:
        # Perfect distance: Maintain
        acceleration = 0.0

    # Use extended stance at optimal range
    stance = "extended" if abs(distance - target_distance) < tolerance else "neutral"

    return {"acceleration": acceleration, "stance": stance}