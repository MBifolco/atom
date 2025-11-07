"""
Distance Keeper 5m

Test dummy that maintains exactly 5 meter distance from opponent.
Uses strong acceleration to stay at long range.

Purpose: Test long-range combat, zoning strategies,
and pursuit/evasion at distance.
"""


def decide(snapshot):
    """
    5m distance keeper test dummy.

    Maintains 5 meter distance from opponent using strong movement.
    """
    my_position = snapshot["you"]["position"]
    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)

    target_distance = 5.0
    tolerance = 0.4

    if distance > target_distance + tolerance:
        # Too far: Approach slightly
        if opponent_position > my_position:
            acceleration = 1.5  # Move right slowly
        else:
            acceleration = -1.5  # Move left slowly
    elif distance < target_distance - tolerance:
        # Too close: Back away quickly
        if opponent_position > my_position:
            acceleration = -3.5  # Move left fast
        else:
            acceleration = 3.5  # Move right fast
    else:
        # Perfect distance: Maintain
        acceleration = 0.0

    # Use defending stance at long range
    stance = "defending"

    return {"acceleration": acceleration, "stance": stance}