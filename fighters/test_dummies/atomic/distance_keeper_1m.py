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
    my_velocity = snapshot["you"]["velocity"]
    distance = snapshot["opponent"]["distance"]
    arena_width = snapshot["arena"]["width"]

    target_distance = 1.0
    tolerance = 0.2

    # Determine opponent direction using position heuristics
    if my_position < arena_width * 0.3:
        # We're on left side, opponent likely to the right
        opponent_to_right = True
    elif my_position > arena_width * 0.7:
        # We're on right side, opponent likely to the left
        opponent_to_right = False
    else:
        # In middle - use velocity to maintain direction
        opponent_to_right = my_velocity >= 0

    if distance > target_distance + tolerance:
        # Too far: Approach
        acceleration = 2.0 if opponent_to_right else -2.0
    elif distance < target_distance - tolerance:
        # Too close: Back away
        acceleration = -2.0 if opponent_to_right else 2.0
    else:
        # Perfect distance: Maintain
        acceleration = 0.0

    # Use extended stance at optimal range
    stance = "extended" if abs(distance - target_distance) < tolerance else "neutral"

    return {"acceleration": acceleration, "stance": stance}