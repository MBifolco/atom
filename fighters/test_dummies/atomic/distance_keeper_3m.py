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
    my_velocity = snapshot["you"]["velocity"]
    distance = snapshot["opponent"]["distance"]
    arena_width = snapshot["arena"]["width"]

    target_distance = 3.0
    tolerance = 0.3

    # Determine opponent direction using position heuristics
    if my_position < arena_width * 0.3:
        opponent_to_right = True
    elif my_position > arena_width * 0.7:
        opponent_to_right = False
    else:
        opponent_to_right = my_velocity >= 0

    if distance > target_distance + tolerance:
        # Too far: Approach
        acceleration = 2.5 if opponent_to_right else -2.5
    elif distance < target_distance - tolerance:
        # Too close: Back away
        acceleration = -2.5 if opponent_to_right else 2.5
    else:
        # Perfect distance: Maintain
        acceleration = 0.0

    # Use neutral stance at mid-range
    stance = "neutral"

    return {"acceleration": acceleration, "stance": stance}