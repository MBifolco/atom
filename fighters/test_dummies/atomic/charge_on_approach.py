"""
Charge On Approach

Test dummy that remains stationary until opponent comes within 4 meters,
then charges forward with extended stance.

Purpose: Test reaction to aggressive charges, counter-charge mechanics,
and sudden engagement transitions.
"""


def decide(snapshot):
    """
    Charge on approach test dummy.

    Stationary until opponent < 4m, then charges with extended stance.
    """
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]
    opponent_distance = snapshot["opponent"]["distance"]
    distance = opponent_distance  # Use provided distance

    if distance < 4.0:
        # Opponent close: CHARGE!
        if (my_position < arena_width * 0.4):
            acceleration = 5.0  # Charge right
        else:
            acceleration = -5.0  # Charge left
        stance = "extended"
    else:
        # Opponent far: Wait patiently
        acceleration = 0.0
        stance = "neutral"

    return {"acceleration": acceleration, "stance": stance}