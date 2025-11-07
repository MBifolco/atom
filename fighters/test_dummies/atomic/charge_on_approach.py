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
    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)

    if distance < 4.0:
        # Opponent close: CHARGE!
        if opponent_position > my_position:
            acceleration = 5.0  # Charge right
        else:
            acceleration = -5.0  # Charge left
        stance = "extended"
    else:
        # Opponent far: Wait patiently
        acceleration = 0.0
        stance = "neutral"

    return {"acceleration": acceleration, "stance": stance}