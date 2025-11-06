"""
Burst Attacker

Behavioral fighter that demonstrates burst damage patterns.
Conserves stamina until reaching 90%, then unleashes aggressive
burst attacks until stamina depletes to 20%.

Purpose: Test against burst damage windows, stamina-based
aggression cycles, and spike damage mitigation.
"""


def decide(snapshot):
    """
    Burst attacker behavioral fighter.

    Cycles between conservation and explosive attack phases.
    """
    my_position = snapshot["you"]["position"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["stamina_max"]

    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)
    arena_width = snapshot["arena"]["width"]

    # Wall detection
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # Escape from walls
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Burst phase management
    if my_stamina_pct > 0.9:
        # BURST MODE ACTIVATED!
        # Aggressive approach with extended stance
        if distance > 1.5:
            if opponent_position > my_position:
                acceleration = 5.0
            else:
                acceleration = -5.0
        else:
            # In range: Maximum aggression
            acceleration = 1.0 if opponent_position > my_position else -1.0

        stance = "extended"

    elif my_stamina_pct > 0.2:
        # Mid-burst: Continue if already close
        if distance < 2.0:
            # Continue attacking
            acceleration = 0.5 if opponent_position > my_position else -0.5
            stance = "extended"
        else:
            # Too far, conserve
            acceleration = 0.0
            stance = "neutral"

    else:
        # Recovery phase: Full defensive retreat
        if distance < 3.0:
            # Back away
            if opponent_position > my_position:
                acceleration = -2.0
            else:
                acceleration = 2.0
            stance = "retracted"
        else:
            # Safe distance: Recover
            acceleration = 0.0
            stance = "retracted"

    return {"acceleration": acceleration, "stance": stance}