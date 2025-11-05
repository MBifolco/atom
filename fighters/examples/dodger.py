"""
Dodger - Evasion and Movement Specialist

Strategy:
- Constantly moves to avoid opponent
- Kites away and counter-attacks
- Teaches learning fighter how to pursue and predict movement
- Uses stamina efficiently for mobility
"""


def decide(snapshot):
    """
    Evasive fighter that teaches pursuit and prediction.
    
    Avoids direct combat through superior movement, counter-attacks when
    opponent overextends trying to catch it. Forces learner to think about
    prediction and positioning rather than just forward aggression.
    """
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall
    near_left_wall = my_position < 1.5
    near_right_wall = my_position > arena_width - 1.5

    # Critical HP - aggressive escape
    if my_hp_pct < 0.3:
        if near_left_wall or near_right_wall:
            return {"acceleration": -4.0, "stance": "retracted"}
        else:
            return {"acceleration": -4.0, "stance": "defending"}

    # Low stamina - recover while moving away
    if my_stamina_pct < 0.25:
        if distance < 2.0:
            return {"acceleration": -2.0, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Wall detected - move away from wall laterally
    if near_left_wall:
        return {"acceleration": 2.5, "stance": "neutral"}
    elif near_right_wall:
        return {"acceleration": -2.5, "stance": "neutral"}

    # Opponent charging in - back away
    if distance < 1.5 and opp_velocity > 1.5:
        return {"acceleration": -3.5, "stance": "retracted"}

    # Counter-attack opportunity - opponent overextended and low stamina
    if distance < 2.0 and opp_hp_pct < 0.5 and my_stamina_pct > 0.5:
        return {"acceleration": 1.5, "stance": "extended"}

    # Normal evasion pattern - maintain distance by moving away
    if distance < 2.5:
        # Opponent too close - create space
        if my_stamina_pct > 0.4:
            return {"acceleration": -2.5, "stance": "retracted"}
        else:
            return {"acceleration": -1.0, "stance": "retracted"}
    elif distance < 4.0:
        # Good distance - stay mobile
        return {"acceleration": -0.5, "stance": "neutral"}
    else:
        # Far away - let opponent chase, recover stamina
        if my_stamina_pct < 0.6:
            return {"acceleration": 0.0, "stance": "neutral"}
        else:
            # Have stamina - continue moving away to maintain distance
            return {"acceleration": -1.5, "stance": "neutral"}
