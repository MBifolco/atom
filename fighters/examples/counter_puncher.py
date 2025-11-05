"""
Counter Puncher - Timing and Patience Specialist

Strategy:
- Waits for opponent to overextend
- Punishes aggressive mistakes
- Teaches learning fighter about timing, patience, and risk/reward
- Uses defensive stance effectively
"""


def decide(snapshot):
    """
    Patient counter-attacking fighter that teaches timing and restraint.
    
    Deliberately holds back and waits for opponent mistakes. Only attacks
    when opponent is overextended or low on stamina. Punishes reckless
    aggression with well-timed counter-attacks.
    """
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall
    near_left_wall = my_position < 1.0
    near_right_wall = my_position > arena_width - 1.0

    # Critical HP - escape
    if my_hp_pct < 0.3:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.0, "stance": "defending"}
        else:
            return {"acceleration": -3.5, "stance": "defending"}

    # Low stamina - recover
    if my_stamina_pct < 0.2:
        return {"acceleration": 0.0, "stance": "neutral"}

    # Wall avoidance
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            return {"acceleration": -1.5, "stance": "neutral"}

    # COUNTER-PUNCHING LOGIC
    
    # Opponent just charged in and is now within striking distance
    # This is the KEY vulnerability window - punish it
    if distance < 1.5 and opp_velocity > 1.0:
        # Opponent charging in = overextension opportunity
        if my_stamina_pct > 0.4:
            return {"acceleration": 0.5, "stance": "extended"}
        else:
            return {"acceleration": 0.0, "stance": "extended"}
    
    # Opponent is exhausted and close - capitalize
    if distance < 2.0 and opp_hp_pct < 0.4:
        if my_stamina_pct > 0.5:
            return {"acceleration": 1.0, "stance": "extended"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}
    
    # Opponent recovering (low velocity after charge) - still vulnerable
    if distance < 2.5 and opp_velocity < -0.5 and my_stamina_pct > 0.6:
        return {"acceleration": 0.5, "stance": "extended"}
    
    # Opponent at medium range - play defensive and wait
    if distance >= 2.0 and distance <= 4.0:
        if opp_velocity > 1.0:
            # Opponent approaching - prepare defense
            return {"acceleration": 0.0, "stance": "defending"}
        else:
            # Opponent not rushing - maintain position
            return {"acceleration": 0.0, "stance": "neutral"}
    
    # Opponent far away - either advance slowly or maintain position
    if distance > 4.0:
        if my_stamina_pct > 0.6 and opp_hp_pct > 0.5:
            # Have stamina and opponent healthy - slow advance
            return {"acceleration": 1.0, "stance": "neutral"}
        else:
            # Either low stamina or opponent hurt - don't commit
            return {"acceleration": 0.0, "stance": "neutral"}
    
    # Default: Wait and maintain distance
    return {"acceleration": 0.0, "stance": "defending"}
