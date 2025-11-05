"""
Berserker - Relentless All-Out Attack Specialist

Strategy:
- Attacks aggressively at all times
- Never backs down or defends
- Teaches learning fighter to develop absolute defense
- Forces learner to survive onslaught and counter-attack
"""


def decide(snapshot):
    """
    Relentless berserker that teaches defense against sustained pressure.
    
    Attacks constantly with extended stance and aggressive acceleration.
    Has minimal HP management. Forces learner to develop blocking/defense
    skills and teaches how to survive and counter against relentless offense.
    """
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall
    near_left_wall = my_position < 1.0
    near_right_wall = my_position > arena_width - 1.0

    # Only retreat if CRITICAL HP (desperate)
    if my_hp_pct < 0.15:
        if near_left_wall:
            return {"acceleration": 2.5, "stance": "extended"}
        elif near_right_wall:
            return {"acceleration": -2.5, "stance": "extended"}
        else:
            # Even at critical HP, keep attacking but back away slightly
            return {"acceleration": -1.0, "stance": "extended"}

    # Even at low stamina, keep attacking but reduce power
    if my_stamina_pct < 0.2:
        if distance < 1.5:
            # Already in range - light attack
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            # Out of range - slow advance
            return {"acceleration": 1.0, "stance": "neutral"}

    # Avoid wall but keep attacking
    if near_left_wall:
        return {"acceleration": 1.5, "stance": "extended"}
    elif near_right_wall:
        return {"acceleration": -1.5, "stance": "extended"}

    # BERSERKER COMBAT PATTERNS

    # Close range - maximum aggression
    if distance < 1.0:
        if my_stamina_pct > 0.3:
            return {"acceleration": 0.5, "stance": "extended"}
        else:
            return {"acceleration": 0.0, "stance": "extended"}

    # Mid range - aggressive advance
    elif distance < 2.5:
        if my_stamina_pct > 0.4:
            return {"acceleration": 3.0, "stance": "extended"}
        elif my_stamina_pct > 0.2:
            return {"acceleration": 1.5, "stance": "extended"}
        else:
            return {"acceleration": 0.0, "stance": "extended"}

    # Long range - relentless pursuit
    else:
        if my_stamina_pct > 0.5:
            return {"acceleration": 4.0, "stance": "extended"}
        elif my_stamina_pct > 0.3:
            return {"acceleration": 3.0, "stance": "extended"}
        elif my_stamina_pct > 0.15:
            return {"acceleration": 1.5, "stance": "extended"}
        else:
            return {"acceleration": 0.5, "stance": "extended"}
