"""
Grappler - Close Combat Specialist

Strategy:
- Forces close-range combat
- Extended stance at close range constantly
- Teaches learning fighter close-quarters fighting
- Teaches defensive stance usage
"""


def decide(snapshot):
    """
    Close-combat specialist that teaches in-fighting skills.
    
    Always tries to get close and forces fighting at melee range. Uses
    extended stance constantly when in range. Forces learner to either
    match close-range intensity or develop strong distancing/defense.
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

    # Critical HP - still get close but be slightly defensive
    if my_hp_pct < 0.3:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.0, "stance": "defending"}
        else:
            return {"acceleration": -2.5, "stance": "defending"}

    # Low stamina - still close but don't attack
    if my_stamina_pct < 0.2:
        if distance > 1.5:
            # Far from opponent - recover while closing
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            # Already in close range - just neutral
            return {"acceleration": 0.0, "stance": "neutral"}

    # Wall avoidance
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "neutral"}
        else:
            return {"acceleration": -2.0, "stance": "neutral"}

    # GRAPPLING PATTERNS

    # Already in close range - FIGHT
    if distance < 1.5:
        if my_stamina_pct > 0.4:
            # Have stamina - attack hard
            return {"acceleration": 0.5, "stance": "extended"}
        else:
            # Low stamina - light attacks
            return {"acceleration": 0.0, "stance": "extended"}

    # Medium range - close in
    elif distance < 3.0:
        if my_stamina_pct > 0.5:
            # Have stamina - aggressive close
            return {"acceleration": 3.0, "stance": "extended"}
        elif my_stamina_pct > 0.3:
            # Moderate stamina - steady advance
            return {"acceleration": 2.0, "stance": "neutral"}
        else:
            # Low stamina - recover while advancing
            return {"acceleration": 1.0, "stance": "neutral"}

    # Far range - relentless pursuit
    else:  # distance >= 3.0
        if my_stamina_pct > 0.6:
            return {"acceleration": 4.0, "stance": "neutral"}
        elif my_stamina_pct > 0.4:
            return {"acceleration": 3.0, "stance": "neutral"}
        else:
            return {"acceleration": 2.0, "stance": "neutral"}
