"""
Hit-and-Run - Mobility and Tactics Specialist

Strategy:
- Hits opponent then immediately backs away
- Repeats hit-retreat cycle constantly
- Teaches learning fighter dealing with mobile opponents
- Forces learner to predict and catch opponents
"""


def decide(snapshot):
    """
    Mobile hit-and-run fighter that teaches pursuit skills.
    
    Constantly circles opponent, attacks briefly, then backs away. Never
    commits to sustained combat. Forces learner to predict movement
    patterns, manage pursuit, and set traps for mobile opponents.
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
    if my_hp_pct < 0.25:
        if near_left_wall:
            return {"acceleration": 3.0, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -3.0, "stance": "defending"}
        else:
            return {"acceleration": -4.0, "stance": "defending"}

    # Very low stamina - recover while keeping distance
    if my_stamina_pct < 0.2:
        if distance < 2.0:
            return {"acceleration": -2.5, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Wall avoidance - absolutely critical for hit-and-run
    if near_left_wall:
        return {"acceleration": 3.5, "stance": "neutral"}
    elif near_right_wall:
        return {"acceleration": -3.5, "stance": "neutral"}

    # HIT-AND-RUN CYCLE

    # Phase 1: Close to opponent - ATTACK
    if distance < 1.5:
        if my_stamina_pct > 0.5:
            # Have stamina - hit hard then run
            return {"acceleration": 1.5, "stance": "extended"}
        else:
            # Low stamina - just run away
            return {"acceleration": -3.0, "stance": "retracted"}

    # Phase 2: Medium distance - Either retreat or advance to attack
    elif distance < 3.0:
        if my_stamina_pct > 0.6:
            # High stamina - advance for next hit
            return {"acceleration": 2.5, "stance": "neutral"}
        elif my_stamina_pct > 0.4:
            # Moderate stamina - approach carefully
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            # Low stamina - recover distance
            return {"acceleration": -1.5, "stance": "neutral"}

    # Phase 3: Far distance - Recover stamina and circle
    else:  # distance >= 3.0
        if my_stamina_pct < 0.4:
            # Need to recover - maintain distance
            return {"acceleration": 0.0, "stance": "neutral"}
        else:
            # Have stamina - advance for next attack cycle
            if opp_hp_pct > 0.4:
                # Opponent healthy - be cautious with approach
                return {"acceleration": 1.0, "stance": "neutral"}
            else:
                # Opponent hurt - more aggressive approach
                return {"acceleration": 2.0, "stance": "neutral"}
