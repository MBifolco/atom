"""
Zoner - Range Control and Poking Specialist

Strategy:
- Maintains maximum distance and pokes opponent
- Uses light attacks to damage from range
- Teaches learning fighter range management
- Forces learner to close distance effectively
"""


def decide(snapshot):
    """
    Range-control specialist that teaches distance management.
    
    Keeps maximum distance while dealing constant light damage through
    jabs and movement. Retreats further when opponent closes. Forces
    learner to manage approach distance and break through ranged pressure.
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

    # Critical HP - aggressive retreat with defense
    if my_hp_pct < 0.3:
        if near_left_wall:
            return {"acceleration": 2.5, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.5, "stance": "defending"}
        else:
            return {"acceleration": -3.0, "stance": "defending"}

    # Low stamina - recover while maintaining distance
    if my_stamina_pct < 0.25:
        if distance < 3.0:
            return {"acceleration": -2.0, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Wall detected - move away from wall
    if near_left_wall:
        return {"acceleration": 2.0, "stance": "neutral"}
    elif near_right_wall:
        return {"acceleration": -2.0, "stance": "neutral"}

    # ZONING PATTERNS

    # Opponent charging in - create distance
    if distance < 1.5 and opp_velocity > 1.5:
        return {"acceleration": -3.5, "stance": "retracted"}

    # Too close for safety - back away quickly
    if distance < 2.0:
        if my_stamina_pct > 0.5:
            # Have stamina - retreat fast with light jab
            return {"acceleration": -2.5, "stance": "extended"}
        else:
            # Low stamina - just retreat
            return {"acceleration": -2.0, "stance": "retracted"}

    # Optimal zoning range (3-5m) - poke from range
    elif distance >= 2.0 and distance <= 5.0:
        if my_stamina_pct > 0.4:
            # Have stamina - move around while extended
            if distance > 4.0:
                # Too far, advance slightly
                return {"acceleration": 1.0, "stance": "extended"}
            elif distance < 3.0:
                # A bit close, back away while attacking
                return {"acceleration": -0.5, "stance": "extended"}
            else:
                # Perfect range - stay mobile
                return {"acceleration": 0.5, "stance": "extended"}
        else:
            # Low stamina - maintain distance without attacking
            if distance > 3.5:
                return {"acceleration": 0.5, "stance": "neutral"}
            else:
                return {"acceleration": -1.0, "stance": "neutral"}

    # Far away - slowly advance to optimal zoning range
    else:  # distance > 5.0
        if my_stamina_pct > 0.6:
            return {"acceleration": 2.0, "stance": "extended"}
        else:
            return {"acceleration": 1.0, "stance": "neutral"}
