"""
Tank - Defensive Counter-Puncher

Strategy:
- Maintains optimal distance (2-4m)
- Defends when opponent charges
- Counter-attacks when close
- Avoids walls strategically
"""


def decide(snapshot):
    """
    Defensive counter-punching fighter.

    Maintains distance, defends against charges, counter-attacks when opponent overextends.
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

    # Critical HP - retreat with defense
    if my_hp_pct < 0.35:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.0, "stance": "defending"}
        else:
            return {"acceleration": -3.0, "stance": "defending"}

    # Low stamina - recover
    if my_stamina_pct < 0.25:
        return {"acceleration": 0.0, "stance": "neutral"}

    # Avoid wall
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            return {"acceleration": -1.5, "stance": "neutral"}

    # Opponent charging in - brace for impact
    if distance < 2.0 and opp_velocity > 1.5:
        return {"acceleration": 0.0, "stance": "defending"}

    # Counter-attack opportunity - close range
    if distance < 1.2 and my_stamina_pct > 0.4:
        return {"acceleration": 1.0, "stance": "extended"}

    # Maintain optimal distance (2-4m)
    if distance < 2.0:
        # Too close - back away
        return {"acceleration": -2.0, "stance": "retracted"}
    elif distance > 4.0:
        # Too far - close in
        if my_stamina_pct > 0.5:
            return {"acceleration": 2.5, "stance": "neutral"}
        else:
            return {"acceleration": 1.0, "stance": "neutral"}
    else:
        # Good distance - hold position
        return {"acceleration": 0.0, "stance": "neutral"}
