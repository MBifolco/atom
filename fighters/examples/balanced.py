"""
Balanced - Adaptive Fighter

Strategy:
- Aggressive when winning
- Defensive when losing
- Adapts based on HP advantage
- Smart stamina management
"""


def decide(snapshot):
    """
    Balanced tactical fighter that adapts strategy based on situation.
    """
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check walls
    near_left_wall = my_position < 1.0
    near_right_wall = my_position > arena_width - 1.0

    # Critical HP - defensive retreat
    if my_hp_pct < 0.3:
        if near_left_wall:
            return {"acceleration": 2.5, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.5, "stance": "defending"}
        else:
            return {"acceleration": -3.5, "stance": "defending"}

    # Exhausted - recover
    if my_stamina_pct < 0.2:
        return {"acceleration": 0.0, "stance": "neutral"}

    # Avoid wall grinding
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "neutral"}
        else:
            return {"acceleration": -2.0, "stance": "neutral"}

    # Calculate HP advantage
    hp_advantage = my_hp_pct - opp_hp_pct

    # Winning significantly - press advantage
    if hp_advantage > 0.25 and my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 3.5, "stance": "neutral"}

    # Losing significantly - play defensive
    elif hp_advantage < -0.25:
        if distance < 2.0 and opp_velocity > 1.0:
            return {"acceleration": -2.0, "stance": "defending"}
        elif distance > 3.5:
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Even match - measured aggression based on stamina
    else:
        if my_stamina_pct > 0.6:
            # Have stamina - be aggressive
            if distance < 1.5:
                return {"acceleration": 0.0, "stance": "extended"}
            else:
                return {"acceleration": 3.0, "stance": "neutral"}
        elif my_stamina_pct > 0.3:
            # Moderate stamina - be cautious
            if distance < 1.0:
                return {"acceleration": 0.0, "stance": "extended"}
            elif distance > 3.0:
                return {"acceleration": 2.0, "stance": "neutral"}
            else:
                return {"acceleration": 0.0, "stance": "neutral"}
        else:
            # Low stamina - recover
            return {"acceleration": 0.0, "stance": "neutral"}
