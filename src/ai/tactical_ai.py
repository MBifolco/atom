"""
Atom Combat - Tactical AI

Improved fighter AI with better retreat logic and stamina management.
"""


def tactical_aggressive(snapshot):
    """
    Tactical aggressive fighter.

    - Pressures opponent but manages stamina
    - Retreats when low HP or exhausted
    - Avoids wall grinding
    """
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall (within 1.0m)
    near_left_wall = my_position < 1.0
    near_right_wall = my_position > arena_width - 1.0

    # Emergency retreat - low HP
    if my_hp_pct < 0.4:
        return {"acceleration": -4.0, "stance": "defending"}

    # Exhausted - fall back to neutral and recover
    if my_stamina_pct < 0.2:
        return {"acceleration": 0.0, "stance": "neutral"}

    # Avoid wall grinding - back away if stuck on wall
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "neutral"}  # Move right
        else:
            return {"acceleration": -2.0, "stance": "neutral"}  # Move left

    # Close range - strike if have stamina
    if distance < 1.0:
        if my_stamina_pct > 0.4:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Mid range - manage stamina while advancing
    elif distance < 3.0:
        if my_stamina_pct > 0.6:
            return {"acceleration": 3.0, "stance": "neutral"}
        elif my_stamina_pct > 0.3:
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Long range - advance steadily
    else:
        if my_stamina_pct > 0.5:
            return {"acceleration": 3.5, "stance": "neutral"}
        else:
            return {"acceleration": 1.0, "stance": "neutral"}


def tactical_defensive(snapshot):
    """
    Tactical defensive fighter.

    - Counter-punches when opponent overextends
    - Maintains optimal distance
    - Retreats strategically
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


def tactical_balanced(snapshot):
    """
    Tactical balanced fighter.

    - Adapts strategy based on situation
    - Aggressive when winning, defensive when losing
    - Smart stamina management
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
