"""
Rusher - Aggressive Fighter

Strategy:
- Rushes forward constantly
- Extends when close to opponent
- Backs away from walls
- Retreats when HP is critical
"""


def decide(snapshot):
    """
    Aggressive rushing fighter.

    Always pressures opponent but manages stamina and avoids walls.
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
