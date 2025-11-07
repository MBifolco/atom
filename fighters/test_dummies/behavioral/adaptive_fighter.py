"""
Adaptive Fighter

Behavioral fighter that changes strategy based on opponent behavior.
Detects patterns like aggression, passivity, or kiting and adjusts
tactics accordingly.

Purpose: Test against adaptive strategies, pattern recognition,
and dynamic tactical adjustments.
"""


def decide(snapshot):
    """
    Adaptive fighter behavioral fighter.

    Analyzes opponent patterns and adjusts strategy dynamically.
    """
    my_position = snapshot["you"]["position"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["stamina_max"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["hp_max"]

    opponent_distance = snapshot["opponent"]["distance"]
    opponent_velocity = snapshot["opponent"]["velocity"]
    opponent_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["stamina_max"]
    opponent_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["hp_max"]
    opponent_stance = snapshot["opponent"]["stance"]

    distance = opponent_distance  # Use provided distance
    arena_width = snapshot["arena"]["width"]
    tick = snapshot["tick"]

    # Wall detection
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # Escape from walls
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Detect opponent patterns
    is_aggressive = opponent_stance == "extended" and distance < 3.0
    is_defensive = opponent_stance == "defending" or opponent_stance == "retracted"
    is_fleeing = abs(opponent_velocity) > 2.0 and distance > 3.0
    is_low_stamina = opponent_stamina_pct < 0.3

    # Adapt strategy based on opponent
    if is_low_stamina:
        # Exploit exhausted opponent
        if distance > 2.0:
            # Rush them
            if (my_position < arena_width * 0.4):
                acceleration = 4.0
            else:
                acceleration = -4.0
        else:
            # Pressure them
            acceleration = 1.0 if (my_position < arena_width * 0.4) else -1.0

        stance = "extended" if my_stamina_pct > 0.2 else "neutral"

    elif is_aggressive:
        # Counter aggression with defense and spacing
        if distance < 2.0:
            # Back away
            if (my_position < arena_width * 0.4):
                acceleration = -3.0
            else:
                acceleration = 3.0
            stance = "defending"
        else:
            # Maintain distance
            acceleration = 0.0
            stance = "defending" if my_stamina_pct > 0.3 else "retracted"

    elif is_defensive:
        # Break through defense with calculated pressure
        if distance > 2.5:
            # Approach
            if (my_position < arena_width * 0.4):
                acceleration = 2.5
            else:
                acceleration = -2.5
            stance = "neutral"
        else:
            # Pressure carefully
            acceleration = 0.5 if (my_position < arena_width * 0.4) else -0.5
            stance = "extended" if my_stamina_pct > 0.5 else "neutral"

    elif is_fleeing:
        # Pursue fleeing opponent
        if (my_position < arena_width * 0.4):
            acceleration = 5.0
        else:
            acceleration = -5.0
        stance = "neutral"  # Speed over damage

    else:
        # Default balanced approach
        if distance < 2.0:
            # Close range tactics
            if my_stamina_pct > opponent_stamina_pct:
                stance = "extended"
                acceleration = 0.3 if (my_position < arena_width * 0.4) else -0.3
            else:
                stance = "defending"
                acceleration = -1.0 if (my_position < arena_width * 0.4) else 1.0
        else:
            # Approach cautiously
            if (my_position < arena_width * 0.4):
                acceleration = 1.5
            else:
                acceleration = -1.5
            stance = "neutral"

    return {"acceleration": acceleration, "stance": stance}