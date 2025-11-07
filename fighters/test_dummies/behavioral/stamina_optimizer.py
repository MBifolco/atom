"""
Stamina Optimizer

Behavioral fighter that demonstrates perfect stamina management.
Never wastes stamina, times attacks for maximum efficiency,
and maintains sustained pressure through resource control.

Purpose: Test against efficient resource usage, long-game
strategies, and sustained combat effectiveness.
"""


def decide(snapshot):
    """
    Stamina optimizer behavioral fighter.

    Perfect stamina management with calculated aggression windows.
    """
    my_position = snapshot["you"]["position"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]

    opponent_distance = snapshot["opponent"]["distance"]
    opponent_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]
    opponent_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]

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

    # Stamina-efficient combat decisions
    if my_stamina_pct > 0.7:
        # High stamina: Can afford aggression
        if distance < 2.0:
            # Close range: Efficient attacks
            if opponent_stamina_pct < 0.3:
                # Opponent exhausted: Capitalize
                stance = "extended"
                acceleration = 0.5 if (my_position < arena_width * 0.4) else -0.5
            else:
                # Both have stamina: Calculated trades
                stance = "neutral"  # Balanced approach
                acceleration = 0.0
        else:
            # Mid-long range: Approach efficiently
            if (my_position < arena_width * 0.4):
                acceleration = 2.0
            else:
                acceleration = -2.0
            stance = "neutral"

    elif my_stamina_pct > 0.4:
        # Medium stamina: Selective engagement
        if distance < 1.5 and opponent_stamina_pct < 0.2:
            # Exploit exhausted opponent
            stance = "extended"
            acceleration = 0.3 if (my_position < arena_width * 0.4) else -0.3
        else:
            # Maintain neutral, recover slowly
            stance = "neutral"
            acceleration = 0.0

    elif my_stamina_pct > 0.2:
        # Low stamina: Defensive preservation
        if distance < 2.0:
            # Too close: Create space
            if (my_position < arena_width * 0.4):
                acceleration = -2.0
            else:
                acceleration = 2.0
            stance = "defending"
        else:
            # Safe distance: Recover
            acceleration = 0.0
            stance = "neutral"

    else:
        # Critical stamina: Full recovery mode
        if distance < 3.0:
            # Retreat
            if (my_position < arena_width * 0.4):
                acceleration = -3.0
            else:
                acceleration = 3.0
        else:
            acceleration = 0.0

        stance = "retracted"  # Maximum recovery

    return {"acceleration": acceleration, "stance": stance}