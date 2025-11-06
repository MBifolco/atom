"""
Perfect Kiter

Behavioral fighter that demonstrates optimal kiting strategy.
Maintains 2-3m distance, attacks while retreating, and uses
superior positioning to chip away at opponents.

Purpose: Test against hit-and-run tactics, spacing control,
and sustained pressure from range.
"""


def decide(snapshot):
    """
    Perfect kiter behavioral fighter.

    Maintains optimal distance while dealing consistent damage.
    """
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["stamina_max"]

    opponent_position = snapshot["opponent"]["position"]
    opponent_velocity = snapshot["opponent"]["velocity"]
    distance = abs(opponent_position - my_position)
    arena_width = snapshot["arena"]["width"]

    # Wall detection - critical for kiting
    near_left_wall = my_position < 3.0
    near_right_wall = my_position > arena_width - 3.0

    # Wall escape takes priority
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Kiting distance management
    optimal_min = 2.0
    optimal_max = 3.0

    if distance < optimal_min:
        # Too close: Retreat quickly
        if opponent_position > my_position:
            acceleration = -4.0
        else:
            acceleration = 4.0

        # Attack while retreating if stamina permits
        if my_stamina_pct > 0.3:
            stance = "extended"
        else:
            stance = "neutral"

    elif distance > optimal_max:
        # Too far: Approach carefully
        if opponent_position > my_position:
            acceleration = 2.0
        else:
            acceleration = -2.0

        stance = "neutral"  # Conserve stamina while closing

    else:
        # Perfect kiting range
        # Match opponent's movement to maintain distance
        if opponent_velocity > 0.5:
            # Opponent moving right
            if opponent_position > my_position:
                acceleration = 1.0  # Move with them slowly
            else:
                acceleration = 3.0  # Move away faster
        elif opponent_velocity < -0.5:
            # Opponent moving left
            if opponent_position < my_position:
                acceleration = -1.0  # Move with them slowly
            else:
                acceleration = -3.0  # Move away faster
        else:
            # Opponent stationary: Small adjustments
            acceleration = 0.0

        # Attack at optimal range
        if my_stamina_pct > 0.25:
            stance = "extended"
        else:
            stance = "defending"  # Defensive when low stamina

    return {"acceleration": acceleration, "stance": stance}