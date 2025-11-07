"""
Wall Fighter

Behavioral fighter that intentionally uses walls as part of strategy.
Backs opponents into walls, then capitalizes on their reduced mobility.

Purpose: Test wall-trapping strategies, corner pressure,
and escape from disadvantageous positions.
"""


def decide(snapshot):
    """
    Wall fighter behavioral fighter.

    Forces opponents to walls and exploits their limited movement.
    """
    my_position = snapshot["you"]["position"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]

    opponent_distance = snapshot["opponent"]["distance"]
    distance = opponent_distance  # Use provided distance
    arena_width = snapshot["arena"]["width"]

    # Check opponent's wall proximity
    opponent_near_left_wall = opponent_position < 1.5
    opponent_near_right_wall = opponent_position > arena_width - 1.5

    # Check our wall proximity
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # Don't trap ourselves
    if near_left_wall and not opponent_near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}
    if near_right_wall and not opponent_near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Exploit opponent at wall
    if opponent_near_left_wall or opponent_near_right_wall:
        # Opponent trapped! Press advantage
        if distance > 2.0:
            # Close in for the kill
            if (my_position < arena_width * 0.4):
                acceleration = 4.0
            else:
                acceleration = -4.0
            stance = "neutral"  # Speed to close
        else:
            # In range: Punish
            if my_stamina_pct > 0.3:
                stance = "extended"
                # Small movements to maintain pressure
                if (my_position < arena_width * 0.4):
                    acceleration = 1.0
                else:
                    acceleration = -1.0
            else:
                stance = "defending"
                acceleration = 0.0

    else:
        # Push opponent toward nearest wall
        center = arena_width / 2
        if opponent_position < center:
            # Push toward left wall
            if distance > 2.5:
                # Approach from right
                if my_position < opponent_position:
                    acceleration = 3.0  # Get to their right
                else:
                    acceleration = -0.5  # Push left slowly
            else:
                # In position: Push them left
                acceleration = -2.0
                stance = "extended" if my_stamina_pct > 0.4 else "neutral"

        else:
            # Push toward right wall
            if distance > 2.5:
                # Approach from left
                if my_position > opponent_position:
                    acceleration = -3.0  # Get to their left
                else:
                    acceleration = 0.5  # Push right slowly
            else:
                # In position: Push them right
                acceleration = 2.0
                stance = "extended" if my_stamina_pct > 0.4 else "neutral"

    return {"acceleration": acceleration, "stance": stance}