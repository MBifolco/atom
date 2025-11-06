"""
Zoner - Range Control and Poking Specialist

Master of distance control who wins through calculated pokes.

Strategy:
- Controls optimal range (3-4m) and pokes consistently
- Uses extended stance to maximize reach
- Retreats only when opponent is too close (<2m)
- Advances when opponent is too far (>4m)
- Creates contact through positioning, not just running away
- Punishes aggressive approaches
"""


def decide(snapshot):
    """
    True zoner - controls space and pokes from optimal range.

    Maintains 3-4m distance for maximum effectiveness. Uses extended
    stance to poke when positioned correctly. Only retreats when truly
    necessary. Creates consistent contact through smart positioning.
    """
    # Extract key information
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_velocity = snapshot["you"]["velocity"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall - detect early for better positioning
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # CRITICAL: Avoid wall damage
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}  # Strong escape from wall
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}  # Strong escape from wall

    # ZONING PATTERNS - Control the distance

    # TOO CLOSE (<2m) - Create space
    if distance < 2.0:
        # Opponent is rushing - punish with poke while backing
        if opp_velocity > 2.0:
            # Quick poke and retreat
            return {"acceleration": -2.0, "stance": "extended"}

        # Normal close range - back away
        elif my_stamina_pct > 0.2:
            # Have stamina - fighting retreat
            return {"acceleration": -2.5, "stance": "extended"}
        else:
            # Low stamina - just create space
            return {"acceleration": -3.0, "stance": "neutral"}

    # PERFECT ZONING RANGE (2-3.5m) - POKE ZONE
    elif distance >= 2.0 and distance < 3.5:
        # This is our domain - control it!

        # Check opponent movement
        if opp_velocity > 1.5:
            # They're approaching - hold ground and poke
            return {"acceleration": 0.0, "stance": "extended"}

        elif opp_velocity < -1.5:
            # They're retreating - pursue with pokes
            return {"acceleration": 2.0, "stance": "extended"}

        else:
            # They're stationary/slow - optimal poking
            if my_stamina_pct > 0.15:
                # Perfect range - more aggressive poking movements
                if distance > 2.8:
                    # Slightly far - move forward aggressively while poking
                    return {"acceleration": 1.5, "stance": "extended"}
                elif distance < 2.3:
                    # Slightly close - small retreat while poking
                    return {"acceleration": -1.0, "stance": "extended"}
                else:
                    # PERFECT 2.3-2.8m - small forward pressure for contact
                    return {"acceleration": 0.3, "stance": "extended"}
            else:
                # Low stamina - maintain position
                return {"acceleration": 0.0, "stance": "neutral"}

    # OUTER RANGE (3.5-5m) - Close to optimal
    elif distance >= 3.5 and distance < 5.0:
        # Still in extended reach range - advance with pokes

        if my_stamina_pct > 0.3:
            # Have stamina - aggressive advance with pokes
            return {"acceleration": 2.5, "stance": "extended"}
        elif my_stamina_pct > 0.15:
            # Moderate stamina - steady advance
            return {"acceleration": 1.5, "stance": "extended"}
        else:
            # Low stamina - slow advance
            return {"acceleration": 1.0, "stance": "neutral"}

    # TOO FAR (5m+) - Close distance quickly
    else:
        # We need to get to zoning range
        if my_stamina_pct > 0.4:
            # Sprint to optimal range
            return {"acceleration": 4.0, "stance": "neutral"}
        elif my_stamina_pct > 0.2:
            # Quick advance
            return {"acceleration": 3.0, "stance": "neutral"}
        else:
            # Steady advance
            return {"acceleration": 2.0, "stance": "neutral"}
