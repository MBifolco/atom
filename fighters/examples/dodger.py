"""
Dodger - Evasion and Counter-Attack Specialist

Master of dodge and counter who punishes aggressive opponents.

Strategy:
- Dodges incoming attacks then immediately counters
- Creates openings through superior movement
- Punishes overextension and exhaustion
- Not just evasion - MUST counter-attack to win
- Uses all stances tactically during counters
- Forces contact through calculated strikes
"""


def decide(snapshot):
    """
    True dodge-and-counter fighter that evades then strikes.

    Avoids attacks through movement, then immediately counters when
    opponent is vulnerable. Not just running away - actively creates
    and exploits openings. Must balance evasion with aggression.
    """
    # Extract key information
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_velocity = snapshot["you"]["velocity"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if near wall - critical to avoid
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # CRITICAL: Avoid wall damage but counter if possible
    if near_left_wall:
        if distance < 2.0 and my_stamina_pct > 0.3:
            return {"acceleration": 4.0, "stance": "extended"}  # Counter while escaping
        return {"acceleration": 4.0, "stance": "neutral"}
    if near_right_wall:
        if distance < 2.0 and my_stamina_pct > 0.3:
            return {"acceleration": -4.0, "stance": "extended"}  # Counter while escaping
        return {"acceleration": -4.0, "stance": "neutral"}

    # DODGE AND COUNTER PATTERNS

    # COUNTER OPPORTUNITY 1: Opponent charging
    if distance < 2.5 and opp_velocity > 2.0:
        if my_stamina_pct > 0.3:
            # Sidestep and counter-strike
            return {"acceleration": 4.0, "stance": "extended"}
        else:
            # Low stamina - just dodge
            return {"acceleration": -3.0, "stance": "defending"}

    # COUNTER OPPORTUNITY 2: Opponent exhausted
    if distance < 3.0 and opp_stamina_pct < 0.25:
        # They're tired - PUNISH THEM
        return {"acceleration": 4.0, "stance": "extended"}

    # COUNTER OPPORTUNITY 3: Opponent retreating/recovering
    if distance < 3.5 and opp_velocity < -1.0:
        # They're backing away - chase and strike
        return {"acceleration": 4.5, "stance": "extended"}

    # COUNTER OPPORTUNITY 4: After successful dodge
    if distance > 2.0 and distance < 3.5 and abs(my_velocity) > 2.0:
        # We just dodged - now counter
        if my_stamina_pct > 0.2:
            return {"acceleration": 3.5, "stance": "extended"}

    # CLOSE RANGE (<1.5m) - Emergency dodge or counter
    if distance < 1.5:
        # Too close - decide quickly
        if opp_stamina_pct < 0.3 and my_stamina_pct > 0.3:
            # They're tired, we're not - ATTACK
            return {"acceleration": 2.0, "stance": "extended"}
        elif my_stamina_pct > 0.4:
            # Quick strike then escape
            return {"acceleration": -2.5, "stance": "extended"}
        else:
            # Must escape
            return {"acceleration": -3.5, "stance": "defending"}

    # DODGE RANGE (1.5-2.5m) - Active dodging with counter setup
    elif distance < 2.5:
        # Check if we should counter or dodge
        if my_stamina_pct > opp_stamina_pct and my_stamina_pct > 0.35:
            # We have advantage - aggressive counter
            return {"acceleration": 2.5, "stance": "extended"}
        elif my_stamina_pct > 0.25:
            # Normal dodge with counter prep
            return {"acceleration": -2.0, "stance": "neutral"}
        else:
            # Low stamina - pure dodge
            return {"acceleration": -2.5, "stance": "defending"}

    # OPTIMAL RANGE (2.5-3.5m) - Set up counters
    elif distance < 3.5:
        # Perfect range for dodge-counter tactics

        # Check for counter opportunity
        if opp_velocity > 1.5:
            # They're approaching - prepare counter
            if my_stamina_pct > 0.4:
                return {"acceleration": 0.5, "stance": "extended"}
            else:
                return {"acceleration": -1.0, "stance": "neutral"}
        else:
            # Maintain optimal range
            if my_stamina_pct > 0.5:
                # Good stamina - threaten with extended
                return {"acceleration": 0.0, "stance": "extended"}
            else:
                # Recover while maintaining distance
                return {"acceleration": -0.5, "stance": "neutral"}

    # FAR RANGE (3.5m+) - Reset and recover
    else:
        # Too far - need to close for counter opportunities
        if my_hp_pct < opp_hp_pct and my_stamina_pct > 0.4:
            # We're losing - must engage
            return {"acceleration": 3.0, "stance": "neutral"}
        elif my_stamina_pct > 0.6:
            # High stamina - approach for setup
            return {"acceleration": 2.0, "stance": "neutral"}
        else:
            # Recover stamina
            return {"acceleration": 0.0, "stance": "neutral"}
