"""
Balanced - Adaptive Fighter

True balanced fighter that adapts stance and strategy based on situation.

Strategy:
- Switches between all stances tactically
- Aggressive (extended) when winning
- Defensive (defending) when losing
- Neutral for repositioning and stamina recovery
- Adapts distance based on stamina and HP
- Never gives up - fights to the end
"""


def decide(snapshot):
    """
    Truly balanced fighter using all stances adaptively.

    Matches opponent's aggression level and counters appropriately.
    Uses full range of stances for tactical advantage.
    """
    # Extract key information
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_velocity = snapshot["you"]["velocity"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check walls - critical to avoid
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # CRITICAL: Avoid wall damage
    if near_left_wall:
        return {"acceleration": 3.5, "stance": "defending"}  # Push away safely
    if near_right_wall:
        return {"acceleration": -3.5, "stance": "defending"}  # Push away safely

    # Calculate tactical advantage
    hp_advantage = my_hp_pct - opp_hp_pct
    stamina_advantage = my_stamina_pct - opp_stamina_pct

    # CLOSE COMBAT (< 1.5m) - Choose stance based on situation
    if distance < 1.5:
        # We're in fighting range - adapt stance to situation

        # Critical HP - maximum defense
        if my_hp_pct < 0.15:
            return {"acceleration": -1.0, "stance": "defending"}

        # Winning and have stamina - press advantage
        elif hp_advantage > 0.2 and my_stamina_pct > 0.3:
            return {"acceleration": 0.5, "stance": "extended"}

        # Losing but opponent is tired - exploit weakness
        elif hp_advantage < -0.2 and opp_stamina_pct < 0.3:
            return {"acceleration": 1.0, "stance": "extended"}

        # Both low stamina - defend and recover
        elif my_stamina_pct < 0.2 and opp_stamina_pct < 0.2:
            return {"acceleration": 0.0, "stance": "defending"}

        # Good stamina, even match - calculated offense
        elif my_stamina_pct > 0.5 and abs(hp_advantage) < 0.2:
            return {"acceleration": 0.3, "stance": "extended"}

        # Moderate stamina - balanced approach
        elif my_stamina_pct > 0.25:
            # Alternate between offense and defense
            if stamina_advantage > 0:
                return {"acceleration": 0.2, "stance": "extended"}
            else:
                return {"acceleration": 0.0, "stance": "defending"}

        # Low stamina - defend
        else:
            return {"acceleration": -0.5, "stance": "defending"}

    # MEDIUM RANGE (1.5-3m) - Positioning battle
    elif distance < 3.0:
        # Check if opponent is charging
        opponent_charging = opp_velocity > 2.0 and distance < 2.5

        if opponent_charging:
            # Counter-charge or defend based on stamina
            if my_stamina_pct > 0.4:
                # Meet them head-on
                return {"acceleration": 3.0, "stance": "extended"}
            else:
                # Brace for impact
                return {"acceleration": -0.5, "stance": "defending"}

        # We're winning - control distance
        elif hp_advantage > 0.3:
            if my_stamina_pct > 0.4:
                # Close in for the kill
                return {"acceleration": 3.5, "stance": "extended"}
            else:
                # Maintain pressure while recovering
                return {"acceleration": 1.5, "stance": "neutral"}

        # We're losing - tactical repositioning
        elif hp_advantage < -0.3:
            if my_stamina_pct > opp_stamina_pct:
                # We have stamina advantage - attack
                return {"acceleration": 3.0, "stance": "neutral"}
            else:
                # Create distance to recover
                return {"acceleration": -2.0, "stance": "defending"}

        # Even match - stamina-based approach
        else:
            if my_stamina_pct > 0.6:
                # High stamina - aggressive approach
                return {"acceleration": 3.0, "stance": "extended"}
            elif my_stamina_pct > 0.3:
                # Moderate stamina - measured approach
                return {"acceleration": 2.0, "stance": "neutral"}
            else:
                # Low stamina - defensive positioning
                return {"acceleration": 0.5, "stance": "defending"}

    # LONG RANGE (3m+) - Strategic approach
    else:
        # Emergency situation - we're badly hurt
        if my_hp_pct < 0.2 and hp_advantage < 0:
            # Try to survive and find openings
            if my_stamina_pct > 0.5:
                # Have stamina - aggressive comeback attempt
                return {"acceleration": 4.0, "stance": "neutral"}
            else:
                # Conserve and look for opportunity
                return {"acceleration": 1.0, "stance": "defending"}

        # We're dominating - finish them
        elif hp_advantage > 0.4:
            return {"acceleration": 4.0, "stance": "extended"}

        # Close fight - smart engagement
        else:
            # High stamina - close distance aggressively
            if my_stamina_pct > 0.7:
                return {"acceleration": 4.5, "stance": "neutral"}

            # Moderate stamina - steady approach
            elif my_stamina_pct > 0.4:
                return {"acceleration": 3.0, "stance": "neutral"}

            # Low stamina - cautious advance
            elif my_stamina_pct > 0.2:
                return {"acceleration": 2.0, "stance": "defending"}

            # Very low stamina - recover while closing
            else:
                return {"acceleration": 1.0, "stance": "neutral"}
