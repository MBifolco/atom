"""
Grappler - Close Combat Specialist

Master of in-fighting who dominates at point-blank range.

Strategy:
- Forces close-range combat relentlessly
- Extended stance constantly at close range for maximum damage
- "Sticks" to opponent like glue
- Chases down runners with determination
- Never gives opponent breathing room
- High collision count through constant contact
"""


def decide(snapshot):
    """
    Close-combat specialist that excels at in-fighting.

    Relentlessly closes distance and maintains contact. Uses extended
    stance aggressively in close range. Pursues retreating opponents
    and punishes them for running.
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

    # Check if near wall - detect early to avoid getting stuck
    near_left_wall = my_position < 2.0  # Early detection
    near_right_wall = my_position > arena_width - 2.0  # Early detection

    # CRITICAL: Avoid wall damage at all costs
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}  # Strong escape from wall
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}  # Strong escape from wall

    # CORE GRAPPLING BEHAVIOR: Stick to opponent like glue

    # CLOSE RANGE (<1.2m) - Maximum aggression
    if distance < 1.2:
        # Perfect grappling range - never let go!

        # Have any stamina at all - keep attacking
        if my_stamina_pct > 0.05:
            # Check if opponent is trying to escape
            if abs(opp_velocity) > 2.0:
                # They're running - chase them!
                if opp_velocity > 0:
                    return {"acceleration": 2.0, "stance": "extended"}
                else:
                    return {"acceleration": -2.0, "stance": "extended"}
            else:
                # They're fighting - maintain contact with slight pressure
                return {"acceleration": 0.8, "stance": "extended"}

        # Completely exhausted - still stay close
        else:
            # Maintain contact even while recovering
            return {"acceleration": 0.3, "stance": "neutral"}

    # GRAPPLING RANGE (1.2-2m) - Close aggressively
    elif distance < 2.0:
        # Still in fighting range - close the gap

        if my_stamina_pct > 0.15:
            # Rush in with extended for immediate damage
            return {"acceleration": 4.0, "stance": "extended"}

        # Low stamina but keep pressure
        elif my_stamina_pct > 0.05:
            # Close in defending to conserve stamina
            return {"acceleration": 3.0, "stance": "defending"}

        # Critical stamina - still advance
        else:
            return {"acceleration": 2.0, "stance": "neutral"}

    # MEDIUM RANGE (2-3.5m) - Pursuit mode
    elif distance < 3.5:
        # Check if opponent is retreating
        opponent_retreating = opp_velocity * (1 if my_position < arena_width/2 else -1) < -2.0

        if opponent_retreating:
            # They're running - PUNISH THEM
            return {"acceleration": 5.0, "stance": "extended"}

        # Normal approach based on stamina
        if my_stamina_pct > 0.3:
            # Have stamina - aggressive pursuit with extended
            return {"acceleration": 4.5, "stance": "extended"}

        elif my_stamina_pct > 0.15:
            # Moderate stamina - steady pursuit
            return {"acceleration": 3.5, "stance": "neutral"}

        else:
            # Low stamina - still pursue but carefully
            return {"acceleration": 2.5, "stance": "defending"}

    # FAR RANGE (3.5m+) - Maximum pursuit
    else:
        # Check game state
        hp_advantage = my_hp_pct - opp_hp_pct

        # Opponent is running and winning - must catch them
        if hp_advantage < -0.1:
            # Desperate chase - they can't escape
            return {"acceleration": 5.0, "stance": "extended"}

        # We're winning - controlled approach
        elif hp_advantage > 0.2:
            if my_stamina_pct > 0.4:
                # Steady pressure
                return {"acceleration": 3.5, "stance": "neutral"}
            else:
                # Low stamina - patient approach
                return {"acceleration": 2.0, "stance": "neutral"}

        # Even match - stamina-based pursuit
        else:
            if my_stamina_pct > 0.5:
                # High stamina - full sprint
                return {"acceleration": 5.0, "stance": "neutral"}

            elif my_stamina_pct > 0.25:
                # Moderate stamina - quick approach
                return {"acceleration": 4.0, "stance": "neutral"}

            else:
                # Low stamina - steady pursuit
                return {"acceleration": 3.0, "stance": "neutral"}
