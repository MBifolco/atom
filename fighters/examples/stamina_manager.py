"""
Stamina Manager - Resource Efficiency Specialist

Strategy:
- Hyper-aware of stamina levels
- Alternates aggressive and recovery phases
- Teaches learning fighter to manage stamina as a resource
- Uses recovery windows to advantage
"""


def decide(snapshot):
    """
    Stamina-aware fighter that teaches resource management.
    
    Demonstrates clear stamina management patterns: aggressive when
    recovering, conservative when depleted. Punishes wasteful stamina use.
    Forces learner to understand stamina economy and pacing.
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

    # Critical HP - escape at any cost
    if my_hp_pct < 0.25:
        if near_left_wall:
            return {"acceleration": 2.0, "stance": "defending"}
        elif near_right_wall:
            return {"acceleration": -2.0, "stance": "defending"}
        else:
            return {"acceleration": -3.0, "stance": "defending"}

    # Wall avoidance
    if near_left_wall or near_right_wall:
        if near_left_wall:
            return {"acceleration": 1.5, "stance": "neutral"}
        else:
            return {"acceleration": -1.5, "stance": "neutral"}

    # STAMINA MANAGEMENT PHASES
    
    # Phase 1: Extreme exhaustion - MUST recover
    if my_stamina_pct < 0.15:
        return {"acceleration": 0.0, "stance": "neutral"}
    
    # Phase 2: Low stamina - recovery phase
    elif my_stamina_pct < 0.35:
        # Move away and recover
        if distance < 2.0:
            return {"acceleration": -1.5, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}
    
    # Phase 3: Moderate stamina - careful combat
    elif my_stamina_pct < 0.55:
        # Can attack but must be strategic
        if distance < 1.2 and opp_velocity < 1.0:
            return {"acceleration": 0.0, "stance": "extended"}
        elif distance < 2.0:
            # Opponent close but rushing - back away
            return {"acceleration": -1.0, "stance": "neutral"}
        elif distance > 3.5:
            # Slowly advance but conserve stamina
            return {"acceleration": 1.0, "stance": "neutral"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}
    
    # Phase 4: Good stamina - aggressive attack window
    elif my_stamina_pct < 0.75:
        if distance < 1.2:
            # In range - extended attack
            if my_hp_pct > opp_hp_pct:
                return {"acceleration": 0.5, "stance": "extended"}
            else:
                return {"acceleration": 0.0, "stance": "extended"}
        elif distance > 3.0:
            # Far away - advance with moderate speed
            return {"acceleration": 2.0, "stance": "neutral"}
        else:
            # Mid-range - close in
            return {"acceleration": 1.5, "stance": "neutral"}
    
    # Phase 5: High stamina - maximum offense
    else:  # > 0.75
        if distance < 1.0:
            # Very close - aggressive extended
            return {"acceleration": 1.0, "stance": "extended"}
        elif distance > 3.5:
            # Far away - hard charge
            return {"acceleration": 3.5, "stance": "neutral"}
        else:
            # Mid-range - aggressive advance
            return {"acceleration": 2.5, "stance": "neutral"}
