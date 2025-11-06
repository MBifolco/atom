"""
Tank - Defensive Fighter

A true defensive tank that stands its ground and outlasts opponents.

Strategy:
- Primary stance is "defending" for maximum damage reduction
- Never retreats - stands ground and absorbs hits
- Counter-attacks when opponent is close and vulnerable
- Uses superior defense multiplier to win through attrition
- Controls the center of the arena
"""


def decide(snapshot):
    """
    Tank fighter - stands ground and outlasts opponents.

    Uses defending stance as primary stance, counter-attacks when close,
    and never retreats from combat.
    """
    # Extract key information
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Check if we're near walls (avoid getting trapped)
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # CRITICAL: Avoid wall damage at all costs
    if near_left_wall:
        return {"acceleration": 3.0, "stance": "defending"}  # Push away from left wall
    if near_right_wall:
        return {"acceleration": -3.0, "stance": "defending"}  # Push away from right wall

    # Emergency: Very low HP - maximum defense
    if my_hp_pct < 0.2:
        # Turtle up completely
        return {"acceleration": 0.0, "stance": "defending"}

    # CORE TANK BEHAVIOR: Stand ground when close
    if distance < 2.0:
        # We're in fighting range - this is where tank excels

        # Counter-attack opportunity: opponent is close and we have stamina
        if my_stamina_pct > 0.3:
            # PUNISH - use extended stance to counter-attack
            # Lean slightly forward to maintain contact
            return {"acceleration": 0.5, "stance": "extended"}

        # Low stamina but close - defend and recover
        elif my_stamina_pct > 0.1:
            # Stand ground with defending stance
            # Small forward acceleration to resist pushback
            return {"acceleration": 0.3, "stance": "defending"}

        # Critical stamina - brief recovery
        else:
            # Neutral to recover stamina faster, but stay close
            return {"acceleration": 0.2, "stance": "neutral"}

    # Medium range (2-4m) - control space
    elif distance < 4.0:
        # Check if opponent is charging at us
        if opp_velocity > 1.5 and distance < 3.0:
            # Brace for impact with defending stance
            # Lean into them to absorb momentum
            return {"acceleration": 1.0, "stance": "defending"}

        # Good stamina - apply pressure
        if my_stamina_pct > 0.5:
            # Move forward with extended reach to threaten
            return {"acceleration": 1.5, "stance": "extended"}

        # Moderate stamina - defensive approach
        elif my_stamina_pct > 0.2:
            # Approach slowly in defending stance
            return {"acceleration": 1.0, "stance": "defending"}

        # Low stamina - minimal movement
        else:
            # Hold position, recover in neutral
            return {"acceleration": 0.0, "stance": "neutral"}

    # Far range (4m+) - close distance steadily
    else:
        # We're winning on HP - no need to chase
        if my_hp_pct > opp_hp_pct + 0.2:
            # Let them come to us, maintain center control
            if my_position < arena_width / 2 - 2:
                # We're too far left, move to center
                return {"acceleration": 1.0, "stance": "defending"}
            elif my_position > arena_width / 2 + 2:
                # We're too far right, move to center
                return {"acceleration": -1.0, "stance": "defending"}
            else:
                # We're in center, hold position
                return {"acceleration": 0.0, "stance": "defending"}

        # We need to engage - approach steadily
        if my_stamina_pct > 0.4:
            # Approach with defending stance ready
            return {"acceleration": 2.0, "stance": "defending"}
        else:
            # Low stamina approach
            return {"acceleration": 1.5, "stance": "neutral"}
