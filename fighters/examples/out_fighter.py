"""
Out-Fighter - Long range boxing style
Maintains distance, uses reach advantage, hit and move tactics
Excellent footwork, focuses on points over power
"""

def decide(state):
    """
    Out-fighter strategy:
    - Maintain optimal distance (just at edge of range)
    - Hit and immediately move away
    - Excellent footwork and distance control
    - Accumulate damage over time rather than knockout
    """
    # Parse state
    you = state["you"]
    opponent = state["opponent"]

    # Key metrics
    distance = opponent["distance"]
    direction = opponent["direction"]  # -1 (left), +1 (right)
    my_velocity = you["velocity"]
    opp_velocity = opponent["velocity"]
    my_stamina_pct = you["stamina"] / you["max_stamina"]
    my_position = you["position"]

    # Combat ranges
    STRIKE_RANGE = 1.3  # Outer edge of striking range
    SAFE_DISTANCE = 2.8  # Optimal kiting distance
    DANGER_ZONE = 0.8  # Too close - need to escape

    # Default mobile stance
    action = {
        "acceleration": 0.0,
        "stance": "neutral"
    }

    # Emergency escape if too close
    if distance < DANGER_ZONE:
        action["acceleration"] = -1.0 * direction  # Full retreat away from opponent
        action["stance"] = "neutral"  # Focus on movement
        return action

    # Hit and run mechanics
    if my_stamina_pct > 0.5:
        if distance > STRIKE_RANGE + 0.2 and distance < SAFE_DISTANCE:
            # Move in for a quick strike
            action["acceleration"] = 0.8 * direction  # Toward opponent
            action["stance"] = "neutral"

        elif distance <= STRIKE_RANGE and distance > DANGER_ZONE:
            # Strike and prepare to move
            action["stance"] = "extended"
            # Check if moving toward opponent
            moving_toward = (my_velocity * direction) > 0.5
            if moving_toward:
                action["acceleration"] = -0.7 * direction  # Hit and move back away
            else:
                action["acceleration"] = 0.3 * direction  # Quick in toward opponent

        elif distance > SAFE_DISTANCE:
            # Too far - maintain optimal distance
            action["acceleration"] = 0.4 * direction  # Toward opponent
            action["stance"] = "neutral"

        else:
            # Good distance - circle and look for openings
            action["acceleration"] = -0.3 * direction  # Away from opponent
            action["stance"] = "neutral"

    # Low stamina - focus on distance and recovery
    else:
        if distance < SAFE_DISTANCE:
            action["acceleration"] = -0.6 * direction  # Away from opponent
            action["stance"] = "defending" if my_stamina_pct < 0.3 else "neutral"
        else:
            action["acceleration"] = 0.0
            action["stance"] = "defending"  # Recover stamina

    # Use arena space wisely - avoid backing into edges
    arena_edge_close = my_position < 2.0 or my_position > 10.0
    # If close to edge and trying to move away from opponent, reverse to move toward them instead
    if arena_edge_close and (action["acceleration"] * direction) < 0:
        # About to back into wall - move toward opponent instead
        action["acceleration"] = 0.5 * direction  # Toward opponent
        action["stance"] = "extended" if distance < STRIKE_RANGE else "neutral"

    # Endgame - maintain distance and points lead
    if you["hp"] > opponent["hp"] * 1.2:
        # We're winning - stay safe
        if distance < SAFE_DISTANCE:
            action["acceleration"] = -0.5 * direction  # Away from opponent
        action["stance"] = "defending" if my_stamina_pct < 0.6 else "neutral"

    return action