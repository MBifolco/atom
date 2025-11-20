"""
Counter Puncher - Defensive boxing style
Waits for opponent mistakes, excellent defense, strikes at openings
Patient and tactical, uses opponent's aggression against them
"""

def decide(state):
    """
    Counter puncher strategy:
    - Primarily defensive stance
    - Waits for opponent to attack
    - Counters when opponent is extended and vulnerable
    - Excellent stamina management through defense
    """
    # Parse state
    you = state["you"]
    opponent = state["opponent"]

    # Key metrics
    distance = opponent["distance"]
    direction = opponent["direction"]  # -1 (left), +1 (right)
    opp_velocity = opponent["velocity"]
    my_stamina_pct = you["stamina"] / you["max_stamina"]
    opp_stamina_pct = opponent["stamina"] / opponent["max_stamina"]

    # Combat ranges
    COUNTER_RANGE = 0.9  # Perfect counter distance
    COMFORT_ZONE = 2.0  # Preferred defensive distance

    # Default defensive posture
    action = {
        "acceleration": 0.0,
        "stance": "defending"  # Default to defense
    }

    # Detect opponent charging (good counter opportunity)
    opponent_charging = abs(opp_velocity) > 1.5 and distance < 2.5

    # Counter opportunity detection
    opponent_vulnerable = (
        opp_stamina_pct < 0.4 or  # Low stamina
        opponent_charging or  # Charging in
        (distance < 1.5 and abs(opp_velocity) > 0.5)  # Close and moving
    )

    # Execute counter
    if opponent_vulnerable and my_stamina_pct > 0.4:
        if distance < COUNTER_RANGE:
            # Perfect counter strike!
            action["stance"] = "extended"
            action["acceleration"] = 0.4 * direction  # Toward opponent
        elif distance < COUNTER_RANGE + 0.5:
            # Step into the counter
            action["stance"] = "extended"
            action["acceleration"] = 0.6 * direction  # Toward opponent
        else:
            # Prepare counter position
            action["stance"] = "defending"
            action["acceleration"] = -0.2 * direction  # Away from opponent

    # Distance management
    elif distance < COMFORT_ZONE - 0.5:
        # Too close - create space
        action["acceleration"] = -0.5 * direction  # Away from opponent
        action["stance"] = "defending"

    elif distance > COMFORT_ZONE + 1.0:
        # Too far - maintain optimal distance
        action["acceleration"] = 0.3 * direction  # Toward opponent
        action["stance"] = "neutral"  # Save stamina while repositioning

    else:
        # Perfect defensive distance
        action["stance"] = "defending"
        action["acceleration"] = 0.0

    # Stamina advantage press
    if my_stamina_pct > 0.8 and opp_stamina_pct < 0.3:
        # Opponent is exhausted - time to attack
        if distance > COUNTER_RANGE:
            action["acceleration"] = 0.7 * direction  # Toward opponent
            action["stance"] = "neutral"
        else:
            action["acceleration"] = 0.3 * direction  # Toward opponent
            action["stance"] = "extended"

    return action