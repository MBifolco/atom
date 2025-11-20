"""
Boxer - Classic boxing style
Quick jabs, good footwork, excellent stamina management
Uses hit-and-move tactics with disciplined stance changes
"""

def decide(state):
    """
    Boxer strategy:
    - Quick jabs when in range
    - Good footwork to control distance
    - Disciplined stamina management
    - Switches to defending when low on stamina
    """
    # Parse state
    you = state["you"]
    opponent = state["opponent"]

    # Key metrics
    distance = opponent["distance"]
    direction = opponent["direction"]  # -1 (left), +1 (right)
    my_stamina_pct = you["stamina"] / you["max_stamina"]
    opp_hp_pct = opponent["hp"] / opponent["max_hp"]

    # Optimal jabbing distance
    JAB_RANGE = 1.2
    SAFE_DISTANCE = 2.5

    # Stamina management thresholds
    LOW_STAMINA = 0.3
    HIGH_STAMINA = 0.7

    # Default action
    action = {
        "acceleration": 0.0,
        "stance": "neutral"
    }

    # Stamina recovery mode
    if my_stamina_pct < LOW_STAMINA:
        # Back off and defend to recover stamina
        if distance < SAFE_DISTANCE:
            action["acceleration"] = -0.5 * direction  # Back away from opponent
        action["stance"] = "defending"  # Regenerate stamina
        return action

    # Attack mode (good stamina)
    if my_stamina_pct > HIGH_STAMINA:
        if distance > JAB_RANGE + 0.3:
            # Move in for the jab
            action["acceleration"] = 0.7 * direction  # Toward opponent
            action["stance"] = "neutral"  # Save stamina while moving
        elif distance < JAB_RANGE:
            # In range - throw the jab!
            action["acceleration"] = 0.3 * direction  # Small forward pressure toward opponent
            action["stance"] = "extended"  # Attack!
        else:
            # Perfect distance - maintain and jab
            action["acceleration"] = 0.0
            action["stance"] = "extended"

    # Conservative mode (medium stamina)
    else:
        if distance < JAB_RANGE:
            # Quick jab if opponent is close
            action["stance"] = "extended"
            action["acceleration"] = 0.0
        elif distance < SAFE_DISTANCE:
            # Maintain safe distance
            action["acceleration"] = -0.3 * direction  # Away from opponent
            action["stance"] = "neutral"
        else:
            # Recovery stance at safe distance
            action["stance"] = "defending"
            action["acceleration"] = 0.0

    # Pursuit mode - chase down low HP opponent
    if opp_hp_pct < 0.3 and my_stamina_pct > 0.5:
        if distance > JAB_RANGE:
            action["acceleration"] = 1.0 * direction  # Full pursuit toward opponent
            action["stance"] = "neutral"
        else:
            action["acceleration"] = 0.5 * direction  # Toward opponent
            action["stance"] = "extended"  # Finish them

    return action