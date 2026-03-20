"""
Swarmer - Pressure fighting style
Constant forward pressure, high work rate, overwhelms with volume
Gets inside and stays there, relentless attack
"""

def decide(state):
    """
    Swarmer strategy:
    - Constant forward pressure
    - High punch volume at close range
    - Minimal defense, maximum offense
    - Stays close to opponent at all times
    """
    # Parse state
    you = state["you"]
    opponent = state["opponent"]

    # Key metrics
    distance = opponent["distance"]
    direction = opponent["direction"]  # -1 (left), +1 (right)
    my_stamina_pct = you["stamina"] / you["max_stamina"]
    my_hp_pct = you["hp"] / you["max_hp"]

    # Combat ranges
    SWARM_RANGE = 0.8  # Ideal swarming distance (very close)
    PRESSURE_RANGE = 1.5  # Maximum acceptable distance

    # Relentless forward pressure
    action = {
        "acceleration": 1.0 * direction,  # Always moving toward opponent
        "stance": "extended"  # Always attacking
    }

    # Distance-based tactics
    if distance < SWARM_RANGE:
        # Perfect swarming distance - unleash combinations
        action["acceleration"] = 0.2 * direction  # Maintain pressure toward opponent
        action["stance"] = "extended"

    elif distance < PRESSURE_RANGE:
        # Good distance - keep pressuring
        action["acceleration"] = 0.6 * direction  # Toward opponent
        action["stance"] = "extended"

    else:
        # Too far - close distance aggressively
        action["acceleration"] = 1.0 * direction  # Toward opponent
        action["stance"] = "neutral"  # Save stamina while closing

    # Stamina management (minimal)
    if my_stamina_pct < 0.1:
        # Emergency stamina recovery
        action["stance"] = "defending"
        action["acceleration"] = 0.0
    elif my_stamina_pct < 0.25:
        # Brief recovery while maintaining pressure
        action["stance"] = "neutral"
        action["acceleration"] = 0.5 * direction  # Toward opponent

    # Never give opponent space to breathe
    if distance > SWARM_RANGE and my_stamina_pct > 0.2:
        action["acceleration"] = 1.0 * direction  # Chase them down toward opponent

    # Go berserk if either fighter is hurt
    if (my_hp_pct < 0.4 or opponent["hp"] / opponent["max_hp"] < 0.4):
        if my_stamina_pct > 0.15:
            action["acceleration"] = 1.0 * direction  # Toward opponent
            action["stance"] = "extended"

    return action