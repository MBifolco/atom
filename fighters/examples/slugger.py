"""
Slugger - Power punching style
Heavy hits, aggressive forward pressure, less concerned about defense
Builds momentum for devastating strikes
"""

def decide(state):
    """
    Slugger strategy:
    - Aggressive forward pressure
    - Charges up for heavy hits (builds velocity)
    - Less defensive, more offensive
    - Willing to trade hits for damage
    """
    # Parse state
    you = state["you"]
    opponent = state["opponent"]

    # Key metrics
    distance = opponent["distance"]
    direction = opponent["direction"]  # -1 (left), +1 (right)
    my_velocity = you["velocity"]
    my_stamina_pct = you["stamina"] / you["max_stamina"]
    my_hp_pct = you["hp"] / you["max_hp"]
    opp_hp_pct = opponent["hp"] / opponent["max_hp"]

    # Combat ranges
    POWER_RANGE = 1.0  # Optimal for heavy hits
    CHARGE_DISTANCE = 3.0  # Distance to start building momentum

    # Default aggressive stance
    action = {
        "acceleration": 0.8 * direction,  # Forward pressure toward opponent
        "stance": "extended"
    }

    # Critical stamina - must defend
    if my_stamina_pct < 0.15:
        action["acceleration"] = 0.0
        action["stance"] = "defending"
        return action

    # Build momentum for power hit
    if distance > CHARGE_DISTANCE:
        # Charge forward with full acceleration
        action["acceleration"] = 1.0 * direction  # Toward opponent
        action["stance"] = "neutral"  # Save stamina while charging

    elif distance > POWER_RANGE + 0.5:
        # Continue approach, prepare for strike
        action["acceleration"] = 0.9 * direction  # Toward opponent
        action["stance"] = "neutral" if my_stamina_pct < 0.5 else "extended"

    elif distance <= POWER_RANGE:
        # POWER HIT ZONE!
        if my_stamina_pct > 0.3:
            # Full power strike
            action["acceleration"] = 0.5 * direction  # Toward opponent
            action["stance"] = "extended"
        else:
            # Low stamina - still attack but less acceleration
            action["acceleration"] = 0.2 * direction  # Toward opponent
            action["stance"] = "extended"

    # Berserk mode - go all out when either fighter is low
    if (opp_hp_pct < 0.25 or my_hp_pct < 0.25) and my_stamina_pct > 0.2:
        action["acceleration"] = 1.0 * direction  # Toward opponent
        action["stance"] = "extended"

    # Never retreat (slugger mentality)
    # Keep moving toward opponent even if acceleration would be negative

    return action