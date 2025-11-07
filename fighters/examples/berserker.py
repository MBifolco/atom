"""
Berserker - Relentless All-Out Attack Specialist

Unstoppable force of pure aggression and fury.

Strategy:
- NEVER stops attacking - ignores stamina completely
- Maximum aggression at all times
- Extended stance ALWAYS for maximum damage
- Never retreats, never defends, never hesitates
- Highest collision count of all fighters
- Forces opponents to deal with relentless pressure
"""


def decide(snapshot):
    """
    True berserker - pure relentless aggression without restraint.

    Ignores stamina, ignores HP, just attacks constantly with maximum force.
    Uses extended stance 100% of the time. Never backs down. Never stops.
    The most aggressive fighter possible.
    """
    # Extract minimal information - berserker doesn't need much
    distance = snapshot["opponent"]["distance"]
    my_position = snapshot["you"]["position"]
    opp_velocity = snapshot["opponent"]["velocity"]
    arena_width = snapshot["arena"]["width"]

    # NO STAMINA CHECKS - Berserker ignores exhaustion
    # NO HP CHECKS - Berserker fights to the death

    # Check if near wall - detect early to avoid getting stuck
    near_left_wall = my_position < 2.0  # Early detection
    near_right_wall = my_position > arena_width - 2.0  # Early detection

    # CRITICAL: Avoid wall damage - use neutral stance for better movement
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}  # Maximum escape velocity
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}  # Maximum escape velocity

    # BERSERKER RAGE PATTERNS - ALL MAXIMUM AGGRESSION

    # CLOSE RANGE (<1.5m) - OVERWHELMING FORCE
    if distance < 1.5:
        # Perfect killing range - maximum pressure
        # Check if opponent trying to escape
        if abs(opp_velocity) > 2.0:
            # They're running - CHASE WITH FURY
            if opp_velocity > 0:
                return {"acceleration": 3.5, "stance": "extended"}
            else:
                return {"acceleration": -3.5, "stance": "extended"}
        else:
            # They're fighting - OVERWHELM THEM
            return {"acceleration": 2.5, "stance": "extended"}

    # STRIKING RANGE (1.5-2.5m) - BRUTAL RUSH
    elif distance < 2.5:
        # Rush in with maximum force
        return {"acceleration": 5.0, "stance": "extended"}

    # MEDIUM RANGE (2.5-4m) - RELENTLESS CHARGE
    elif distance < 4.0:
        # Full speed charge with attack ready
        return {"acceleration": 5.0, "stance": "extended"}

    # FAR RANGE (4m+) - MAXIMUM PURSUIT
    else:
        # Sprint at them with killing intent
        # NO STAMINA CHECK - always maximum speed
        return {"acceleration": 5.0, "stance": "extended"}
