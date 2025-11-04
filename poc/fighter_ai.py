"""
Atom Combat - Fighter AI

Decision functions for different fighter archetypes.
"""

from typing import Dict


def aggressive_fighter(snapshot: Dict) -> Dict:
    """Always rushes forward and extends when close."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    # If close, extend to hit
    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}

    # If have stamina, rush forward
    elif stamina > 2.0:
        return {"acceleration": 5.0, "stance": "neutral"}

    # Low stamina - conserve energy
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def defensive_fighter(snapshot: Dict) -> Dict:
    """Reactive fighter that counters and defends."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    stamina = snapshot["you"]["stamina"]

    # Opponent charging in - brace
    if distance < 2.0 and opp_velocity > 1.0:
        return {"acceleration": 0.0, "stance": "defending"}

    # Close enough to counter
    elif distance < 1.5 and stamina > 3.0:
        return {"acceleration": 2.0, "stance": "extended"}

    # Maintain distance
    elif distance > 5.0:
        return {"acceleration": 2.0, "stance": "neutral"}

    # Default: hold position
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def balanced_fighter(snapshot: Dict) -> Dict:
    """Balanced fighter that adapts to situation."""
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]

    # Emergency retreat
    if my_hp_pct < 0.3:
        return {"acceleration": -4.0, "stance": "neutral"}

    # Winning - press advantage
    if my_hp_pct > opp_hp_pct + 0.2 and my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 4.0, "stance": "neutral"}

    # Losing - play defensive
    if my_hp_pct < opp_hp_pct - 0.2:
        if distance < 2.0:
            return {"acceleration": -3.0, "stance": "defending"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Even match - measured aggression
    if my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 3.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}
