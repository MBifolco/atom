"""
Stamina Cycler

Test dummy that cycles between high-drain and recovery phases.
Attacks in extended stance until stamina hits 20%, then recovers
in retracted stance until 90%.

Purpose: Test stamina management detection, exploitation of
recovery phases, and adaptation to changing aggression levels.
"""


def decide(snapshot):
    """
    Stamina cycler test dummy.

    Cycles between aggressive (extended) and recovery (retracted) phases
    based on stamina thresholds.
    """
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]

    # Use tick to track current phase
    tick = snapshot["tick"]

    # Simple state machine based on stamina
    if my_stamina_pct > 0.9:
        # High stamina: Attack phase
        stance = "extended"
    elif my_stamina_pct < 0.2:
        # Low stamina: Recovery phase
        stance = "retracted"
    else:
        # Mid stamina: Continue current pattern
        # Check if we were attacking or recovering based on stamina trend
        if my_stamina_pct > 0.5:
            stance = "extended"  # Still enough to attack
        else:
            stance = "retracted"  # Need to recover

    # Stationary to focus on stamina pattern
    return {"acceleration": 0.0, "stance": stance}