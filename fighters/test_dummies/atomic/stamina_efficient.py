"""
Stamina Efficient

Test dummy that maintains optimal stamina usage.
Uses neutral stance primarily, only switching to extended
when stamina is above 80%.

Purpose: Test against conservative stamina management,
long-duration combat, and endurance strategies.
"""


def decide(snapshot):
    """
    Stamina efficient test dummy.

    Conserves stamina by using neutral stance mostly,
    only attacking when stamina is very high.
    """
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["stamina_max"]

    # Conservative stamina management
    if my_stamina_pct > 0.8:
        # High stamina: Can afford to attack briefly
        stance = "extended"
    elif my_stamina_pct < 0.3:
        # Low stamina: Recovery mode
        stance = "retracted"
    else:
        # Normal operation: Neutral for efficiency
        stance = "neutral"

    # Stationary to focus on stamina pattern
    return {"acceleration": 0.0, "stance": stance}