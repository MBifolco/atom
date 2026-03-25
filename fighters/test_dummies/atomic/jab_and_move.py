"""
Jab and Move — simplified Boxer.
Approaches opponent until within jab range, attacks when stamina allows,
and backs away when fatigued.  Combines approach, range-based attacking,
and basic stamina conservation.
Used for Level 6: Bridge (simplified Expert) training.
"""


def decide(state):
    """Move into jab range, attack, retreat when tired."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]

    if stamina_pct < 0.25:
        # Fatigued — back away and recover
        accel = -direction * 2.0
        stance = "neutral"
    elif distance > 1.2:
        # Outside jab range — approach
        accel = direction * 2.5
        stance = "neutral"
    else:
        # In jab range — attack if stamina allows
        accel = 0.0
        stance = "extended" if stamina_pct > 0.40 else "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
