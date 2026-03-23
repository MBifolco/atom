"""
Stamina Punisher — monitors opponent stamina and exploits fatigue.
Attacks tired opponents aggressively, maintains safe distance against
fresh opponents, and stays cautious in the dead zone.
Used for Level 5: Advanced training.
"""


def decide(state):
    """Punish opponent when their stamina is low."""
    opponent = state["opponent"]
    direction = opponent["direction"]
    distance = opponent["distance"]
    opp_stamina_pct = opponent["stamina"] / opponent["max_stamina"]

    if opp_stamina_pct < 0.4:
        # Opponent tired — attack aggressively
        accel = direction * 3.0
        stance = "extended"
    elif opp_stamina_pct > 0.7:
        # Opponent fresh — maintain safe distance ~3m
        if distance < 2.5:
            accel = -direction * 2.0
        elif distance > 3.5:
            accel = direction * 1.0
        else:
            accel = 0.0
        stance = "neutral"
    else:
        # Dead zone — cautious at ~2m
        if distance > 2.5:
            accel = direction * 1.0
        elif distance < 1.5:
            accel = -direction * 1.0
        else:
            accel = 0.0
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
