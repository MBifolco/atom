"""
Pursues aggressively in extended stance until stamina drops below 20%,
then recovers in neutral. Dead zone between 20%-40% uses moderate pursuit.
Teaches the agent to exploit stamina depletion windows.
Used for Level 3: Intermediate training.
"""


def decide(state):
    """Burn stamina attacking, then recover."""
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]
    direction = state["opponent"]["direction"]

    if direction == 0:
        direction = 1.0  # Default direction if overlapping

    if stamina_pct > 0.4:
        # Full attack mode
        accel = direction * 3.0
        stance = "extended"
    elif stamina_pct < 0.2:
        # Recovering -- stop and rest
        accel = 0.0
        stance = "neutral"
    else:
        # Winding down -- moderate pursuit
        accel = direction * 1.5
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance,
    }
