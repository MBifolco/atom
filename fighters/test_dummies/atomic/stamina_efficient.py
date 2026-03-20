"""
Conservative stamina management.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Manages stamina conservatively."""
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]

    # Choose stance based on stamina
    if stamina_pct > 0.8:
        stance = "extended"  # Attack when fresh
    elif stamina_pct < 0.3:
        stance = "defending"  # Defend when tired
    else:
        stance = "neutral"  # Normal otherwise

    return {
        "acceleration": 0.0,  # Stationary
        "stance": stance
    }