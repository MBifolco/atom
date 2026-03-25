"""
Close Range Brawler — simplified Swarmer.
Rushes the opponent aggressively and stays in extended stance almost always.
Reduces acceleration when very close to avoid overshooting, and only
defends at near-zero stamina.  Combines aggressive closing, proximity
control, and emergency stamina defense.
Used for Level 6: Bridge (simplified Expert) training.
"""


def decide(state):
    """Rush in aggressively, stay extended, defend only when empty."""
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    stamina_pct = state["you"]["stamina"] / state["you"]["max_stamina"]

    if stamina_pct < 0.10:
        # Nearly empty — emergency defend
        accel = 0.0
        stance = "defending"
    elif distance < 0.5:
        # Very close — ease off to stay in range
        accel = direction * 0.3
        stance = "extended"
    else:
        # Close the distance aggressively
        accel = direction * 0.8
        stance = "extended"

    return {
        "acceleration": accel,
        "stance": stance,
    }
