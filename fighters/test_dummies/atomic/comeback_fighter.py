"""
Starts conservative and gets more aggressive over time.
Linear aggression ramp over 200 ticks: defend early, transition to
neutral mid-range play, then go full aggression. Wall-aware when retreating.
Used for Level 5: Advanced training.
"""


def decide(state):
    """Ramp aggression linearly over the first 200 ticks."""
    tick = state["tick"]
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]
    position = state["you"]["position"]
    arena_width = state["arena"]["width"]

    aggression = min(1.0, tick / 200.0)

    if direction == 0:
        direction = 1.0  # Default direction if overlapping

    if aggression < 0.3:
        # Conservative -- retreat and defend
        accel = -direction * 1.5
        stance = "defending"
        # Wall awareness
        if position < 1.0:
            accel = 1.5
        elif position > arena_width - 1.0:
            accel = -1.5
    elif aggression < 0.7:
        # Neutral -- maintain ~2m, use extended when close
        if distance > 2.5:
            accel = direction * 1.5
        elif distance < 1.5:
            accel = -direction * 1.0
        else:
            accel = 0.0
        stance = "extended" if distance < 1.0 else "neutral"
    else:
        # Full aggression
        accel = direction * 4.0
        stance = "extended"

    return {
        "acceleration": accel,
        "stance": stance,
    }
