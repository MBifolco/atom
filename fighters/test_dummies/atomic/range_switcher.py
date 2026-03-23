"""
Range Switcher — tick-based cycle: 45 ticks at safe range (~3m, neutral
stance), then 15 ticks of burst attack (rush in, extended stance).
Teaches the agent to read and react to timing-based patterns.
Used for Level 5: Advanced training.
"""


def decide(state):
    """Cycle between safe-range kiting and burst attacks."""
    tick = state["tick"]
    direction = state["opponent"]["direction"]
    distance = state["opponent"]["distance"]

    cycle = tick % 60

    if cycle < 45:
        # Safe range phase — maintain ~3m distance
        if distance < 2.5:
            accel = -direction * 2.0
        elif distance > 3.5:
            accel = direction * 1.0
        else:
            accel = 0.0
        stance = "neutral"
    else:
        # Burst attack phase — rush in
        accel = direction * 4.0
        stance = "extended"

    return {
        "acceleration": accel,
        "stance": stance,
    }
