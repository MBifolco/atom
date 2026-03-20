"""
Smooth side-to-side movement pattern.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Smooth oscillating movement side to side."""
    # Use tick to create smooth oscillation
    # Ticks run at 60 Hz, so 240 ticks = 4 second cycle
    if "tick" in state:
        t = (state["tick"] / 60.0) % 4.0  # 4 second cycle
    elif "you" in state and "position" in state["you"]:
        # Use own position as proxy for oscillation
        t = (abs(state["you"]["position"]) * 2.0) % 4.0
    else:
        # Fallback: simple alternation
        t = 0.0

    # Smooth acceleration based on cycle
    if t < 2.0:
        # Moving right
        accel = 1.5 * (1.0 - abs(t - 1.0))  # Peak at t=1
    else:
        # Moving left
        accel = -1.5 * (1.0 - abs(t - 3.0))  # Peak at t=3

    # Vary stance based on distance
    distance = abs(state["opponent"]["distance"])
    if distance < 1.0:
        stance = "defending"
    elif distance < 2.0:
        stance = "extended"
    else:
        stance = "neutral"

    return {
        "acceleration": accel,
        "stance": stance
    }