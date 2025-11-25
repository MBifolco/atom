"""
Always uses extended stance to waste stamina.
Used for Level 3: Intermediate training.
"""

def decide(state):
    """Always uses extended stance to deplete stamina quickly."""
    return {
        "acceleration": 0.0,  # Stationary
        "stance": "extended"  # Always extended to waste stamina
    }