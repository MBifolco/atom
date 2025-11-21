"""
Stationary fighter in extended stance - stands still while attacking.
Used for Level 1: Fundamentals training.
"""

def decide(state):
    """Always returns extended stance, no movement."""
    return {
        "acceleration": 0.0,
        "stance": "extended"
    }