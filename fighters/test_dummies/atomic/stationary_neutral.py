"""
Stationary fighter in neutral stance - stands still.
Used for Level 1: Fundamentals training.
"""

def decide(state):
    """Always returns neutral stance, no movement."""
    return {
        "acceleration": 0.0,
        "stance": "neutral"
    }