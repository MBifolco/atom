"""
Stationary fighter in defending stance - stands still while defending.
Used for Level 1: Fundamentals training.
"""

def decide(state):
    """Always returns defending stance, no movement."""
    return {
        "acceleration": 0.0,
        "stance": "defending"
    }