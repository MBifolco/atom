"""
Dummy Fighter - Extremely weak opponent for training RL agents.

Just moves forward slowly with neutral stance.
"""

def decide(snapshot):
    """
    Very simple strategy: move forward slowly.

    This gives RL agents an easy opponent to learn basic combat against.
    """
    return {
        "acceleration": 1.0,  # Slow forward movement
        "stance": "neutral"
    }
