"""
Punching Bag - Stationary target for RL agents to learn basics.

Does absolutely nothing. Perfect for learning to approach and strike.
"""

def decide(snapshot):
    """
    Do nothing strategy: stand still, neutral stance.

    This lets RL agents learn the absolute basics:
    - Move into range
    - Switch to extended stance
    - Land hits
    """
    return {
        "acceleration": 0.0,  # Don't move
        "stance": "neutral"   # Neutral stance
    }
