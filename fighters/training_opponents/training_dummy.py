"""
Training Dummy - Ultra-light stationary target.

Perfect first opponent for RL agents to learn against.
Configure this as a 50kg fighter so it takes more damage.
"""

def decide(snapshot):
    """
    Stand still and do nothing.

    When used as 50kg fighter, it will take significant damage
    from any contact, making it easy for RL agents to win.
    """
    return {
        "acceleration": 0.0,
        "stance": "neutral"
    }
