"""
Always uses extended stance to waste stamina.

NOTE: Intentionally unused in curriculum. Functionally identical to
stationary_extended. Replaced by stamina_burner for active pursuit.
"""

def decide(state):
    """Always uses extended stance to deplete stamina quickly."""
    return {
        "acceleration": 0.0,  # Stationary
        "stance": "extended"  # Always extended to waste stamina
    }