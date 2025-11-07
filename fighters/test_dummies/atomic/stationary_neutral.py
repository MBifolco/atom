"""
Stationary Neutral

Test dummy that stands perfectly still in neutral stance.
Never moves, never changes stance.

Purpose: Test basic collision detection, default damage values,
and neutral stance properties.
"""


def decide(snapshot):
    """
    Stationary neutral stance test dummy.

    Always returns 0 acceleration and neutral stance.
    """
    return {"acceleration": 0.0, "stance": "neutral"}