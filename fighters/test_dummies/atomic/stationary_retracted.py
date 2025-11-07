"""
Stationary Retracted

Test dummy that stands perfectly still in retracted stance.
Never moves, always maintains retracted stance.

Purpose: Test retracted stance minimal profile, reduced reach,
and passive properties.
"""


def decide(snapshot):
    """
    Stationary retracted stance test dummy.

    Always returns 0 acceleration and retracted stance.
    """
    return {"acceleration": 0.0, "stance": "retracted"}