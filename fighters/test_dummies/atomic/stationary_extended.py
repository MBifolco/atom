"""
Stationary Extended

Test dummy that stands perfectly still in extended stance.
Never moves, always maintains extended stance.

Purpose: Test extended stance reach, stamina drain,
and attack damage at maximum extension.
"""


def decide(snapshot):
    """
    Stationary extended stance test dummy.

    Always returns 0 acceleration and extended stance.
    """
    return {"acceleration": 0.0, "stance": "extended"}