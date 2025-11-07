"""
Stationary Defending

Test dummy that stands perfectly still in defending stance.
Never moves, always maintains defending stance.

Purpose: Test defending stance damage reduction (1.63x defense),
stamina cost, and defensive properties.
"""


def decide(snapshot):
    """
    Stationary defending stance test dummy.

    Always returns 0 acceleration and defending stance.
    """
    return {"acceleration": 0.0, "stance": "defending"}