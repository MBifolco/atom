"""
Stamina Waster

Test dummy that always uses extended stance to rapidly drain stamina.
Useful for testing how fighters handle low-stamina opponents.

Purpose: Test stamina depletion mechanics, low-stamina behavior,
and how fighters exploit exhausted opponents.
"""


def decide(snapshot):
    """
    Stamina waster test dummy.

    Always uses extended stance to drain stamina quickly.
    Maintains neutral movement.
    """
    # Always use extended stance to waste stamina
    # Extended stance drains stamina faster
    return {"acceleration": 0.0, "stance": "extended"}