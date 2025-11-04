"""
Wanderer - Random movement, no fighting strategy.

Difficulty: Level 2 (after training_dummy)

Just wanders around randomly. Sometimes accidentally hits things.
Good for learning that movement and positioning matter.
"""

import random

def decide(snapshot):
    """
    Random movement strategy.

    Moves randomly but doesn't strategically choose stances for combat.
    Helps AI learn that position affects collision opportunities.
    """
    # Random acceleration between -2 and +2 (not full speed)
    acceleration = random.uniform(-2.0, 2.0)

    # Mostly neutral, occasionally other stances
    stance_choice = random.random()
    if stance_choice < 0.7:
        stance = "neutral"
    elif stance_choice < 0.85:
        stance = "extended"
    elif stance_choice < 0.95:
        stance = "retracted"
    else:
        stance = "defending"

    return {
        "acceleration": acceleration,
        "stance": stance
    }
