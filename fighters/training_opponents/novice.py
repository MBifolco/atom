"""
Novice - Basic competent fighter.

Difficulty: Level 4

Has the fundamentals:
- Moves toward opponent
- Uses extended stance at correct range
- Basic stamina awareness
- But: predictable, no adaptation, no defense
"""

def decide(snapshot):
    """
    Competent but predictable strategy.

    Good enough to punish passive play, but exploitable by tactical fighters.
    """
    distance = snapshot["opponent"]["distance"]
    my_stamina = snapshot["you"]["stamina"]
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]

    # Basic wall avoidance
    if my_position < 1.0:
        acceleration = 4.0  # Move away from left wall
    elif my_position > arena_width - 1.0:
        acceleration = -4.0  # Move away from right wall
    elif distance > 2.0:
        acceleration = 3.0  # Move toward opponent
    elif distance < 0.5:
        acceleration = -1.0  # Back up if too close
    else:
        acceleration = 0.0  # Good distance, hold position

    # Proper stance timing
    if distance < 1.2 and my_stamina > 3.0:
        stance = "extended"  # Strike when in range and have stamina
    elif my_stamina < 2.0:
        stance = "neutral"  # Regen stamina when low
    else:
        stance = "neutral"

    return {
        "acceleration": acceleration,
        "stance": stance
    }
