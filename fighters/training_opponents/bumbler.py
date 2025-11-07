"""
Bumbler - Tries to fight but does it poorly.

Difficulty: Level 3

Has basic combat instincts but terrible execution:
- Moves toward opponent but not strategically
- Uses extended stance but at wrong times
- Doesn't manage stamina
- No defensive awareness
"""

def decide(snapshot):
    """
    Poor combat strategy with bad timing.

    Helps AI learn that WHEN you do things matters, not just WHAT you do.

    Now aware of stamina mechanics but manages it poorly:
    - Spams extended stance too much
    - Only backs off when completely exhausted
    - No strategic timing or defensive play
    """
    # Calculate distance properly
    my_position = snapshot["you"]["position"]
    opponent_position = snapshot["opponent"]["position"]
    distance = abs(opponent_position - my_position)

    my_stamina = snapshot["you"]["stamina"]
    max_stamina = snapshot["you"]["stamina_max"]  # Correct field name
    stamina_pct = my_stamina / max_stamina if max_stamina > 0 else 0

    # Always tries to move forward (no retreat logic)
    # Doesn't stop properly, will collide clumsily
    if distance > 0.5:
        # Move toward opponent aggressively
        if opponent_position > my_position:
            acceleration = 3.5  # Move right toward opponent
        else:
            acceleration = -3.5  # Move left toward opponent
    else:
        # Keep pushing even when close (clumsy!)
        if opponent_position > my_position:
            acceleration = 1.0  # Push right
        else:
            acceleration = -1.0  # Push left

    # Uses extended stance too aggressively (poor stamina management)
    # Knows can't attack at 0 stamina, but waits too long to back off
    if stamina_pct < 0.1:
        # Exhausted - forced to regenerate
        stance = "neutral"
    elif distance < 4.0:
        # Spam attack whenever in range (drains stamina fast)
        stance = "extended"
    else:
        stance = "neutral"

    # Still has poor execution:
    # - No optimal stamina management (should regen earlier)
    # - No wall avoidance
    # - No HP awareness
    # - Never uses defensive stances strategically

    return {
        "acceleration": acceleration,
        "stance": stance
    }
