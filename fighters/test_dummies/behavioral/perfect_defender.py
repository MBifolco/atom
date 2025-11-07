"""
Perfect Defender

Behavioral fighter that demonstrates optimal defensive play.
Maintains defending stance, manages stamina carefully,
backs away when pressured, and only counters when safe.

Purpose: Test against maximum defense strategies,
validate damage reduction mechanics, and benchmark
offensive capabilities.
"""


def decide(snapshot):
    """
    Perfect defender behavioral fighter.

    Prioritizes defense above all else, only attacks when completely safe.
    """
    my_position = snapshot["you"]["position"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["stamina_max"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["hp_max"]

    opponent_position = snapshot["opponent"]["position"]
    opponent_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["stamina_max"]

    distance = abs(opponent_position - my_position)
    arena_width = snapshot["arena"]["width"]

    # Wall detection
    near_left_wall = my_position < 2.0
    near_right_wall = my_position > arena_width - 2.0

    # Escape from walls first
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Distance-based defensive strategy
    if distance < 1.5:
        # Too close: Back away while defending
        if opponent_position > my_position:
            acceleration = -3.0
        else:
            acceleration = 3.0

        # Counter-attack only if opponent is exhausted
        if opponent_stamina_pct < 0.1 and my_stamina_pct > 0.5:
            stance = "extended"
        else:
            stance = "defending"

    elif distance < 3.0:
        # Optimal defensive range: Hold position
        acceleration = 0.0

        # Strong defense unless we have advantage
        if opponent_stamina_pct < 0.2 and my_stamina_pct > 0.6:
            stance = "extended"  # Punish exhaustion
        else:
            stance = "defending"

    else:
        # Far away: Maintain distance
        acceleration = 0.0

        # Recovery stance when safe
        if my_stamina_pct < 0.3:
            stance = "retracted"
        else:
            stance = "defending"

    return {"acceleration": acceleration, "stance": stance}