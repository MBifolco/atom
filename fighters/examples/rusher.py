"""
Rusher - Aggressive Fighter

Relentless forward pressure with constant attacking stance.

Strategy:
- ALWAYS rushes forward with maximum aggression
- Uses extended stance while advancing for reach advantage
- Only stops to avoid wall damage
- Never retreats - commits fully to offense
- Overwhelms opponents with constant pressure
"""


def decide(snapshot):
    """
    Hyper-aggressive rushing fighter.

    Constantly advances with extended stance, trading stamina and defense
    for maximum offensive pressure. Only backs off from walls.
    """
    # Extract key information
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]
    arena_width = snapshot["arena"]["width"]

    # Check if we're near walls - detect early to avoid getting stuck
    near_left_wall = my_position < 2.0  # Early detection
    near_right_wall = my_position > arena_width - 2.0  # Early detection

    # CRITICAL: Avoid wall damage - only time we back off
    if near_left_wall:
        return {"acceleration": 5.0, "stance": "neutral"}  # Strong escape from left wall
    if near_right_wall:
        return {"acceleration": -5.0, "stance": "neutral"}  # Strong escape from right wall

    # CORE RUSHER BEHAVIOR: Always advance with aggression
    if distance < 1.5:
        # We're in striking range - ATTACK!

        # Perfect range - maximum damage output
        if my_stamina_pct > 0.05:  # Attack even at very low stamina
            # Maintain slight forward pressure while attacking
            return {"acceleration": 1.0, "stance": "extended"}

        # Completely exhausted - brief recovery but stay close
        else:
            # Still push forward even when recovering
            return {"acceleration": 0.5, "stance": "neutral"}

    # Medium range (1.5-3m) - close the gap aggressively
    elif distance < 3.0:
        # Rush in with extended stance for immediate attack
        if my_stamina_pct > 0.15:
            # Full speed rush with attack ready
            return {"acceleration": 4.0, "stance": "extended"}

        # Low stamina but keep pressure
        elif my_stamina_pct > 0.05:
            # Advance with defending to conserve some stamina
            return {"acceleration": 3.0, "stance": "defending"}

        # Critical stamina - advance in neutral
        else:
            # Still moving forward, never backing down
            return {"acceleration": 2.0, "stance": "neutral"}

    # Far range (3m+) - maximum speed rush
    else:
        # Check if we're winning - if so, even more aggressive
        if opp_hp_pct < my_hp_pct:
            # Smell blood - go for the kill
            return {"acceleration": 5.0, "stance": "extended"}

        # Normal rush approach
        if my_stamina_pct > 0.2:
            # Sprint forward with extended reach
            return {"acceleration": 4.5, "stance": "extended"}

        # Low stamina rush
        elif my_stamina_pct > 0.1:
            # Keep rushing but in defending stance
            return {"acceleration": 3.5, "stance": "defending"}

        # Minimal stamina - still advance
        else:
            # Never stop moving forward
            return {"acceleration": 2.5, "stance": "neutral"}
