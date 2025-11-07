"""
Test Dummy Utilities

Helper functions for creating test dummy fighters.
"""

from typing import Dict, Any, Tuple, Optional


def get_opponent_position(snapshot: Dict[str, Any]) -> float:
    """Get the opponent's position from snapshot."""
    return snapshot["opponent"]["position"]


def get_my_position(snapshot: Dict[str, Any]) -> float:
    """Get my position from snapshot."""
    return snapshot["you"]["position"]


def get_distance(snapshot: Dict[str, Any]) -> float:
    """Get distance to opponent."""
    return snapshot["opponent"]["distance"]


def get_my_velocity(snapshot: Dict[str, Any]) -> float:
    """Get my current velocity."""
    return snapshot["you"]["velocity"]


def get_opponent_velocity(snapshot: Dict[str, Any]) -> float:
    """Get opponent's current velocity."""
    return snapshot["opponent"]["velocity"]


def get_arena_width(snapshot: Dict[str, Any]) -> float:
    """Get arena width."""
    return snapshot["arena"]["width"]


def get_my_hp_percent(snapshot: Dict[str, Any]) -> float:
    """Get my HP as percentage."""
    return snapshot["you"]["hp"] / snapshot["you"]["max_hp"]


def get_opponent_hp_percent(snapshot: Dict[str, Any]) -> float:
    """Get opponent HP as percentage."""
    return snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]


def get_my_stamina_percent(snapshot: Dict[str, Any]) -> float:
    """Get my stamina as percentage."""
    return snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]


def get_opponent_stamina_percent(snapshot: Dict[str, Any]) -> float:
    """Get opponent stamina as percentage."""
    return snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]


def is_near_left_wall(snapshot: Dict[str, Any], margin: float = 2.0) -> bool:
    """Check if near left wall."""
    return get_my_position(snapshot) < margin


def is_near_right_wall(snapshot: Dict[str, Any], margin: float = 2.0) -> bool:
    """Check if near right wall."""
    arena_width = get_arena_width(snapshot)
    return get_my_position(snapshot) > arena_width - margin


def is_near_any_wall(snapshot: Dict[str, Any], margin: float = 2.0) -> bool:
    """Check if near any wall."""
    return is_near_left_wall(snapshot, margin) or is_near_right_wall(snapshot, margin)


def approach_opponent(snapshot: Dict[str, Any], speed: float = 2.0) -> float:
    """Calculate acceleration to approach opponent."""
    my_pos = get_my_position(snapshot)
    opp_pos = get_opponent_position(snapshot)

    if my_pos < opp_pos:
        return speed  # Move right
    elif my_pos > opp_pos:
        return -speed  # Move left
    else:
        return 0.0  # At opponent


def flee_from_opponent(snapshot: Dict[str, Any], speed: float = 3.0) -> float:
    """Calculate acceleration to flee from opponent."""
    my_pos = get_my_position(snapshot)
    opp_pos = get_opponent_position(snapshot)

    if my_pos < opp_pos:
        accel = -speed  # Move left
    elif my_pos > opp_pos:
        accel = speed  # Move right
    else:
        accel = speed  # Emergency escape

    # Don't flee into walls
    if is_near_left_wall(snapshot) and accel < 0:
        accel = speed
    elif is_near_right_wall(snapshot) and accel > 0:
        accel = -speed

    return accel


def maintain_distance(snapshot: Dict[str, Any], target_distance: float,
                     approach_speed: float = 2.0, retreat_speed: float = 2.0,
                     tolerance: float = 0.5) -> float:
    """Calculate acceleration to maintain target distance."""
    distance = get_distance(snapshot)
    my_pos = get_my_position(snapshot)
    opp_pos = get_opponent_position(snapshot)

    if distance > target_distance + tolerance:
        # Too far - approach
        if my_pos < opp_pos:
            return approach_speed
        else:
            return -approach_speed
    elif distance < target_distance - tolerance:
        # Too close - retreat
        if my_pos < opp_pos:
            return -retreat_speed
        else:
            return retreat_speed
    else:
        # Perfect distance
        return 0.0


def shuttle_movement(snapshot: Dict[str, Any], left_bound: float, right_bound: float,
                     speed: float = 2.0) -> float:
    """Calculate acceleration for shuttle movement pattern."""
    my_pos = get_my_position(snapshot)
    my_vel = get_my_velocity(snapshot)

    if my_pos <= left_bound:
        return speed  # Move right
    elif my_pos >= right_bound:
        return -speed  # Move left
    elif my_vel > 0:  # Moving right
        if my_pos >= right_bound - 0.5:
            return -speed  # Start turning
        else:
            return speed  # Continue right
    else:  # Moving left
        if my_pos <= left_bound + 0.5:
            return speed  # Start turning
        else:
            return -speed  # Continue left


def circle_movement(snapshot: Dict[str, Any], direction: str = "left",
                   speed: float = 2.0) -> float:
    """Calculate acceleration for circular movement."""
    my_pos = get_my_position(snapshot)
    arena_width = get_arena_width(snapshot)

    if direction == "left":
        accel = -speed
        # Bounce off left wall
        if my_pos < 1.0:
            accel = speed
    else:
        accel = speed
        # Bounce off right wall
        if my_pos > arena_width - 1.0:
            accel = -speed

    return accel


def mirror_opponent_movement(snapshot: Dict[str, Any]) -> float:
    """Mirror the opponent's acceleration."""
    opp_vel = get_opponent_velocity(snapshot)
    # Estimate opponent's acceleration from velocity
    return opp_vel * 0.5  # Rough approximation


def counter_opponent_movement(snapshot: Dict[str, Any]) -> float:
    """Move opposite to opponent's movement."""
    opp_vel = get_opponent_velocity(snapshot)
    # Move opposite direction
    return -opp_vel * 0.5


def cycle_stances(tick: int, cycle_length: int = 20) -> str:
    """Cycle through stances based on tick count."""
    stances = ["neutral", "extended", "retracted", "defending"]
    phase = (tick // cycle_length) % len(stances)
    return stances[phase]


def random_stance(seed: int) -> str:
    """Get a pseudo-random stance based on seed."""
    stances = ["neutral", "extended", "retracted", "defending"]
    return stances[seed % len(stances)]


def stance_by_distance(snapshot: Dict[str, Any]) -> str:
    """Choose stance based on distance to opponent."""
    distance = get_distance(snapshot)

    if distance < 1.5:
        return "extended"  # Close - attack
    elif distance < 3.0:
        return "neutral"  # Medium - balanced
    elif distance < 5.0:
        return "defending"  # Far - defensive
    else:
        return "retracted"  # Very far - minimal profile


def stance_by_stamina(snapshot: Dict[str, Any]) -> str:
    """Choose stance based on stamina level."""
    stamina_pct = get_my_stamina_percent(snapshot)

    if stamina_pct > 0.6:
        return "extended"  # High stamina - aggressive
    elif stamina_pct > 0.3:
        return "neutral"  # Medium stamina - balanced
    else:
        return "defending"  # Low stamina - defensive


def stance_by_hp_difference(snapshot: Dict[str, Any]) -> str:
    """Choose stance based on HP difference."""
    my_hp = get_my_hp_percent(snapshot)
    opp_hp = get_opponent_hp_percent(snapshot)

    if my_hp > opp_hp + 0.2:
        return "extended"  # Winning - aggressive
    elif my_hp < opp_hp - 0.2:
        return "defending"  # Losing - defensive
    else:
        return "neutral"  # Even - balanced