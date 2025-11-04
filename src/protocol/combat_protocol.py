"""
Atom Combat - Combat Protocol

The contract between fighters and the Arena.
Defines what fighters can sense and what actions they can take.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Snapshot:
    """
    What a fighter perceives at a given tick.
    This is the ONLY information available to the fighter's decision function.
    """
    tick: int

    # Fighter's own state
    you: Dict[str, Any]  # {position, velocity, hp, max_hp, stamina, max_stamina, stance}

    # Opponent information (may be filtered/bucketed by sensors)
    opponent: Dict[str, Any]  # {distance, velocity, hp, max_hp, stamina, max_stamina, stance_hint}

    # Arena information
    arena: Dict[str, Any]  # {width}


@dataclass
class Action:
    """
    What a fighter can do each tick.
    """
    acceleration: float  # -MAX_ACCELERATION to +MAX_ACCELERATION
    stance: str  # "neutral", "extended", "retracted", "defending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acceleration": self.acceleration,
            "stance": self.stance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        return cls(
            acceleration=data["acceleration"],
            stance=data["stance"]
        )


class ProtocolValidator:
    """Validates actions against protocol rules."""

    def __init__(self, max_acceleration: float, valid_stances: list):
        self.max_acceleration = max_acceleration
        self.valid_stances = set(valid_stances)

    def validate_action(self, action: Action) -> tuple[bool, str]:
        """
        Validate an action against protocol rules.

        Returns: (is_valid, error_message)
        """
        # Check acceleration bounds
        if abs(action.acceleration) > self.max_acceleration:
            return False, f"Acceleration {action.acceleration} exceeds max {self.max_acceleration}"

        # Check stance validity
        if action.stance not in self.valid_stances:
            return False, f"Invalid stance '{action.stance}'. Valid: {self.valid_stances}"

        return True, ""

    def clamp_action(self, action: Action) -> Action:
        """Clamp invalid action to nearest valid action."""
        clamped_accel = max(-self.max_acceleration,
                           min(self.max_acceleration, action.acceleration))

        valid_stance = action.stance if action.stance in self.valid_stances else "neutral"

        return Action(acceleration=clamped_accel, stance=valid_stance)


def generate_snapshot(
    my_fighter,
    opp_fighter,
    tick: int,
    arena_width: float
) -> Dict[str, Any]:
    """
    Generate snapshot for a fighter (no sensor filtering in minimal POC).

    In full implementation, this would apply sensor constraints
    (bucketing, precision, detection ranges, etc.)
    """
    distance = abs(opp_fighter.position - my_fighter.position)

    # Determine relative velocity (negative if approaching)
    if my_fighter.position < opp_fighter.position:
        # I'm on the left
        rel_velocity = opp_fighter.velocity - my_fighter.velocity
    else:
        # I'm on the right
        rel_velocity = my_fighter.velocity - opp_fighter.velocity

    return {
        "tick": tick,
        "you": {
            "position": my_fighter.position,
            "velocity": my_fighter.velocity,
            "hp": my_fighter.hp,
            "max_hp": my_fighter.max_hp,
            "stamina": my_fighter.stamina,
            "max_stamina": my_fighter.max_stamina,
            "stance": my_fighter.stance
        },
        "opponent": {
            "distance": distance,
            "velocity": rel_velocity,
            "hp": opp_fighter.hp,
            "max_hp": opp_fighter.max_hp,
            "stamina": opp_fighter.stamina,
            "max_stamina": opp_fighter.max_stamina,
            "stance_hint": opp_fighter.stance
        },
        "arena": {
            "width": arena_width
        }
    }
