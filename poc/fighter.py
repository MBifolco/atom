"""
Atom Combat - Fighter

Fighter state representation and stat calculation.
"""

from world_constants import (
    MIN_MASS, MAX_MASS, HP_MIN, HP_MAX, STAMINA_MAX, STAMINA_MIN
)


def calculate_fighter_stats(mass: float) -> dict:
    """
    The world determines HP and stamina from mass.
    Creates natural tradeoffs:
    - Heavy: High HP, low stamina (tank, slow)
    - Light: Low HP, high stamina (fragile, mobile)

    Optimized ranges for spectacle:
    40kg → 48 HP, 12.4 stamina (glass cannon)
    70kg → 88 HP, 8.8 stamina (balanced)
    91kg → 125 HP, 5.8 stamina (tank)
    """
    # HP increases with mass (more mass = more damage absorption)
    hp = HP_MIN + (mass - MIN_MASS) * (HP_MAX - HP_MIN) / (MAX_MASS - MIN_MASS)

    # Stamina decreases with mass (more mass = harder to move)
    stamina = STAMINA_MAX - (mass - MIN_MASS) * (STAMINA_MAX - STAMINA_MIN) / (MAX_MASS - MIN_MASS)

    return {
        "max_hp": round(hp, 1),
        "max_stamina": round(stamina, 1)
    }


class FighterState:
    """Represents the physical state of a fighter during a match."""

    def __init__(self, name: str, mass: float, position: float):
        self.name = name
        self.mass = mass

        # World calculates stats from mass
        stats = calculate_fighter_stats(mass)
        self.max_hp = stats["max_hp"]
        self.max_stamina = stats["max_stamina"]

        # Dynamic state
        self.position = position
        self.velocity = 0.0
        self.hp = self.max_hp
        self.stamina = self.max_stamina
        self.stance = "neutral"

    def is_alive(self) -> bool:
        """Check if fighter is still in the fight."""
        return self.hp > 0

    def __repr__(self):
        return (f"FighterState({self.name}, {self.mass}kg, "
                f"HP:{self.hp:.1f}/{self.max_hp}, "
                f"Stamina:{self.stamina:.1f}/{self.max_stamina})")
