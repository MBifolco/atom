"""
Atom Combat - Fighter State

Fighter state representation with world-calculated stats.
"""

from dataclasses import dataclass


@dataclass
class FighterState:
    """
    Fighter state in the arena.
    Mass is the only fighter specification - all other stats are world-calculated.
    """
    name: str
    mass: float
    position: float
    velocity: float
    hp: float
    max_hp: float
    stamina: float
    max_stamina: float
    stance: str
    last_hit_tick: int = -999  # Tick of last hit landed/taken (for cooldown)

    @classmethod
    def create(cls, name: str, mass: float, position: float, world_config) -> 'FighterState':
        """
        Create a fighter with world-calculated stats.

        Args:
            name: Fighter name
            mass: Fighter mass (kg) - the only spec parameter
            position: Starting position in arena
            world_config: WorldConfig instance to calculate stats from
        """
        stats = world_config.calculate_fighter_stats(mass)

        return cls(
            name=name,
            mass=mass,
            position=position,
            velocity=0.0,
            hp=stats["max_hp"],
            max_hp=stats["max_hp"],
            stamina=stats["max_stamina"],
            max_stamina=stats["max_stamina"],
            stance="neutral",
            last_hit_tick=-999
        )

    def to_dict(self) -> dict:
        """Convert fighter state to dictionary."""
        return {
            "name": self.name,
            "mass": self.mass,
            "position": self.position,
            "velocity": self.velocity,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "stamina": self.stamina,
            "max_stamina": self.max_stamina,
            "stance": self.stance,
            "last_hit_tick": self.last_hit_tick
        }
