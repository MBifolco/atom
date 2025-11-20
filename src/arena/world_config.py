"""
Atom Combat - World Configuration

All world constants in a config object.
Can be loaded from files or created programmatically.
"""

from dataclasses import dataclass, field
from typing import Dict
import json
import yaml


@dataclass
class StanceConfig:
    """Configuration for a single stance."""
    reach: float
    width: float
    drain: float
    defense: float


@dataclass
class WorldConfig:
    """
    Complete world configuration.
    All physics constants, damage formulas, and stance definitions.
    """
    # Physics
    arena_width: float = 12.4760
    friction: float = 0.3225
    max_acceleration: float = 4.3751
    max_velocity: float = 2.6696
    dt: float = 0.0842  # seconds per tick

    # Stamina economy
    stamina_accel_cost: float = 0.5  # High cost for acceleration
    stamina_base_regen: float = 0.03  # Base regen (doubled from 0.015)
    stamina_neutral_bonus: float = 3.5  # Good bonus for neutral stance

    # Damage
    base_collision_damage: float = 3.1096
    velocity_damage_scale: float = 0.3507
    mass_damage_scale: float = 0.3530

    # Fighter constraints
    min_mass: float = 40.1071
    max_mass: float = 90.7961
    hp_min: float = 47.9535
    hp_max: float = 125.4919
    stamina_max: float = 12.3595
    stamina_min: float = 5.7635

    # Discrete hit system
    hit_cooldown_ticks: int = 5  # Minimum ticks between hits
    hit_impact_threshold: float = 0.5  # Minimum impact force to register hit
    hit_recoil_multiplier: float = 0.3  # Velocity reduction on hit
    hit_stamina_cost: float = 2.0  # Stamina cost when landing hit
    block_stamina_cost: float = 1.0  # Stamina cost when blocking hit

    # Stances (3-stance system for boxing-style combat)
    stances: Dict[str, StanceConfig] = field(default_factory=lambda: {
        "neutral": StanceConfig(reach=0.2768, width=0.4428, drain=0.0, defense=1.0612),
        "extended": StanceConfig(reach=0.8189, width=0.1681, drain=0.08, defense=0.8872),  # Vulnerable when attacking
        "defending": StanceConfig(reach=0.3811, width=0.5421, drain=-0.10, defense=1.50),  # GAINS stamina while defending
    })

    def calculate_fighter_stats(self, mass: float) -> dict:
        """
        Calculate HP and stamina from mass based on world formulas.

        Returns: {"max_hp": float, "max_stamina": float}
        """
        # HP increases with mass
        hp = self.hp_min + (mass - self.min_mass) * (self.hp_max - self.hp_min) / (self.max_mass - self.min_mass)

        # Stamina decreases with mass
        stamina = self.stamina_max - (mass - self.min_mass) * (self.stamina_max - self.stamina_min) / (self.max_mass - self.min_mass)

        return {
            "max_hp": round(hp, 1),
            "max_stamina": round(stamina, 1)
        }

    @classmethod
    def load_from_json(cls, filepath: str) -> 'WorldConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'WorldConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'WorldConfig':
        """Create config from dictionary."""
        # Parse stances if present
        if 'stances' in data and isinstance(data['stances'], dict):
            stances = {}
            for stance_name, stance_data in data['stances'].items():
                stances[stance_name] = StanceConfig(**stance_data)
            data['stances'] = stances

        return cls(**data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'stances':
                result[key] = {
                    name: {
                        'reach': stance.reach,
                        'width': stance.width,
                        'drain': stance.drain,
                        'defense': stance.defense
                    }
                    for name, stance in value.items()
                }
            else:
                result[key] = value
        return result

    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default optimized config (from 150-sample spectacle search)
SPECTACLE_CONFIG = WorldConfig()
