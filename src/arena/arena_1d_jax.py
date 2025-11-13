"""
Atom Combat - JAX-Accelerated Arena Physics Engine

1D physics simulation with JAX for GPU acceleration and JIT compilation.
Functional, immutable implementation compatible with JAX transformations.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import List, Tuple, Dict, NamedTuple
import chex

from .world_config import WorldConfig


@chex.dataclass
class FighterStateJAX:
    """
    JAX-compatible fighter state (immutable).

    Uses chex.dataclass for automatic JAX pytree registration.
    All updates return new instances (functional style).
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

    @classmethod
    def from_fighter_state(cls, fighter):
        """Convert from regular FighterState to JAX version."""
        return cls(
            name=fighter.name,
            mass=float(fighter.mass),
            position=float(fighter.position),
            velocity=float(fighter.velocity),
            hp=float(fighter.hp),
            max_hp=float(fighter.max_hp),
            stamina=float(fighter.stamina),
            max_stamina=float(fighter.max_stamina),
            stance=fighter.stance
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mass": float(self.mass),
            "position": float(self.position),
            "velocity": float(self.velocity),
            "hp": float(self.hp),
            "max_hp": float(self.max_hp),
            "stamina": float(self.stamina),
            "max_stamina": float(self.max_stamina),
            "stance": self.stance
        }


class ArenaStateJAX(NamedTuple):
    """Immutable arena state for JAX transformations."""
    fighter_a: FighterStateJAX
    fighter_b: FighterStateJAX
    tick: int


class Arena1DJAX:
    """
    JAX-accelerated 1D physics arena.

    Uses functional programming style for JAX compatibility:
    - All operations are pure functions
    - No mutations - returns new states
    - JIT-compiled for performance
    """

    def __init__(self, fighter_a, fighter_b, config: WorldConfig, seed=42):
        """
        Initialize arena with two fighters.

        Args:
            fighter_a: First fighter (FighterState or FighterStateJAX)
            fighter_b: Second fighter (FighterState or FighterStateJAX)
            config: WorldConfig instance
            seed: Random seed (not used in deterministic JAX version)
        """
        self.config = config
        self.seed = seed

        # Convert to JAX fighters if needed
        if not isinstance(fighter_a, FighterStateJAX):
            fighter_a = FighterStateJAX.from_fighter_state(fighter_a)
        if not isinstance(fighter_b, FighterStateJAX):
            fighter_b = FighterStateJAX.from_fighter_state(fighter_b)

        # Validate fighters
        self._validate_fighter(fighter_a)
        self._validate_fighter(fighter_b)

        # Initialize state
        self.state = ArenaStateJAX(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            tick=0
        )

    def _validate_fighter(self, fighter: FighterStateJAX):
        """Validate fighter constraints."""
        if not (self.config.min_mass <= fighter.mass <= self.config.max_mass):
            raise ValueError(
                f"Fighter '{fighter.name}' mass {fighter.mass}kg outside legal range "
                f"({self.config.min_mass:.1f}-{self.config.max_mass:.1f}kg)"
            )

    def step(self, action_a: Dict, action_b: Dict) -> List[Dict]:
        """
        Execute one physics tick (wrapper for JAX step).

        Returns:
            List of events
        """
        # Call JIT-compiled step function
        new_state, events = self._jax_step(
            self.state,
            action_a,
            action_b,
            self.config
        )

        # Update internal state
        self.state = new_state

        return events

    @staticmethod
    def _jax_step(
        state: ArenaStateJAX,
        action_a: Dict,
        action_b: Dict,
        config: WorldConfig
    ) -> Tuple[ArenaStateJAX, List[Dict]]:
        """
        Pure functional step (not JIT-compiled yet - will add after validation).

        Returns:
            (new_state, events)
        """
        fighter_a = state.fighter_a
        fighter_b = state.fighter_b
        tick = state.tick

        events = []

        # 1. Update velocities
        fighter_a = Arena1DJAX._update_velocity_jax(fighter_a, action_a, config)
        fighter_b = Arena1DJAX._update_velocity_jax(fighter_b, action_b, config)

        # 2. Update positions
        fighter_a = Arena1DJAX._update_position_jax(fighter_a, config)
        fighter_b = Arena1DJAX._update_position_jax(fighter_b, config)

        # 3. Update stances (enforce stamina requirement)
        new_stance_a = Arena1DJAX._enforce_stamina_stance_jax(
            fighter_a, action_a["stance"]
        )
        new_stance_b = Arena1DJAX._enforce_stamina_stance_jax(
            fighter_b, action_b["stance"]
        )
        fighter_a = fighter_a.replace(stance=new_stance_a)
        fighter_b = fighter_b.replace(stance=new_stance_b)

        # 4. Collision detection and damage
        footprint_a = Arena1DJAX._compute_footprint_jax(fighter_a, config)
        footprint_b = Arena1DJAX._compute_footprint_jax(fighter_b, config)

        collision = Arena1DJAX._check_collision_jax(footprint_a, footprint_b)

        # Calculate damage if collision
        if collision:
            damage_to_a, damage_to_b = Arena1DJAX._calculate_collision_damage_jax(
                fighter_a, fighter_b, config
            )

            # Apply damage
            new_hp_a = float(jnp.maximum(0.0, fighter_a.hp - damage_to_a))
            new_hp_b = float(jnp.maximum(0.0, fighter_b.hp - damage_to_b))
            fighter_a = fighter_a.replace(hp=new_hp_a)
            fighter_b = fighter_b.replace(hp=new_hp_b)

            # Record collision event
            events.append({
                "type": "COLLISION",
                "tick": int(tick),
                "damage_to_a": round(float(damage_to_a), 1),
                "damage_to_b": round(float(damage_to_b), 1),
                "relative_velocity": float(jnp.abs(fighter_a.velocity - fighter_b.velocity))
            })

        # 5. Update stamina
        fighter_a = Arena1DJAX._update_stamina_jax(fighter_a, action_a, config)
        fighter_b = Arena1DJAX._update_stamina_jax(fighter_b, action_b, config)

        # Create new state
        new_state = ArenaStateJAX(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            tick=int(tick) + 1
        )

        return new_state, events

    @staticmethod
    def _update_velocity_jax(
        fighter: FighterStateJAX,
        action: Dict,
        config: WorldConfig
    ) -> FighterStateJAX:
        """Update velocity with acceleration and friction (pure function)."""
        # Clamp acceleration
        accel = jnp.clip(
            action["acceleration"],
            -config.max_acceleration,
            config.max_acceleration
        )

        # Apply acceleration
        new_velocity = fighter.velocity + accel * config.dt

        # Apply friction
        new_velocity = new_velocity * (1.0 - config.friction * config.dt)

        # Clamp velocity
        new_velocity = jnp.clip(
            new_velocity,
            -config.max_velocity,
            config.max_velocity
        )

        return fighter.replace(velocity=new_velocity)

    @staticmethod
    def _update_position_jax(
        fighter: FighterStateJAX,
        config: WorldConfig
    ) -> FighterStateJAX:
        """Update position and handle wall collisions (pure function)."""
        new_position = fighter.position + fighter.velocity * config.dt

        # Wall collision (left wall)
        hit_left = new_position < 0
        new_position = jnp.where(hit_left, 0.0, new_position)
        new_velocity = jnp.where(hit_left, 0.0, fighter.velocity)

        # Wall collision (right wall)
        hit_right = new_position > config.arena_width
        new_position = jnp.where(hit_right, config.arena_width, new_position)
        new_velocity = jnp.where(hit_right, 0.0, new_velocity)

        return fighter.replace(
            position=new_position,
            velocity=new_velocity
        )

    @staticmethod
    def _enforce_stamina_stance_jax(
        fighter: FighterStateJAX,
        requested_stance: str
    ) -> str:
        """Enforce stamina requirement for extended stance."""
        # If requesting extended but no stamina, return neutral
        if requested_stance == "extended" and fighter.stamina <= 0:
            return "neutral"
        return requested_stance

    @staticmethod
    def _compute_footprint_jax(
        fighter: FighterStateJAX,
        config: WorldConfig
    ) -> Tuple[float, float]:
        """Compute fighter footprint based on stance."""
        stance_config = config.stances[fighter.stance]
        reach = stance_config.reach

        left_edge = fighter.position - reach
        right_edge = fighter.position + reach

        return (left_edge, right_edge)

    @staticmethod
    def _check_collision_jax(
        footprint_a: Tuple[float, float],
        footprint_b: Tuple[float, float]
    ) -> bool:
        """Check if footprints overlap."""
        a_left, a_right = footprint_a
        b_left, b_right = footprint_b

        # Interval overlap: NOT (a entirely left of b OR b entirely left of a)
        return jnp.logical_not(
            jnp.logical_or(a_right < b_left, b_right < a_left)
        )

    @staticmethod
    def _calculate_collision_damage_jax(
        fighter_a: FighterStateJAX,
        fighter_b: FighterStateJAX,
        config: WorldConfig
    ) -> Tuple[float, float]:
        """
        Calculate damage for both fighters in collision.

        Returns:
            (damage_to_a, damage_to_b)
        """
        # Damage to A (from B attacking)
        if fighter_b.stance == "extended":
            damage_to_a = Arena1DJAX._calculate_damage_jax(fighter_b, fighter_a, config)
        else:
            damage_to_a = 0.0

        # Damage to B (from A attacking)
        if fighter_a.stance == "extended":
            damage_to_b = Arena1DJAX._calculate_damage_jax(fighter_a, fighter_b, config)
        else:
            damage_to_b = 0.0

        return damage_to_a, damage_to_b

    @staticmethod
    def _calculate_damage_jax(
        attacker: FighterStateJAX,
        defender: FighterStateJAX,
        config: WorldConfig
    ) -> float:
        """Calculate damage from attacker to defender."""
        # Relative velocity
        relative_velocity = jnp.abs(attacker.velocity - defender.velocity)

        # Mass ratio
        mass_ratio = attacker.mass / defender.mass

        # Defense multiplier
        defense_mult = config.stances[defender.stance].defense

        # Stamina scaling (25% to 100%)
        stamina_pct = attacker.stamina / attacker.max_stamina
        stamina_mult = 0.25 + 0.75 * stamina_pct

        # Damage formula
        damage = (
            config.base_collision_damage
            * (1.0 + relative_velocity * config.velocity_damage_scale)
            * jnp.power(mass_ratio, config.mass_damage_scale)
            / defense_mult
            * stamina_mult
        )

        return damage

    @staticmethod
    def _update_stamina_jax(
        fighter: FighterStateJAX,
        action: Dict,
        config: WorldConfig
    ) -> FighterStateJAX:
        """Update stamina based on action (pure function)."""
        # Acceleration cost (scaled by mass)
        mass_factor = fighter.mass / 70.0
        accel_cost = (
            jnp.abs(action["acceleration"])
            * config.stamina_accel_cost
            * config.dt
            * mass_factor
        )

        # Stance drain
        stance_drain = config.stances[action["stance"]].drain

        # Regen (bonus for neutral stance)
        regen = config.stamina_base_regen
        is_neutral = action["stance"] == "neutral"
        regen = jnp.where(is_neutral, regen * config.stamina_neutral_bonus, regen)

        # Apply delta
        delta = -accel_cost - stance_drain + regen
        new_stamina = jnp.clip(
            fighter.stamina + delta,
            0.0,
            fighter.max_stamina
        )

        # If stamina hits zero, reduce velocity
        new_velocity = jnp.where(
            new_stamina == 0.0,
            fighter.velocity * 0.5,
            fighter.velocity
        )

        return fighter.replace(
            stamina=new_stamina,
            velocity=new_velocity
        )

    def is_finished(self) -> bool:
        """Check if match is over."""
        return (self.state.fighter_a.hp <= 0 or
                self.state.fighter_b.hp <= 0)

    def get_winner(self) -> str:
        """Get winner name."""
        a_hp = self.state.fighter_a.hp
        b_hp = self.state.fighter_b.hp

        if a_hp <= 0 and b_hp <= 0:
            return "draw"
        elif a_hp <= 0:
            return self.state.fighter_b.name
        elif b_hp <= 0:
            return self.state.fighter_a.name
        else:
            return "ongoing"

    # Properties for compatibility with existing code
    @property
    def fighter_a(self):
        """Get fighter A state."""
        return self.state.fighter_a

    @property
    def fighter_b(self):
        """Get fighter B state."""
        return self.state.fighter_b

    @property
    def tick(self):
        """Get current tick."""
        return self.state.tick
