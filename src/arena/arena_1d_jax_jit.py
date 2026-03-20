"""
Atom Combat - JAX-Accelerated Arena Physics Engine (JIT-Optimized)

Phase 3: Fully JIT-compiled JAX physics with integer stances for maximum performance.
Removes Python control flow to enable efficient JIT compilation and vectorization.
"""

import os
import jax

# Force CPU if requested or if GPU fails
if os.environ.get("ATOM_FORCE_CPU", "").lower() in ["1", "true", "yes"]:
    jax.config.update('jax_platform_name', 'cpu')
else:
    # Try GPU first, fallback to CPU if it fails
    try:
        # Test if GPU works
        test = jax.numpy.array([1.0])
        _ = test + 1  # Force computation
    except Exception as e:
        print(f"GPU initialization failed ({e}), falling back to CPU")
        jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import jit, vmap
from typing import List, Tuple, Dict, NamedTuple
import chex

from .world_config import WorldConfig

# Stance constants (for JIT compilation - no Python strings)
STANCE_NEUTRAL = 0
STANCE_EXTENDED = 1
STANCE_DEFENDING = 2  # Moved up from 3

# Stance name mapping (for Python interface)
STANCE_NAMES = ["neutral", "extended", "defending"]  # Removed retracted
STANCE_TO_INT = {name: i for i, name in enumerate(STANCE_NAMES)}


def stance_to_int(stance) -> int:
    """Convert stance string to integer, or return if already int."""
    if isinstance(stance, int):
        return stance
    return STANCE_TO_INT[stance]


def stance_to_str(stance_int: int) -> str:
    """Convert stance integer to string."""
    return STANCE_NAMES[stance_int]


@chex.dataclass
class FighterStateJAX:
    """
    JAX-compatible fighter state (immutable) with integer stance for JIT.

    Uses chex.dataclass for automatic JAX pytree registration.
    All updates return new instances (functional style).

    Note: name field removed - strings can't be JIT-compiled.
    Names are stored separately in Arena1DJAXJit.
    """
    mass: float
    position: float
    velocity: float
    hp: float
    max_hp: float
    stamina: float
    max_stamina: float
    stance: int  # Integer stance for JIT (0=neutral, 1=extended, 2=defending)
    last_hit_tick: int  # Tick of last hit for cooldown tracking

    @classmethod
    def from_fighter_state(cls, fighter):
        """Convert from regular FighterState to JAX version (without name)."""
        return cls(
            mass=float(fighter.mass),
            position=float(fighter.position),
            velocity=float(fighter.velocity),
            hp=float(fighter.hp),
            max_hp=float(fighter.max_hp),
            stamina=float(fighter.stamina),
            max_stamina=float(fighter.max_stamina),
            stance=stance_to_int(fighter.stance),
            last_hit_tick=int(fighter.last_hit_tick)
        )

    def to_dict(self, name: str = "Unknown") -> dict:
        """Convert to dictionary (name provided separately since it's not in JIT state)."""
        return {
            "name": name,
            "mass": float(self.mass),
            "position": float(self.position),
            "velocity": float(self.velocity),
            "hp": float(self.hp),
            "max_hp": float(self.max_hp),
            "stamina": float(self.stamina),
            "max_stamina": float(self.max_stamina),
            "stance": stance_to_str(int(self.stance)),
            "last_hit_tick": int(self.last_hit_tick)
        }


class ArenaStateJAX(NamedTuple):
    """Immutable arena state for JAX transformations."""
    fighter_a: FighterStateJAX
    fighter_b: FighterStateJAX
    tick: int


# Pre-compute stance configs as JAX arrays (indexed by stance int)
def create_stance_arrays(config: WorldConfig):
    """
    Convert stance dict to JAX arrays for JIT compilation.

    Returns:
        (reach_array, defense_array, drain_array)
        Each array indexed by stance int (0=neutral, 1=extended, 2=defending)
    """
    stances_order = ["neutral", "extended", "defending"]  # 3 stances now

    reach = jnp.array([config.stances[s].reach for s in stances_order])
    defense = jnp.array([config.stances[s].defense for s in stances_order])
    drain = jnp.array([config.stances[s].drain for s in stances_order])

    return reach, defense, drain


class Arena1DJAXJit:
    """
    JAX-accelerated 1D physics arena with JIT compilation enabled.

    Phase 3 optimizations:
    - Integer stances (no Python strings in JIT functions)
    - Pre-computed stance config arrays
    - Full JIT compilation of physics step
    - Vectorization-ready (can use vmap)
    """

    def __init__(self, fighter_a, fighter_b, config: WorldConfig, seed=42):
        """
        Initialize arena with two fighters.

        Args:
            fighter_a: First fighter (FighterState or FighterStateJAX)
            fighter_b: Second fighter (FighterState or FighterStateJAX)
            config: WorldConfig instance
            seed: Random seed
        """
        self.config = config
        self.seed = seed

        # Store names separately (can't be JIT-compiled)
        self.name_a = fighter_a.name if hasattr(fighter_a, 'name') else "Fighter A"
        self.name_b = fighter_b.name if hasattr(fighter_b, 'name') else "Fighter B"

        # Pre-compute stance arrays for JIT
        self.stance_reach, self.stance_defense, self.stance_drain = create_stance_arrays(config)

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
                f"Fighter mass {fighter.mass}kg outside legal range "
                f"({self.config.min_mass:.1f}-{self.config.max_mass:.1f}kg)"
            )

    def step(self, action_a: Dict, action_b: Dict) -> List[Dict]:
        """
        Execute one physics tick (wrapper for JIT step).

        Args:
            action_a: {"acceleration": float, "stance": str}
            action_b: {"acceleration": float, "stance": str}

        Returns:
            List of events
        """
        # Convert string stances to integers for JIT
        action_a_int = {
            "acceleration": action_a["acceleration"],
            "stance": stance_to_int(action_a["stance"])
        }
        action_b_int = {
            "acceleration": action_b["acceleration"],
            "stance": stance_to_int(action_b["stance"])
        }

        # Extract config values for JIT (can't pass WorldConfig object)
        dt = self.config.dt
        max_accel = self.config.max_acceleration
        max_vel = self.config.max_velocity
        friction = self.config.friction
        arena_width = self.config.arena_width
        stamina_accel_cost = self.config.stamina_accel_cost
        stamina_base_regen = self.config.stamina_base_regen
        stamina_neutral_bonus = self.config.stamina_neutral_bonus

        # Call JIT-compiled step function
        new_state, events = self._jax_step_jit(
            self.state,
            action_a_int,
            action_b_int,
            dt,
            max_accel,
            max_vel,
            friction,
            arena_width,
            stamina_accel_cost,
            stamina_base_regen,
            stamina_neutral_bonus,
            self.stance_reach,
            self.stance_defense,
            self.stance_drain,
            self.config.hit_cooldown_ticks,
            self.config.hit_impact_threshold,
            self.config.base_collision_damage,  # Using existing param for base damage
            self.config.hit_stamina_cost,
            self.config.block_stamina_cost,
            self.config.hit_recoil_multiplier
        )

        # Generate events from state changes (JIT can't do this efficiently)
        events = []

        # Detect hits by checking HP changes
        old_hp_a = self.state.fighter_a.hp
        old_hp_b = self.state.fighter_b.hp
        new_hp_a = new_state.fighter_a.hp
        new_hp_b = new_state.fighter_b.hp

        # Hit event if damage occurred
        if new_hp_a < old_hp_a:
            events.append({
                "type": "HIT",
                "attacker": self.name_b,
                "defender": self.name_a,
                "damage": float(old_hp_a - new_hp_a),
                "tick": int(self.state.tick)
            })

        if new_hp_b < old_hp_b:
            events.append({
                "type": "HIT",
                "attacker": self.name_a,
                "defender": self.name_b,
                "damage": float(old_hp_b - new_hp_b),
                "tick": int(self.state.tick)
            })

        # Update internal state
        self.state = new_state

        return events

    @staticmethod
    @jit
    def _jax_step_jit(
        state: ArenaStateJAX,
        action_a: Dict,
        action_b: Dict,
        dt: float,
        max_accel: float,
        max_vel: float,
        friction: float,
        arena_width: float,
        stamina_accel_cost: float,
        stamina_base_regen: float,
        stamina_neutral_bonus: float,
        stance_reach: jnp.ndarray,
        stance_defense: jnp.ndarray,
        stance_drain: jnp.ndarray,
        hit_cooldown_ticks: int,
        hit_impact_threshold: float,
        base_damage: float,
        hit_stamina_cost: float,
        block_stamina_cost: float,
        hit_recoil_multiplier: float
    ) -> Tuple[ArenaStateJAX, List[Dict]]:
        """
        Pure functional JIT-compiled step.

        All stance operations use integer comparisons for JIT efficiency.

        Returns:
            (new_state, events)
        """
        fighter_a = state.fighter_a
        fighter_b = state.fighter_b
        tick = state.tick

        # Note: events list can't be JIT-compiled efficiently, so we'll return minimal info
        # For Phase 3, we'll focus on vectorized training where events aren't needed

        # 1. Update velocities
        fighter_a = Arena1DJAXJit._update_velocity_jax(
            fighter_a, action_a, dt, max_accel, max_vel, friction
        )
        fighter_b = Arena1DJAXJit._update_velocity_jax(
            fighter_b, action_b, dt, max_accel, max_vel, friction
        )

        # 2. Update positions
        fighter_a = Arena1DJAXJit._update_position_jax(fighter_a, dt, arena_width)
        fighter_b = Arena1DJAXJit._update_position_jax(fighter_b, dt, arena_width)

        # 3. Update stances (enforce stamina requirement)
        new_stance_a = Arena1DJAXJit._enforce_stamina_stance_jax(
            fighter_a, action_a["stance"]
        )
        new_stance_b = Arena1DJAXJit._enforce_stamina_stance_jax(
            fighter_b, action_b["stance"]
        )
        fighter_a = fighter_a.replace(stance=new_stance_a)
        fighter_b = fighter_b.replace(stance=new_stance_b)

        # 4. Collision detection and discrete hit processing
        footprint_a = Arena1DJAXJit._compute_footprint_jax(fighter_a, stance_reach)
        footprint_b = Arena1DJAXJit._compute_footprint_jax(fighter_b, stance_reach)

        collision = Arena1DJAXJit._check_collision_jax(footprint_a, footprint_b)

        # Process hit mechanics (discrete, cooldown-based)
        fighter_a, fighter_b, damage_to_a, damage_to_b = Arena1DJAXJit._process_collision_hit_jax(
            fighter_a,
            fighter_b,
            tick,
            collision,
            stance_defense,
            hit_cooldown_ticks,
            hit_impact_threshold,
            base_damage,
            hit_stamina_cost,
            block_stamina_cost,
            hit_recoil_multiplier
        )

        # 5. Update stamina
        fighter_a = Arena1DJAXJit._update_stamina_jax(
            fighter_a, action_a, stance_drain, dt, stamina_accel_cost,
            stamina_base_regen, stamina_neutral_bonus
        )
        fighter_b = Arena1DJAXJit._update_stamina_jax(
            fighter_b, action_b, stance_drain, dt, stamina_accel_cost,
            stamina_base_regen, stamina_neutral_bonus
        )

        # Create new state
        new_state = ArenaStateJAX(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            tick=tick + 1
        )

        # Events not supported in JIT (requires Python control flow)
        # For training, we don't need events - just state updates
        events = []

        return new_state, events

    @staticmethod
    def _update_velocity_jax(
        fighter: FighterStateJAX,
        action: Dict,
        dt: float,
        max_accel: float,
        max_vel: float,
        friction: float
    ) -> FighterStateJAX:
        """Update velocity with acceleration and friction (pure function)."""
        # Clamp acceleration input
        accel_input = jnp.clip(action["acceleration"], -max_accel, max_accel)

        # Mass affects acceleration: for same force, lighter mass accelerates more
        # F = m * a, so actual_accel = accel_input * (70.0 / mass)
        # This makes 70kg the baseline - lighter fighters accelerate faster, heavier slower
        mass_factor = 70.0 / fighter.mass
        actual_accel = accel_input * mass_factor

        # Apply acceleration
        new_velocity = fighter.velocity + actual_accel * dt

        # Apply friction
        new_velocity = new_velocity * (1.0 - friction * dt)

        # Clamp velocity
        new_velocity = jnp.clip(new_velocity, -max_vel, max_vel)

        return fighter.replace(velocity=new_velocity)

    @staticmethod
    def _update_position_jax(
        fighter: FighterStateJAX,
        dt: float,
        arena_width: float
    ) -> FighterStateJAX:
        """Update position and handle wall collisions (pure function)."""
        new_position = fighter.position + fighter.velocity * dt

        # Wall collision (left wall)
        hit_left = new_position < 0
        new_position = jnp.where(hit_left, 0.0, new_position)
        new_velocity = jnp.where(hit_left, 0.0, fighter.velocity)

        # Wall collision (right wall)
        hit_right = new_position > arena_width
        new_position = jnp.where(hit_right, arena_width, new_position)
        new_velocity = jnp.where(hit_right, 0.0, new_velocity)

        return fighter.replace(
            position=new_position,
            velocity=new_velocity
        )

    @staticmethod
    def _enforce_stamina_stance_jax(
        fighter: FighterStateJAX,
        requested_stance: int
    ) -> int:
        """Enforce stamina requirement for extended stance (JIT-friendly)."""
        # If requesting extended (1) but no stamina, return neutral (0)
        is_extended_no_stamina = (requested_stance == STANCE_EXTENDED) & (fighter.stamina <= 0)
        return jnp.where(is_extended_no_stamina, STANCE_NEUTRAL, requested_stance)

    @staticmethod
    def _compute_footprint_jax(
        fighter: FighterStateJAX,
        stance_reach: jnp.ndarray
    ) -> Tuple[float, float]:
        """Compute fighter footprint based on stance (JIT-friendly)."""
        reach = stance_reach[fighter.stance]

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
    def _calculate_impact_force_jax(
        fighter_a: FighterStateJAX,
        fighter_b: FighterStateJAX
    ) -> float:
        """
        Calculate impact force from collision using physics.
        Force = relative_velocity * reduced_mass

        This determines if a hit is "light" or "heavy" dynamically.
        """
        # Relative velocity (closing speed)
        rel_velocity = jnp.abs(fighter_a.velocity - fighter_b.velocity)

        # Reduced mass (m1*m2)/(m1+m2) - physics of collision
        reduced_mass = (fighter_a.mass * fighter_b.mass) / (fighter_a.mass + fighter_b.mass)

        # Impact force
        impact_force = rel_velocity * reduced_mass

        return impact_force

    @staticmethod
    def _check_hit_valid_jax(
        fighter: FighterStateJAX,
        current_tick: int,
        impact_force: float,
        hit_cooldown: int,
        impact_threshold: float
    ) -> bool:
        """
        Check if hit should register (cooldown + threshold).
        """
        ticks_since_last_hit = current_tick - fighter.last_hit_tick
        cooldown_satisfied = ticks_since_last_hit >= hit_cooldown
        impact_sufficient = impact_force >= impact_threshold

        return jnp.logical_and(cooldown_satisfied, impact_sufficient)

    @staticmethod
    def _calculate_discrete_hit_damage_jax(
        attacker: FighterStateJAX,
        defender: FighterStateJAX,
        impact_force: float,
        stance_defense: jnp.ndarray,
        base_damage: float
    ) -> float:
        """
        Calculate damage from discrete hit event.
        Damage scales with impact force (physics-based).
        """
        # Defense multiplier from stance
        defense_mult = stance_defense[defender.stance]

        # Stamina scaling (attacker)
        stamina_pct = attacker.stamina / attacker.max_stamina
        stamina_mult = 0.5 + 0.5 * stamina_pct  # 50-100% based on stamina

        # Damage = base * (1 + impact_force/10) * stamina * (1/defense)
        # Scale impact force to reasonable damage range
        impact_mult = 1.0 + impact_force / 10.0
        damage = base_damage * impact_mult * stamina_mult / defense_mult

        return damage

    @staticmethod
    def _process_collision_hit_jax(
        fighter_a: FighterStateJAX,
        fighter_b: FighterStateJAX,
        current_tick: int,
        collision: bool,
        stance_defense: jnp.ndarray,
        hit_cooldown: int,
        impact_threshold: float,
        base_damage: float,
        hit_stamina_cost: float,
        block_stamina_cost: float,
        hit_recoil: float
    ) -> Tuple[FighterStateJAX, FighterStateJAX, float, float]:
        """
        Process discrete hit mechanics on collision.

        Returns:
            (updated_fighter_a, updated_fighter_b, damage_to_a, damage_to_b)
        """
        # Calculate impact force (physics-based)
        impact_force = Arena1DJAXJit._calculate_impact_force_jax(fighter_a, fighter_b)

        # Check if A can hit B (cooldown + threshold)
        a_can_hit = jnp.logical_and(
            fighter_a.stance == STANCE_EXTENDED,
            Arena1DJAXJit._check_hit_valid_jax(
                fighter_a, current_tick, impact_force, hit_cooldown, impact_threshold
            )
        )

        # Check if B can hit A
        b_can_hit = jnp.logical_and(
            fighter_b.stance == STANCE_EXTENDED,
            Arena1DJAXJit._check_hit_valid_jax(
                fighter_b, current_tick, impact_force, hit_cooldown, impact_threshold
            )
        )

        # Calculate damage (only if hit valid and collision)
        damage_to_b_raw = jnp.where(
            jnp.logical_and(collision, a_can_hit),
            Arena1DJAXJit._calculate_discrete_hit_damage_jax(
                fighter_a, fighter_b, impact_force, stance_defense, base_damage
            ),
            0.0
        )

        damage_to_a_raw = jnp.where(
            jnp.logical_and(collision, b_can_hit),
            Arena1DJAXJit._calculate_discrete_hit_damage_jax(
                fighter_b, fighter_a, impact_force, stance_defense, base_damage
            ),
            0.0
        )

        # Apply damage
        new_hp_a = jnp.maximum(0.0, fighter_a.hp - damage_to_a_raw)
        new_hp_b = jnp.maximum(0.0, fighter_b.hp - damage_to_b_raw)

        # Stamina costs
        # Attacker loses stamina when landing hit
        a_stamina_cost = jnp.where(damage_to_b_raw > 0, hit_stamina_cost, 0.0)
        # Defender loses stamina when blocking hit (but less than attacker)
        b_is_blocking = (fighter_b.stance == STANCE_DEFENDING) & (damage_to_b_raw > 0)
        b_block_cost = jnp.where(b_is_blocking, block_stamina_cost, 0.0)

        # Similar for B attacking A
        b_stamina_cost = jnp.where(damage_to_a_raw > 0, hit_stamina_cost, 0.0)
        a_is_blocking = (fighter_a.stance == STANCE_DEFENDING) & (damage_to_a_raw > 0)
        a_block_cost = jnp.where(a_is_blocking, block_stamina_cost, 0.0)

        # Total stamina deltas
        a_stamina_delta = -(a_stamina_cost + a_block_cost)
        b_stamina_delta = -(b_stamina_cost + b_block_cost)

        new_stamina_a = jnp.maximum(0.0, fighter_a.stamina + a_stamina_delta)
        new_stamina_b = jnp.maximum(0.0, fighter_b.stamina + b_stamina_delta)

        # Recoil (reduce velocity after hit)
        a_recoil = jnp.where(damage_to_b_raw > 0, hit_recoil, 0.0)
        b_recoil = jnp.where(damage_to_a_raw > 0, hit_recoil, 0.0)

        new_vel_a = fighter_a.velocity * (1.0 - a_recoil)
        new_vel_b = fighter_b.velocity * (1.0 - b_recoil)

        # Update last_hit_tick
        new_last_hit_a = jnp.where(
            jnp.logical_or(damage_to_b_raw > 0, damage_to_a_raw > 0),
            current_tick,
            fighter_a.last_hit_tick
        )
        new_last_hit_b = jnp.where(
            jnp.logical_or(damage_to_a_raw > 0, damage_to_b_raw > 0),
            current_tick,
            fighter_b.last_hit_tick
        )

        # Create updated fighters
        fighter_a = fighter_a.replace(
            hp=new_hp_a,
            stamina=new_stamina_a,
            velocity=new_vel_a,
            last_hit_tick=new_last_hit_a
        )
        fighter_b = fighter_b.replace(
            hp=new_hp_b,
            stamina=new_stamina_b,
            velocity=new_vel_b,
            last_hit_tick=new_last_hit_b
        )

        return fighter_a, fighter_b, damage_to_a_raw, damage_to_b_raw

    @staticmethod
    def _calculate_damage_jax(
        attacker: FighterStateJAX,
        defender: FighterStateJAX,
        stance_defense: jnp.ndarray
    ) -> float:
        """Calculate damage from attacker to defender (JIT-friendly)."""
        # Relative velocity
        relative_velocity = jnp.abs(attacker.velocity - defender.velocity)

        # Mass ratio
        mass_ratio = attacker.mass / defender.mass

        # Defense multiplier (from stance array)
        defense_mult = stance_defense[defender.stance]

        # Stamina scaling (10% to 100%) - more severe penalty for low stamina
        stamina_pct = attacker.stamina / attacker.max_stamina
        stamina_mult = 0.1 + 0.9 * stamina_pct

        # Fixed config values (hardcoded for JIT - WorldConfig not JIT-able)
        base_damage = 5.0  # config.base_collision_damage
        velocity_scale = 2.0  # config.velocity_damage_scale
        mass_scale = 0.5  # config.mass_damage_scale

        # Damage formula
        damage = (
            base_damage
            * (1.0 + relative_velocity * velocity_scale)
            * jnp.power(mass_ratio, mass_scale)
            / defense_mult
            * stamina_mult
        )

        return damage

    @staticmethod
    def _update_stamina_jax(
        fighter: FighterStateJAX,
        action: Dict,
        stance_drain: jnp.ndarray,
        dt: float,
        stamina_accel_cost: float,
        stamina_base_regen: float,
        stamina_neutral_bonus: float
    ) -> FighterStateJAX:
        """Update stamina based on action (pure function, JIT-friendly)."""
        # Acceleration cost (scaled by mass)
        mass_factor = fighter.mass / 70.0
        accel_cost = (
            jnp.abs(action["acceleration"])
            * stamina_accel_cost
            * dt
            * mass_factor
        )

        # Stance drain (from array)
        drain = stance_drain[action["stance"]]

        # Regen (bonus for neutral stance, extra bonus for resting)
        regen = stamina_base_regen
        is_neutral = (action["stance"] == STANCE_NEUTRAL)
        regen = jnp.where(is_neutral, regen * stamina_neutral_bonus, regen)

        # Extra regen when resting (low velocity) - simulates catching breath
        is_resting = jnp.abs(fighter.velocity) < 0.5
        regen = jnp.where(is_resting, regen * 1.5, regen)

        # Apply delta
        delta = -accel_cost - drain + regen
        new_stamina = jnp.clip(
            fighter.stamina + delta,
            0.0,
            fighter.max_stamina
        )

        # If stamina hits zero, reduce velocity (less harsh)
        new_velocity = jnp.where(
            new_stamina == 0.0,
            fighter.velocity * 0.75,  # Reduced from 0.5 to 0.75
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
            return self.name_b
        elif b_hp <= 0:
            return self.name_a
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
