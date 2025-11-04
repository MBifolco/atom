"""
Atom Combat - Arena Physics Engine

1D physics simulation with collision detection and damage calculation.
"""

import random
from typing import Dict, List, Tuple

from fighter import FighterState
from world_constants import (
    ARENA_WIDTH, FRICTION, MAX_ACCELERATION, MAX_VELOCITY, DT,
    STAMINA_ACCEL_COST, STAMINA_BASE_REGEN, STAMINA_NEUTRAL_BONUS,
    BASE_COLLISION_DAMAGE, VELOCITY_DAMAGE_SCALE, MASS_DAMAGE_SCALE,
    MIN_MASS, MAX_MASS, STANCES
)


class Arena1D:
    """1D physics arena for combat simulation."""

    def __init__(self, fighter_a: FighterState, fighter_b: FighterState, seed=42):
        # Validate fighter constraints
        self._validate_fighter(fighter_a)
        self._validate_fighter(fighter_b)

        self.fighter_a = fighter_a
        self.fighter_b = fighter_b
        self.tick = 0
        random.seed(seed)

    def _validate_fighter(self, fighter: FighterState):
        """Enforce world spec constraints. Mass is the only spec - HP/stamina derived by world."""
        if not (MIN_MASS <= fighter.mass <= MAX_MASS):
            raise ValueError(
                f"Fighter '{fighter.name}' mass {fighter.mass}kg outside legal range "
                f"({MIN_MASS:.1f}-{MAX_MASS:.1f}kg)"
            )

        # HP and stamina are world-calculated from mass, so they're always valid

    def step(self, action_a: Dict, action_b: Dict) -> List[Dict]:
        """Execute one physics tick with both fighter actions."""
        events = []

        # 1. Update velocities (apply acceleration + friction)
        self._update_velocity(self.fighter_a, action_a)
        self._update_velocity(self.fighter_b, action_b)

        # 2. Update positions
        self._update_position(self.fighter_a)
        self._update_position(self.fighter_b)

        # 3. Update stances
        self.fighter_a.stance = action_a["stance"]
        self.fighter_b.stance = action_b["stance"]

        # 4. Compute footprints and detect collision
        footprint_a = self._compute_footprint(self.fighter_a)
        footprint_b = self._compute_footprint(self.fighter_b)

        if self._check_collision(footprint_a, footprint_b):
            # Calculate damage for both fighters
            damage_to_a = self._calculate_damage(self.fighter_b, self.fighter_a)
            damage_to_b = self._calculate_damage(self.fighter_a, self.fighter_b)

            self.fighter_a.hp = max(0, self.fighter_a.hp - damage_to_a)
            self.fighter_b.hp = max(0, self.fighter_b.hp - damage_to_b)

            events.append({
                "type": "COLLISION",
                "tick": self.tick,
                "damage_to_a": round(damage_to_a, 1),
                "damage_to_b": round(damage_to_b, 1),
                "relative_velocity": abs(self.fighter_a.velocity - self.fighter_b.velocity)
            })

        # 5. Update stamina
        self._update_stamina(self.fighter_a, action_a)
        self._update_stamina(self.fighter_b, action_b)

        self.tick += 1
        return events

    def _update_velocity(self, fighter: FighterState, action: Dict):
        """Update velocity with acceleration and friction."""
        # Clamp acceleration
        accel = max(-MAX_ACCELERATION, min(MAX_ACCELERATION, action["acceleration"]))

        # Apply acceleration
        new_velocity = fighter.velocity + accel * DT

        # Apply friction
        new_velocity *= (1 - FRICTION * DT)

        # Clamp velocity
        fighter.velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, new_velocity))

    def _update_position(self, fighter: FighterState):
        """Update position and handle wall collisions."""
        fighter.position += fighter.velocity * DT

        # Wall collision
        if fighter.position < 0:
            fighter.position = 0
            fighter.velocity = 0
        elif fighter.position > ARENA_WIDTH:
            fighter.position = ARENA_WIDTH
            fighter.velocity = 0

    def _compute_footprint(self, fighter: FighterState) -> Tuple[float, float]:
        """Compute fighter's footprint (left_edge, right_edge) based on stance."""
        stance = STANCES[fighter.stance]
        reach = stance["reach"]

        left_edge = fighter.position - reach
        right_edge = fighter.position + reach

        return (left_edge, right_edge)

    def _check_collision(self, footprint_a: Tuple, footprint_b: Tuple) -> bool:
        """Check if two footprints overlap."""
        a_left, a_right = footprint_a
        b_left, b_right = footprint_b

        # Interval overlap check
        return not (a_right < b_left or b_right < a_left)

    def _calculate_damage(self, attacker: FighterState, defender: FighterState) -> float:
        """Calculate damage dealt by attacker to defender."""
        # Relative velocity (how hard they're hitting each other)
        relative_velocity = abs(attacker.velocity - defender.velocity)

        # Mass ratio (heavier hits harder)
        mass_ratio = attacker.mass / defender.mass

        # Defense multiplier from stance
        defense_mult = STANCES[defender.stance]["defense"]

        # Damage formula
        damage = BASE_COLLISION_DAMAGE \
                 * (1 + relative_velocity * VELOCITY_DAMAGE_SCALE) \
                 * (mass_ratio ** MASS_DAMAGE_SCALE) \
                 / defense_mult

        return damage

    def _update_stamina(self, fighter: FighterState, action: Dict):
        """Update stamina based on acceleration, stance, mass, and regen."""
        # Cost of acceleration (scaled by mass - heavier = more expensive)
        # 70kg is baseline (1.0x cost), lighter is cheaper, heavier is more expensive
        mass_factor = fighter.mass / 70.0
        accel_cost = abs(action["acceleration"]) * STAMINA_ACCEL_COST * DT * mass_factor

        # Stance drain
        stance_drain = STANCES[action["stance"]]["drain"]

        # Regen (bonus if neutral)
        regen = STAMINA_BASE_REGEN
        if action["stance"] == "neutral":
            regen *= STAMINA_NEUTRAL_BONUS

        # Apply delta
        delta = -accel_cost - stance_drain + regen
        fighter.stamina = max(0, min(fighter.max_stamina, fighter.stamina + delta))

        # If stamina hits zero, force neutral and reduce velocity
        if fighter.stamina == 0:
            fighter.stance = "neutral"
            fighter.velocity *= 0.5
