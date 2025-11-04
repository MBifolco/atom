"""
Atom Combat - Arena Physics Engine

1D physics simulation with collision detection and damage calculation.
Config-driven implementation.
"""

import random
from typing import List, Tuple, Dict

from .fighter import FighterState
from .world_config import WorldConfig


class Arena1D:
    """1D physics arena for combat simulation."""

    def __init__(self, fighter_a: FighterState, fighter_b: FighterState, config: WorldConfig, seed=42):
        """
        Initialize arena with two fighters and world configuration.

        Args:
            fighter_a: First fighter
            fighter_b: Second fighter
            config: WorldConfig instance with all physics parameters
            seed: Random seed for reproducibility
        """
        self.config = config

        # Validate fighter constraints
        self._validate_fighter(fighter_a)
        self._validate_fighter(fighter_b)

        self.fighter_a = fighter_a
        self.fighter_b = fighter_b
        self.tick = 0
        random.seed(seed)

    def _validate_fighter(self, fighter: FighterState):
        """Enforce world spec constraints. Mass is the only spec - HP/stamina derived by world."""
        if not (self.config.min_mass <= fighter.mass <= self.config.max_mass):
            raise ValueError(
                f"Fighter '{fighter.name}' mass {fighter.mass}kg outside legal range "
                f"({self.config.min_mass:.1f}-{self.config.max_mass:.1f}kg)"
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

        # 3. Update stances (but enforce stamina requirement)
        self.fighter_a.stance = self._enforce_stamina_stance(self.fighter_a, action_a["stance"])
        self.fighter_b.stance = self._enforce_stamina_stance(self.fighter_b, action_b["stance"])

        # 4. Compute footprints and detect collision
        footprint_a = self._compute_footprint(self.fighter_a)
        footprint_b = self._compute_footprint(self.fighter_b)

        if self._check_collision(footprint_a, footprint_b):
            # Only attackers (extended stance) deal damage
            # Neutral/defensive stances deal NO damage
            damage_to_a = 0.0
            damage_to_b = 0.0

            # Fighter B attacking Fighter A
            if self.fighter_b.stance == "extended":
                damage_to_a = self._calculate_damage(self.fighter_b, self.fighter_a)

            # Fighter A attacking Fighter B
            if self.fighter_a.stance == "extended":
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
        accel = max(-self.config.max_acceleration, min(self.config.max_acceleration, action["acceleration"]))

        # Apply acceleration
        new_velocity = fighter.velocity + accel * self.config.dt

        # Apply friction
        new_velocity *= (1 - self.config.friction * self.config.dt)

        # Clamp velocity
        fighter.velocity = max(-self.config.max_velocity, min(self.config.max_velocity, new_velocity))

    def _update_position(self, fighter: FighterState):
        """Update position and handle wall collisions."""
        fighter.position += fighter.velocity * self.config.dt

        # Wall collision
        if fighter.position < 0:
            fighter.position = 0
            fighter.velocity = 0
        elif fighter.position > self.config.arena_width:
            fighter.position = self.config.arena_width
            fighter.velocity = 0

    def _enforce_stamina_stance(self, fighter: FighterState, requested_stance: str) -> str:
        """
        Enforce stamina requirement for offensive stances.

        At 0 stamina, fighters CANNOT attack (extended stance).
        Defensive stances (defending, retracted) and neutral are still allowed.
        This creates strategic depth - manage stamina or lose attacking ability.
        """
        # Extended (attacking) requires stamina
        if requested_stance == "extended" and fighter.stamina <= 0:
            return "neutral"  # Can't attack without stamina, default to neutral

        return requested_stance

    def _compute_footprint(self, fighter: FighterState) -> Tuple[float, float]:
        """Compute fighter's footprint (left_edge, right_edge) based on stance."""
        stance = self.config.stances[fighter.stance]
        reach = stance.reach

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
        """
        Calculate damage dealt by attacker to defender.

        NOTE: This should only be called when attacker is in extended stance.
        The collision logic already filters out non-attackers.
        """
        # Relative velocity (how hard they're hitting each other)
        relative_velocity = abs(attacker.velocity - defender.velocity)

        # Mass ratio (heavier hits harder)
        mass_ratio = attacker.mass / defender.mass

        # Defense multiplier from stance
        defense_mult = self.config.stances[defender.stance].defense

        # Stamina scaling - exhausted fighters deal less damage
        # 100% stamina = 100% damage, 0% stamina = 25% damage
        stamina_pct = attacker.stamina / attacker.max_stamina
        stamina_mult = 0.25 + (0.75 * stamina_pct)  # Range: 0.25 to 1.0

        # Damage formula
        damage = self.config.base_collision_damage \
                 * (1 + relative_velocity * self.config.velocity_damage_scale) \
                 * (mass_ratio ** self.config.mass_damage_scale) \
                 / defense_mult \
                 * stamina_mult

        return damage

    def _update_stamina(self, fighter: FighterState, action: Dict):
        """Update stamina based on acceleration, stance, mass, and regen."""
        # Cost of acceleration (scaled by mass - heavier = more expensive)
        # 70kg is baseline (1.0x cost), lighter is cheaper, heavier is more expensive
        mass_factor = fighter.mass / 70.0
        accel_cost = abs(action["acceleration"]) * self.config.stamina_accel_cost * self.config.dt * mass_factor

        # Stance drain
        stance_drain = self.config.stances[action["stance"]].drain

        # Regen (bonus if neutral)
        regen = self.config.stamina_base_regen
        if action["stance"] == "neutral":
            regen *= self.config.stamina_neutral_bonus

        # Apply delta
        delta = -accel_cost - stance_drain + regen
        fighter.stamina = max(0, min(fighter.max_stamina, fighter.stamina + delta))

        # If stamina hits zero, reduce velocity (stance is enforced before being set)
        if fighter.stamina == 0:
            fighter.velocity *= 0.5

    def is_finished(self) -> bool:
        """Check if match is over (either fighter at 0 HP)."""
        return self.fighter_a.hp <= 0 or self.fighter_b.hp <= 0

    def get_winner(self) -> str:
        """Get winner name (or 'draw' if both at 0 HP)."""
        if self.fighter_a.hp <= 0 and self.fighter_b.hp <= 0:
            return "draw"
        elif self.fighter_a.hp <= 0:
            return self.fighter_b.name
        elif self.fighter_b.hp <= 0:
            return self.fighter_a.name
        else:
            return "ongoing"
