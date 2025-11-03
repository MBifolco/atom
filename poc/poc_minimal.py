#!/usr/bin/env python3
"""
Atom Combat - Minimal POC

Single-file proof of concept for 1D physics-based combat.
Two fighters battle using acceleration + stance controls.
"""

import random
from typing import Dict, Tuple, List


# ============================================================================
# WORLD SPEC - Physics Constants
# ============================================================================

ARENA_WIDTH = 10.0  # meters
FRICTION = 0.8
MAX_ACCELERATION = 5.0  # m/s²
MAX_VELOCITY = 3.0  # m/s
DT = 0.067  # seconds per tick (15 Hz)

# Stamina
STAMINA_ACCEL_COST = 0.1
STAMINA_BASE_REGEN = 0.15
STAMINA_NEUTRAL_BONUS = 1.5

# Damage (tuned for ~8-12 hits to KO)
BASE_COLLISION_DAMAGE = 2.5
VELOCITY_DAMAGE_SCALE = 0.8
MASS_DAMAGE_SCALE = 0.5

# Stances: {reach, width, drain, defense_multiplier}
STANCES = {
    "neutral": {"reach": 0.2, "width": 0.3, "drain": 0.0, "defense": 1.0},
    "extended": {"reach": 0.6, "width": 0.2, "drain": 0.05, "defense": 0.8},
    "retracted": {"reach": 0.1, "width": 0.2, "drain": 0.02, "defense": 1.0},
    "defending": {"reach": 0.3, "width": 0.4, "drain": 0.03, "defense": 1.5},
}


# ============================================================================
# FIGHTER STATE
# ============================================================================

class FighterState:
    def __init__(self, name, mass, max_hp, max_stamina, position):
        self.name = name
        self.mass = mass
        self.max_hp = max_hp
        self.max_stamina = max_stamina

        # Dynamic state
        self.position = position
        self.velocity = 0.0
        self.hp = max_hp
        self.stamina = max_stamina
        self.stance = "neutral"

    def is_alive(self):
        return self.hp > 0


# ============================================================================
# ARENA - Physics Engine
# ============================================================================

class Arena1D:
    def __init__(self, fighter_a: FighterState, fighter_b: FighterState, seed=42):
        # Validate fighter constraints (from world_spec.md)
        self._validate_fighter(fighter_a)
        self._validate_fighter(fighter_b)

        self.fighter_a = fighter_a
        self.fighter_b = fighter_b
        self.tick = 0
        random.seed(seed)

    def _validate_fighter(self, fighter: FighterState):
        """Enforce world spec constraints to prevent 'perfect fighter' builds."""
        MIN_MASS = 40.0
        MAX_MASS = 100.0
        MAX_HP = 100.0
        MAX_STAMINA = 10.0

        if not (MIN_MASS <= fighter.mass <= MAX_MASS):
            raise ValueError(
                f"Fighter '{fighter.name}' mass {fighter.mass}kg outside legal range "
                f"({MIN_MASS}-{MAX_MASS}kg)"
            )

        if fighter.max_hp > MAX_HP:
            raise ValueError(
                f"Fighter '{fighter.name}' HP {fighter.max_hp} exceeds maximum {MAX_HP}"
            )

        if fighter.max_stamina > MAX_STAMINA:
            raise ValueError(
                f"Fighter '{fighter.name}' stamina {fighter.max_stamina} exceeds maximum {MAX_STAMINA}"
            )

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
        """Apply acceleration and friction to velocity."""
        accel = max(-MAX_ACCELERATION, min(MAX_ACCELERATION, action["acceleration"]))

        # Apply acceleration
        fighter.velocity += accel * DT

        # Apply friction
        fighter.velocity *= (1 - FRICTION * DT)

        # Clamp to max velocity
        fighter.velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, fighter.velocity))

    def _update_position(self, fighter: FighterState):
        """Update position based on velocity, enforce boundaries."""
        fighter.position += fighter.velocity * DT

        # Enforce walls
        if fighter.position < 0:
            fighter.position = 0
            fighter.velocity = 0
        elif fighter.position > ARENA_WIDTH:
            fighter.position = ARENA_WIDTH
            fighter.velocity = 0

    def _compute_footprint(self, fighter: FighterState) -> Tuple[float, float]:
        """Compute the space occupied by fighter based on stance."""
        stance_config = STANCES[fighter.stance]
        left_edge = fighter.position - stance_config["width"] / 2
        right_edge = fighter.position + stance_config["reach"]
        return (left_edge, right_edge)

    def _check_collision(self, footprint_a: Tuple, footprint_b: Tuple) -> bool:
        """Check if two footprints overlap."""
        a_left, a_right = footprint_a
        b_left, b_right = footprint_b
        return not (a_right < b_left or b_right < a_left)

    def _calculate_damage(self, attacker: FighterState, defender: FighterState) -> float:
        """Calculate damage based on mass, velocity, and stance."""
        relative_velocity = abs(attacker.velocity - defender.velocity)
        mass_ratio = attacker.mass / defender.mass
        defense_mult = STANCES[defender.stance]["defense"]

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


# ============================================================================
# FIGHTERS - Decision Logic
# ============================================================================

def aggressive_fighter(snapshot: Dict) -> Dict:
    """Always rushes forward and extends when close."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    # If close, extend to hit
    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}

    # If have stamina, rush forward
    elif stamina > 2.0:
        return {"acceleration": 5.0, "stance": "neutral"}

    # Low stamina - conserve energy
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def defensive_fighter(snapshot: Dict) -> Dict:
    """Waits for opponent, defends when close."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_hp = snapshot["you"]["hp"]

    # If opponent rushing toward me (negative velocity = approaching)
    if opp_velocity < -0.5 and distance < 2.0:
        # Defend if low HP, counter-extend if healthy
        if my_hp < 50:
            return {"acceleration": 0.0, "stance": "defending"}
        else:
            return {"acceleration": 2.0, "stance": "extended"}

    # If far away, slowly advance
    elif distance > 5.0:
        return {"acceleration": 2.0, "stance": "neutral"}

    # Mid-range - maintain position
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


# ============================================================================
# SNAPSHOT GENERATION
# ============================================================================

def generate_snapshot(my_fighter: FighterState, opp_fighter: FighterState, tick: int) -> Dict:
    """Generate snapshot for a fighter (no sensor filtering in minimal POC)."""
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
            "stamina": my_fighter.stamina,
            "stance": my_fighter.stance
        },
        "opponent": {
            "distance": distance,
            "velocity": rel_velocity,
            "stance_hint": opp_fighter.stance
        },
        "arena": {
            "width": ARENA_WIDTH
        }
    }


# ============================================================================
# MATCH ORCHESTRATOR
# ============================================================================

def run_match(fighter_a_spec, fighter_b_spec, max_ticks=600, verbose=True):
    """Run a complete match between two fighters."""

    # Initialize fighters
    fighter_a = FighterState(
        name=fighter_a_spec["name"],
        mass=fighter_a_spec["mass"],
        max_hp=fighter_a_spec["max_hp"],
        max_stamina=fighter_a_spec["max_stamina"],
        position=2.0
    )

    fighter_b = FighterState(
        name=fighter_b_spec["name"],
        mass=fighter_b_spec["mass"],
        max_hp=fighter_b_spec["max_hp"],
        max_stamina=fighter_b_spec["max_stamina"],
        position=8.0
    )

    # Initialize arena
    arena = Arena1D(fighter_a, fighter_b)

    # Get fighter decision functions
    decide_a = fighter_a_spec["decide_fn"]
    decide_b = fighter_b_spec["decide_fn"]

    if verbose:
        print("=" * 60)
        print(f"  ATOM COMBAT POC - Match Starting")
        print(f"  {fighter_a.name} vs {fighter_b.name}")
        print("=" * 60)
        print()

    # Main match loop
    for tick in range(max_ticks):
        # Generate snapshots
        snapshot_a = generate_snapshot(fighter_a, fighter_b, tick)
        snapshot_b = generate_snapshot(fighter_b, fighter_a, tick)

        # Get actions from fighters
        action_a = decide_a(snapshot_a)
        action_b = decide_b(snapshot_b)

        # Execute physics step
        events = arena.step(action_a, action_b)

        # Print every 15 ticks (~1 second of simulation time)
        if verbose and tick % 15 == 0:
            print(f"[T:{tick:3d} | {tick*DT:5.2f}s]")
            print(f"  {fighter_a.name:15s} pos:{fighter_a.position:5.2f} vel:{fighter_a.velocity:5.2f} hp:{fighter_a.hp:5.1f} stam:{fighter_a.stamina:4.1f} [{fighter_a.stance}]")
            print(f"  {fighter_b.name:15s} pos:{fighter_b.position:5.2f} vel:{fighter_b.velocity:5.2f} hp:{fighter_b.hp:5.1f} stam:{fighter_b.stamina:4.1f} [{fighter_b.stance}]")
            print(f"  Distance: {abs(fighter_a.position - fighter_b.position):.2f}m")
            print()

        # Print events
        if verbose:
            for event in events:
                if event["type"] == "COLLISION":
                    print(f"  💥 COLLISION at tick {event['tick']}!")
                    print(f"     {fighter_a.name} takes {event['damage_to_a']:.1f} damage")
                    print(f"     {fighter_b.name} takes {event['damage_to_b']:.1f} damage")
                    print(f"     Relative velocity: {event['relative_velocity']:.2f} m/s")
                    print()

        # Check win conditions
        if not fighter_a.is_alive():
            if verbose:
                print("=" * 60)
                print(f"  💀 KNOCKOUT! {fighter_b.name} wins at tick {tick}")
                print(f"  Final HP: {fighter_a.name}={fighter_a.hp:.1f}, {fighter_b.name}={fighter_b.hp:.1f}")
                print("=" * 60)
            return {"winner": fighter_b.name, "reason": "KO", "tick": tick}

        if not fighter_b.is_alive():
            if verbose:
                print("=" * 60)
                print(f"  💀 KNOCKOUT! {fighter_a.name} wins at tick {tick}")
                print(f"  Final HP: {fighter_a.name}={fighter_a.hp:.1f}, {fighter_b.name}={fighter_b.hp:.1f}")
                print("=" * 60)
            return {"winner": fighter_a.name, "reason": "KO", "tick": tick}

    # Timeout - determine winner by HP
    if verbose:
        print("=" * 60)
        print(f"  ⏱️  TIMEOUT at tick {max_ticks}")
        print(f"  Final HP: {fighter_a.name}={fighter_a.hp:.1f}, {fighter_b.name}={fighter_b.hp:.1f}")

    if fighter_a.hp > fighter_b.hp:
        if verbose:
            print(f"  Winner: {fighter_a.name} by HP")
            print("=" * 60)
        return {"winner": fighter_a.name, "reason": "timeout", "tick": max_ticks}
    elif fighter_b.hp > fighter_a.hp:
        if verbose:
            print(f"  Winner: {fighter_b.name} by HP")
            print("=" * 60)
        return {"winner": fighter_b.name, "reason": "timeout", "tick": max_ticks}
    else:
        if verbose:
            print(f"  Result: DRAW")
            print("=" * 60)
        return {"winner": "draw", "reason": "timeout", "tick": max_ticks}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Define fighter specs
    aggressive_spec = {
        "name": "AggressiveBot",
        "mass": 60.0,
        "max_hp": 100,
        "max_stamina": 12.0,
        "decide_fn": aggressive_fighter
    }

    defensive_spec = {
        "name": "DefensiveBot",
        "mass": 80.0,
        "max_hp": 100,
        "max_stamina": 10.0,
        "decide_fn": defensive_fighter
    }

    # Run match
    result = run_match(aggressive_spec, defensive_spec, max_ticks=300, verbose=True)

    print()
    print(f"Match result: {result}")
