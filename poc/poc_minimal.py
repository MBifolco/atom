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

ARENA_WIDTH = 12.4760  # meters (optimized for spectacle)
FRICTION = 0.3225
MAX_ACCELERATION = 4.3751  # m/s²
MAX_VELOCITY = 2.6696  # m/s
DT = 0.0842  # seconds per tick (~12 Hz)

# Stamina (optimized for drama and exhaustion moments)
STAMINA_ACCEL_COST = 0.2192
STAMINA_BASE_REGEN = 0.0451
STAMINA_NEUTRAL_BONUS = 2.0651

# Damage (optimized for close finishes)
BASE_COLLISION_DAMAGE = 3.1096
VELOCITY_DAMAGE_SCALE = 0.3507
MASS_DAMAGE_SCALE = 0.3530

# Stances: {reach, width, drain, defense_multiplier} (optimized for variety)
STANCES = {
    "neutral": {"reach": 0.2768, "width": 0.4428, "drain": 0.0001, "defense": 1.0612},
    "extended": {"reach": 0.8189, "width": 0.1681, "drain": 0.0324, "defense": 0.8872},
    "retracted": {"reach": 0.1005, "width": 0.1185, "drain": 0.0139, "defense": 1.1542},
    "defending": {"reach": 0.3811, "width": 0.5421, "drain": 0.0611, "defense": 1.6290},
}


# ============================================================================
# FIGHTER STATE
# ============================================================================

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
    MIN_MASS = 40.1071
    MAX_MASS = 90.7961

    HP_MIN = 47.9535
    HP_MAX = 125.4919
    STAMINA_MAX = 12.3595
    STAMINA_MIN = 5.7635

    # HP increases with mass (more mass = more damage absorption)
    hp = HP_MIN + (mass - MIN_MASS) * (HP_MAX - HP_MIN) / (MAX_MASS - MIN_MASS)

    # Stamina decreases with mass (more mass = harder to move)
    stamina = STAMINA_MAX - (mass - MIN_MASS) * (STAMINA_MAX - STAMINA_MIN) / (MAX_MASS - MIN_MASS)

    return {
        "max_hp": round(hp, 1),
        "max_stamina": round(stamina, 1)
    }


class FighterState:
    def __init__(self, name, mass, position):
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
        """Enforce world spec constraints. Mass is the only spec - HP/stamina derived by world."""
        MIN_MASS = 40.1071
        MAX_MASS = 90.7961

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
            "width": ARENA_WIDTH
        }
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def render_arena(fighter_a: FighterState, fighter_b: FighterState, events: List[Dict], tick: int):
    """Render the arena state with visual position indicators."""
    # Arena visualization parameters
    arena_display_width = 50  # characters
    scale = ARENA_WIDTH / arena_display_width

    # Calculate positions on display
    pos_a = int(fighter_a.position / scale)
    pos_b = int(fighter_b.position / scale)

    # Create position bar
    bar = ['-'] * arena_display_width

    # Mark fighter positions with different characters based on stance
    stance_chars = {
        'neutral': '●',
        'extended': '▶',
        'retracted': '◀',
        'defending': '■'
    }

    char_a = stance_chars.get(fighter_a.stance, '●')
    char_b = stance_chars.get(fighter_b.stance, '●')

    # Place fighters (handle overlap)
    # Clamp positions to valid range
    pos_a = min(pos_a, arena_display_width - 1)
    pos_b = min(pos_b, arena_display_width - 1)

    if pos_a == pos_b:
        bar[pos_a] = '⚔'  # Both at same position
    else:
        bar[pos_a] = char_a
        bar[pos_b] = char_b

    arena_bar = ''.join(bar)

    # Check if collision occurred
    collision = any(e['type'] == 'COLLISION' for e in events)

    # Print header
    print(f"\n[Tick {tick:3d} | {tick*DT:5.2f}s]")
    print("┌" + "─" * 52 + "┐")

    # Print arena
    if collision:
        print(f"│ {'💥 COLLISION!':^50s} │")
    else:
        print(f"│ {' ':50s} │")

    print(f"│ |{arena_bar}| │")
    print("└" + "─" * 52 + "┘")

    # Print fighter stats
    print(f"\n{fighter_a.name:^26s} │ {fighter_b.name:^26s}")
    print("─" * 26 + "┼" + "─" * 26)

    # HP bars
    hp_a_pct = fighter_a.hp / fighter_a.max_hp
    hp_b_pct = fighter_b.hp / fighter_b.max_hp
    hp_bar_a = _make_bar(hp_a_pct, 20, '█', '░')
    hp_bar_b = _make_bar(hp_b_pct, 20, '█', '░')
    print(f"HP  {hp_bar_a} {fighter_a.hp:5.1f} │ HP  {hp_bar_b} {fighter_b.hp:5.1f}")

    # Stamina bars
    stam_a_pct = fighter_a.stamina / fighter_a.max_stamina
    stam_b_pct = fighter_b.stamina / fighter_b.max_stamina
    stam_bar_a = _make_bar(stam_a_pct, 20, '▓', '░')
    stam_bar_b = _make_bar(stam_b_pct, 20, '▓', '░')
    print(f"STA {stam_bar_a} {fighter_a.stamina:5.1f} │ STA {stam_bar_b} {fighter_b.stamina:5.1f}")

    # Stats
    print(f"Vel {fighter_a.velocity:+5.2f} m/s          │ Vel {fighter_b.velocity:+5.2f} m/s")
    print(f"Pos {fighter_a.position:5.2f}m             │ Pos {fighter_b.position:5.2f}m")
    print(f"Mass {fighter_a.mass:.0f}kg [{fighter_a.stance:9s}] │ Mass {fighter_b.mass:.0f}kg [{fighter_b.stance:9s}]")

    # Event details
    if collision:
        for event in events:
            if event['type'] == 'COLLISION':
                print(f"\n💥 Impact! Dmg: {fighter_a.name} -{event['damage_to_a']:.1f} HP  |  {fighter_b.name} -{event['damage_to_b']:.1f} HP")
                print(f"   Relative velocity: {event['relative_velocity']:.2f} m/s")

    distance = abs(fighter_a.position - fighter_b.position)
    print(f"\nDistance: {distance:.2f}m")


def _make_bar(percentage: float, width: int, fill_char: str, empty_char: str) -> str:
    """Create a progress bar string."""
    filled = int(percentage * width)
    empty = width - filled
    return fill_char * filled + empty_char * empty


# ============================================================================
# MATCH ORCHESTRATOR
# ============================================================================

def run_match(fighter_a_spec, fighter_b_spec, max_ticks=600, verbose=True, display_frequency=15):
    """Run a complete match between two fighters."""

    # Initialize fighters
    fighter_a = FighterState(
        name=fighter_a_spec["name"],
        mass=fighter_a_spec["mass"],
        position=2.0
    )

    fighter_b = FighterState(
        name=fighter_b_spec["name"],
        mass=fighter_b_spec["mass"],
        position=8.0
    )

    # Initialize arena
    arena = Arena1D(fighter_a, fighter_b)

    # Get fighter decision functions
    decide_a = fighter_a_spec["decide_fn"]
    decide_b = fighter_b_spec["decide_fn"]

    if verbose:
        print("\n" + "═" * 60)
        print(f"{'ATOM COMBAT - 1D ARENA':^60s}")
        print("═" * 60)
        print(f"{fighter_a.name} ({fighter_a.mass:.0f}kg) vs {fighter_b.name} ({fighter_b.mass:.0f}kg)".center(60))
        print("═" * 60)

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

        # Render arena at specified frequency or on collision
        has_collision = any(e['type'] == 'COLLISION' for e in events)
        if verbose and (tick % display_frequency == 0 or has_collision):
            render_arena(fighter_a, fighter_b, events, tick)

        # Check win conditions
        if not fighter_a.is_alive():
            if verbose:
                print("\n" + "═" * 60)
                print(f"{'💀 KNOCKOUT!':^60s}")
                print("═" * 60)
                print(f"{fighter_b.name} wins at {tick*DT:.2f}s (tick {tick})".center(60))
                print(f"Final: {fighter_a.name} {fighter_a.hp:.1f} HP  |  {fighter_b.name} {fighter_b.hp:.1f} HP".center(60))
                print("═" * 60)
            return {"winner": fighter_b.name, "reason": "KO", "tick": tick}

        if not fighter_b.is_alive():
            if verbose:
                print("\n" + "═" * 60)
                print(f"{'💀 KNOCKOUT!':^60s}")
                print("═" * 60)
                print(f"{fighter_a.name} wins at {tick*DT:.2f}s (tick {tick})".center(60))
                print(f"Final: {fighter_a.name} {fighter_a.hp:.1f} HP  |  {fighter_b.name} {fighter_b.hp:.1f} HP".center(60))
                print("═" * 60)
            return {"winner": fighter_a.name, "reason": "KO", "tick": tick}

    # Timeout - determine winner by HP
    if verbose:
        print("\n" + "═" * 60)
        print(f"{'⏱️  TIME EXPIRED':^60s}")
        print("═" * 60)
        print(f"Match duration: {max_ticks*DT:.2f}s ({max_ticks} ticks)".center(60))
        print(f"Final: {fighter_a.name} {fighter_a.hp:.1f} HP  |  {fighter_b.name} {fighter_b.hp:.1f} HP".center(60))

    if fighter_a.hp > fighter_b.hp:
        if verbose:
            print(f"{fighter_a.name} wins by HP".center(60))
            print("═" * 60)
        return {"winner": fighter_a.name, "reason": "timeout", "tick": max_ticks}
    elif fighter_b.hp > fighter_a.hp:
        if verbose:
            print(f"{fighter_b.name} wins by HP".center(60))
            print("═" * 60)
        return {"winner": fighter_b.name, "reason": "timeout", "tick": max_ticks}
    else:
        if verbose:
            print(f"{'DRAW':^60s}")
            print("═" * 60)
        return {"winner": "draw", "reason": "timeout", "tick": max_ticks}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Define fighter specs (only mass - world calculates HP/stamina)
    aggressive_spec = {
        "name": "AggressiveBot",
        "mass": 60.0,  # Light: 73.3 HP, 10.7 stamina
        "decide_fn": aggressive_fighter
    }

    defensive_spec = {
        "name": "DefensiveBot",
        "mass": 80.0,  # Heavy: 86.7 HP, 9.3 stamina
        "decide_fn": defensive_fighter
    }

    # Run match
    result = run_match(aggressive_spec, defensive_spec, max_ticks=300, verbose=True)

    print()
    print(f"Match result: {result}")
