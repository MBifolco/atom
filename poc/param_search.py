#!/usr/bin/env python3
"""
Atom Combat - Parameter Search Tool

Instead of guessing at world parameters, this tool searches the parameter
space to find configurations that create the "best" fights according to
defined quality metrics.

Usage:
    python3 param_search.py --samples 100 --top 5
"""

import random
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from poc_minimal import (
    run_match, aggressive_fighter, defensive_fighter,
    FighterState, Arena1D, STANCES
)


# ============================================================================
# PARAMETER SPACE DEFINITION
# ============================================================================

PARAM_SPACE = {
    # Physics constants
    "ARENA_WIDTH": (8.0, 15.0),              # Arena size in meters
    "FRICTION": (0.3, 1.2),                  # Velocity decay rate
    "MAX_ACCELERATION": (3.0, 8.0),          # m/s² max acceleration
    "MAX_VELOCITY": (2.0, 5.0),              # m/s max velocity cap
    "DT": (0.04, 0.1),                       # Seconds per tick (25Hz-10Hz)

    # Stamina economy
    "STAMINA_ACCEL_COST": (0.05, 0.30),     # Cost per m/s² acceleration
    "STAMINA_BASE_REGEN": (0.005, 0.05),    # Base regen per tick
    "STAMINA_NEUTRAL_BONUS": (1.5, 8.0),    # Multiplier when in neutral

    # Damage scaling
    "BASE_COLLISION_DAMAGE": (1.0, 6.0),    # Base damage per collision
    "VELOCITY_DAMAGE_SCALE": (0.3, 2.0),    # Velocity contribution
    "MASS_DAMAGE_SCALE": (0.2, 0.9),        # Mass contribution exponent

    # Mass constraints
    "MIN_MASS": (30.0, 50.0),               # Minimum fighter mass
    "MAX_MASS": (90.0, 120.0),              # Maximum fighter mass

    # Fighter stats formulas (HP and stamina ranges)
    "HP_MIN": (40.0, 80.0),                 # Light fighter HP
    "HP_MAX": (80.0, 150.0),                # Heavy fighter HP
    "STAMINA_MAX": (8.0, 16.0),             # Light fighter stamina
    "STAMINA_MIN": (4.0, 12.0),             # Heavy fighter stamina

    # Stance: neutral
    "STANCE_NEUTRAL_REACH": (0.1, 0.4),
    "STANCE_NEUTRAL_WIDTH": (0.2, 0.5),
    "STANCE_NEUTRAL_DRAIN": (0.0, 0.01),
    "STANCE_NEUTRAL_DEFENSE": (0.9, 1.1),

    # Stance: extended
    "STANCE_EXTENDED_REACH": (0.4, 1.0),
    "STANCE_EXTENDED_WIDTH": (0.1, 0.3),
    "STANCE_EXTENDED_DRAIN": (0.02, 0.1),
    "STANCE_EXTENDED_DEFENSE": (0.6, 0.9),

    # Stance: retracted
    "STANCE_RETRACTED_REACH": (0.05, 0.2),
    "STANCE_RETRACTED_WIDTH": (0.1, 0.3),
    "STANCE_RETRACTED_DRAIN": (0.01, 0.05),
    "STANCE_RETRACTED_DEFENSE": (0.9, 1.2),

    # Stance: defending
    "STANCE_DEFENDING_REACH": (0.2, 0.5),
    "STANCE_DEFENDING_WIDTH": (0.3, 0.6),
    "STANCE_DEFENDING_DRAIN": (0.01, 0.08),
    "STANCE_DEFENDING_DEFENSE": (1.2, 2.0),
}


# ============================================================================
# ENHANCED FIGHTER AI (with retreat logic)
# ============================================================================

def tactical_aggressive(snapshot: Dict) -> Dict:
    """Aggressive fighter that retreats when hurt or exhausted."""
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    opp_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

    # Critical retreat: Very low HP or stamina
    if my_hp_pct < 0.25 or my_stamina_pct < 0.2:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Opportunistic: Opponent exhausted - press hard!
    if opp_stamina_pct < 0.3 and my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 5.0, "stance": "neutral"}

    # Tactical retreat: Losing HP battle and need to recover stamina
    if my_hp_pct < opp_hp_pct - 0.2 and my_stamina_pct < 0.5:
        return {"acceleration": -3.0, "stance": "neutral"}

    # Close range: Extend to hit
    if distance < 1.0:
        if my_stamina_pct > 0.3:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            # Low stamina at close range - retreat!
            return {"acceleration": -5.0, "stance": "neutral"}

    # Mid range: Close in aggressively if healthy
    if distance < 3.0:
        if my_hp_pct > 0.5 and my_stamina_pct > 0.4:
            return {"acceleration": 5.0, "stance": "neutral"}
        else:
            return {"acceleration": 2.0, "stance": "neutral"}

    # Long range: Advance cautiously
    return {"acceleration": 3.0, "stance": "neutral"}


def tactical_defensive(snapshot: Dict) -> Dict:
    """Defensive fighter with counter-attacking and retreat."""
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_velocity = snapshot["opponent"]["velocity"]
    opp_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

    # Critical retreat: Very low HP
    if my_hp_pct < 0.2:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Opponent charging in - brace for impact
    if distance < 2.0 and opp_velocity > 1.5:
        return {"acceleration": 0.0, "stance": "defending"}

    # Opportunistic counter: Opponent exhausted at close range
    if distance < 1.5 and opp_stamina_pct < 0.3 and my_stamina_pct > 0.5:
        return {"acceleration": 3.0, "stance": "extended"}

    # Close range but low stamina - retreat to recover
    if distance < 1.5 and my_stamina_pct < 0.3:
        return {"acceleration": -4.0, "stance": "neutral"}

    # Counter-attack opportunity: Opponent close and I'm healthy
    if distance < 1.0 and my_hp_pct > 0.5 and my_stamina_pct > 0.5:
        return {"acceleration": 3.0, "stance": "extended"}

    # Maintain distance if healthy
    if distance > 4.0:
        return {"acceleration": 2.0, "stance": "neutral"}

    # Default: Hold position defensively
    return {"acceleration": 0.0, "stance": "neutral"}


def tactical_balanced(snapshot: Dict) -> Dict:
    """Balanced fighter that adapts to HP/stamina situation."""
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    opp_stamina_pct = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

    # Emergency retreat
    if my_hp_pct < 0.25:
        return {"acceleration": -5.0, "stance": "neutral"}

    # Opportunistic: Both hurt but I have stamina advantage
    if my_hp_pct < 0.4 and opp_hp_pct < 0.4 and my_stamina_pct > opp_stamina_pct + 0.2:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 4.0, "stance": "neutral"}

    # Winning HP battle - press advantage
    if my_hp_pct > opp_hp_pct + 0.2:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 4.0, "stance": "neutral"}

    # Losing HP battle - play defensive
    if my_hp_pct < opp_hp_pct - 0.2:
        if distance < 2.0:
            return {"acceleration": -3.0, "stance": "defending"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Even match - pressure with stamina management
    if my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 3.0, "stance": "neutral"}
    else:
        # Low stamina - conserve
        return {"acceleration": 0.0, "stance": "neutral"}


# ============================================================================
# TEST FIGHTER SUITE
# ============================================================================

@dataclass
class FighterArchetype:
    """Defines a fighter archetype for testing."""
    name: str
    mass: float
    decide_fn: callable
    description: str


TEST_FIGHTERS = [
    FighterArchetype("LightAggro", 40.0, tactical_aggressive, "Glass cannon rushdown"),
    FighterArchetype("LightDefensive", 45.0, tactical_defensive, "Evasive counter-puncher"),
    FighterArchetype("LightBalanced", 50.0, tactical_balanced, "Light all-rounder"),
    FighterArchetype("MidAggro", 70.0, tactical_aggressive, "Balanced aggressive"),
    FighterArchetype("MidDefensive", 70.0, tactical_defensive, "Balanced defensive"),
    FighterArchetype("HeavyAggro", 95.0, tactical_aggressive, "Powerhouse brawler"),
    FighterArchetype("HeavyTank", 100.0, tactical_defensive, "Immovable object"),
]


# ============================================================================
# MATCH QUALITY METRICS
# ============================================================================

def calculate_match_quality(telemetry: Dict) -> Dict[str, float]:
    """
    Score a match on SPECTACLE - how exciting is the fight?

    Focus on drama, momentum swings, close finishes, and variety.
    WHO wins doesn't matter - only QUALITY of the fight matters.

    Returns dict of metric_name -> score (higher is better, 0-1 range)
    """
    scores = {}

    # 1. Match Duration (want 100-400 ticks - not too quick, not endless)
    duration = telemetry['tick']
    if duration < 30:
        scores['duration'] = 0.0  # Instant KO, no drama
    elif duration > 500:
        scores['duration'] = 0.2  # Boring slugfest
    elif 100 <= duration <= 400:
        scores['duration'] = 1.0  # Perfect length
    else:
        if duration < 100:
            scores['duration'] = duration / 100
        else:
            scores['duration'] = max(0.2, 1.0 - (duration - 400) / 200)

    # 2. Close Finish (nail-biters are exciting!)
    winner_hp_pct = telemetry.get('winner_hp_pct', 1.0)
    if winner_hp_pct < 0.2:
        scores['close_finish'] = 1.0  # Photo finish!
    elif winner_hp_pct < 0.4:
        scores['close_finish'] = 0.9  # Close call
    elif winner_hp_pct < 0.6:
        scores['close_finish'] = 0.7  # Competitive
    elif winner_hp_pct < 0.8:
        scores['close_finish'] = 0.4  # Dominant
    else:
        scores['close_finish'] = 0.0  # Boring stomp

    # 3. Stamina Drama (exhaustion moments = tension!)
    stamina_samples = telemetry.get('stamina_samples', [])
    if stamina_samples:
        # Count moments where fighters were gasping (<30% stamina)
        critical_moments = sum(1 for s in stamina_samples if s < 0.3)
        drama_rate = critical_moments / len(stamina_samples)

        # Want 10-30% of fight at critical stamina
        if 0.1 <= drama_rate <= 0.3:
            scores['stamina_drama'] = 1.0
        elif 0.05 <= drama_rate < 0.1:
            scores['stamina_drama'] = 0.7
        elif drama_rate > 0.3:
            scores['stamina_drama'] = 0.5  # Too much exhaustion
        else:
            scores['stamina_drama'] = 0.3  # No drama
    else:
        scores['stamina_drama'] = 0.0

    # 4. Comeback Potential (HP lead changes = exciting!)
    hp_changes = telemetry.get('hp_changes', [])
    if len(hp_changes) >= 5:
        lead_changes = 0
        for i in range(1, len(hp_changes)):
            prev_lead = hp_changes[i-1][0] - hp_changes[i-1][1]
            curr_lead = hp_changes[i][0] - hp_changes[i][1]
            if prev_lead * curr_lead < 0:  # Lead swapped
                lead_changes += 1

        # More lead changes = more exciting
        if lead_changes >= 3:
            scores['comeback_potential'] = 1.0
        elif lead_changes == 2:
            scores['comeback_potential'] = 0.8
        elif lead_changes == 1:
            scores['comeback_potential'] = 0.5
        else:
            scores['comeback_potential'] = 0.2
    else:
        scores['comeback_potential'] = 0.3

    # 5. Positional Exchanges (movement across arena, not wall grinding)
    positions = telemetry.get('position_samples', [])
    if positions and len(positions) >= 10:
        # Count position swaps (fighter A left→right, B right→left)
        swaps = 0
        for i in range(1, len(positions)):
            prev_a, prev_b = positions[i-1]
            curr_a, curr_b = positions[i]
            # Swap = relative position flips
            if (prev_a < prev_b and curr_a > curr_b) or (prev_a > prev_b and curr_a < curr_b):
                swaps += 1

        swap_rate = swaps / len(positions)
        # Want 5-20% of ticks to have position swaps
        if 0.05 <= swap_rate <= 0.2:
            scores['positional_exchange'] = 1.0
        elif swap_rate > 0.2:
            scores['positional_exchange'] = 0.6  # Too chaotic
        else:
            scores['positional_exchange'] = swap_rate / 0.05  # Scale up to 0.05
    else:
        scores['positional_exchange'] = 0.0

    # 6. Pacing Variety (mix of speeds, not monotonic)
    velocity_samples = telemetry.get('velocity_samples', [])
    if velocity_samples and len(velocity_samples) >= 10:
        # Calculate variance in speed (fast → slow → fast = good pacing)
        speeds = [abs(v) for v in velocity_samples]
        avg_speed = sum(speeds) / len(speeds)

        if avg_speed > 0.1:
            variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            std_dev = variance ** 0.5

            # Normalize: want std_dev around 0.5-1.5 m/s
            if 0.5 <= std_dev <= 1.5:
                scores['pacing_variety'] = 1.0
            elif std_dev < 0.5:
                scores['pacing_variety'] = std_dev / 0.5
            else:
                scores['pacing_variety'] = max(0.3, 1.0 - (std_dev - 1.5) / 2.0)
        else:
            scores['pacing_variety'] = 0.0  # No movement
    else:
        scores['pacing_variety'] = 0.5

    # 7. Collision Drama (want impactful exchanges, not grinding)
    collisions = telemetry.get('collision_count', 0)
    avg_collision_damage = telemetry.get('avg_collision_damage', 0)

    # Want 8-25 collisions with meaningful damage
    if 8 <= collisions <= 25 and avg_collision_damage > 3.0:
        scores['collision_drama'] = 1.0
    elif collisions < 8:
        scores['collision_drama'] = collisions / 8
    elif collisions > 25:
        scores['collision_drama'] = 0.4  # Wall grinding
    else:
        scores['collision_drama'] = 0.5

    return scores


def aggregate_quality_score(metric_scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Combine individual metric scores into overall quality score.

    Default weights: all metrics equal
    """
    if weights is None:
        weights = {k: 1.0 for k in metric_scores.keys()}

    total_weight = sum(weights.values())
    weighted_sum = sum(metric_scores[k] * weights.get(k, 1.0) for k in metric_scores)

    return weighted_sum / total_weight


# ============================================================================
# INSTRUMENTED MATCH RUNNER
# ============================================================================

def run_instrumented_match(fighter_a_archetype, fighter_b_archetype, params: Dict) -> Dict:
    """
    Run a match with the given parameters and collect detailed telemetry.
    """
    # Temporarily patch the constants (hacky but works for POC)
    import poc_minimal
    original_values = {}

    # Apply simple module-level constants
    simple_params = ["ARENA_WIDTH", "FRICTION", "MAX_ACCELERATION", "MAX_VELOCITY", "DT",
                     "STAMINA_ACCEL_COST", "STAMINA_BASE_REGEN", "STAMINA_NEUTRAL_BONUS",
                     "BASE_COLLISION_DAMAGE", "VELOCITY_DAMAGE_SCALE", "MASS_DAMAGE_SCALE"]

    for param_name in simple_params:
        if param_name in params:
            original_values[param_name] = getattr(poc_minimal, param_name)
            setattr(poc_minimal, param_name, params[param_name])

    # Rebuild STANCES dict from parameters
    if any(k.startswith("STANCE_") for k in params):
        original_values['STANCES'] = poc_minimal.STANCES.copy()
        poc_minimal.STANCES = {
            "neutral": {
                "reach": params.get("STANCE_NEUTRAL_REACH", 0.2),
                "width": params.get("STANCE_NEUTRAL_WIDTH", 0.3),
                "drain": params.get("STANCE_NEUTRAL_DRAIN", 0.0),
                "defense": params.get("STANCE_NEUTRAL_DEFENSE", 1.0)
            },
            "extended": {
                "reach": params.get("STANCE_EXTENDED_REACH", 0.6),
                "width": params.get("STANCE_EXTENDED_WIDTH", 0.2),
                "drain": params.get("STANCE_EXTENDED_DRAIN", 0.05),
                "defense": params.get("STANCE_EXTENDED_DEFENSE", 0.8)
            },
            "retracted": {
                "reach": params.get("STANCE_RETRACTED_REACH", 0.1),
                "width": params.get("STANCE_RETRACTED_WIDTH", 0.2),
                "drain": params.get("STANCE_RETRACTED_DRAIN", 0.02),
                "defense": params.get("STANCE_RETRACTED_DEFENSE", 1.0)
            },
            "defending": {
                "reach": params.get("STANCE_DEFENDING_REACH", 0.3),
                "width": params.get("STANCE_DEFENDING_WIDTH", 0.4),
                "drain": params.get("STANCE_DEFENDING_DRAIN", 0.03),
                "defense": params.get("STANCE_DEFENDING_DEFENSE", 1.5)
            }
        }

    # Override calculate_fighter_stats if HP/stamina params present
    if any(k in params for k in ["HP_MIN", "HP_MAX", "STAMINA_MAX", "STAMINA_MIN", "MIN_MASS", "MAX_MASS"]):
        original_values['calculate_fighter_stats'] = poc_minimal.calculate_fighter_stats

        min_mass = params.get("MIN_MASS", 40.0)
        max_mass = params.get("MAX_MASS", 100.0)
        hp_min = params.get("HP_MIN", 60.0)
        hp_max = params.get("HP_MAX", 100.0)
        stam_max = params.get("STAMINA_MAX", 12.0)
        stam_min = params.get("STAMINA_MIN", 8.0)

        def custom_calculate_fighter_stats(mass: float) -> dict:
            hp = hp_min + (mass - min_mass) * (hp_max - hp_min) / (max_mass - min_mass)
            stamina = stam_max - (mass - min_mass) * (stam_max - stam_min) / (max_mass - min_mass)
            return {"max_hp": round(hp, 1), "max_stamina": round(stamina, 1)}

        poc_minimal.calculate_fighter_stats = custom_calculate_fighter_stats

    try:
        # Create fighter specs
        spec_a = {
            "name": fighter_a_archetype.name,
            "mass": fighter_a_archetype.mass,
            "decide_fn": fighter_a_archetype.decide_fn
        }

        spec_b = {
            "name": fighter_b_archetype.name,
            "mass": fighter_b_archetype.mass,
            "decide_fn": fighter_b_archetype.decide_fn
        }

        # Initialize fighters
        fighter_a = FighterState(spec_a["name"], spec_a["mass"], position=2.0)
        fighter_b = FighterState(spec_b["name"], spec_b["mass"], position=8.0)
        arena = Arena1D(fighter_a, fighter_b)

        # Telemetry collection
        telemetry = {
            'stamina_samples': [],
            'hp_changes': [],
            'actions': [],
            'collision_count': 0,
            'collision_damages': [],
            'position_samples': [],
            'velocity_samples': [],
            'tick': 0
        }

        # Run match
        max_ticks = 600
        for tick in range(max_ticks):
            # Sample telemetry every 5 ticks
            if tick % 5 == 0:
                avg_stamina = (fighter_a.stamina / fighter_a.max_stamina +
                              fighter_b.stamina / fighter_b.max_stamina) / 2
                telemetry['stamina_samples'].append(avg_stamina)
                telemetry['hp_changes'].append((fighter_a.hp, fighter_b.hp))
                telemetry['position_samples'].append((fighter_a.position, fighter_b.position))
                avg_velocity = (abs(fighter_a.velocity) + abs(fighter_b.velocity)) / 2
                telemetry['velocity_samples'].append(avg_velocity)

            # Generate snapshots
            from poc_minimal import generate_snapshot
            snapshot_a = generate_snapshot(fighter_a, fighter_b, tick)
            snapshot_b = generate_snapshot(fighter_b, fighter_a, tick)

            # Get actions
            action_a = spec_a["decide_fn"](snapshot_a)
            action_b = spec_b["decide_fn"](snapshot_b)

            telemetry['actions'].append(action_a['acceleration'])

            # Execute step
            events = arena.step(action_a, action_b)

            # Track collisions and damage
            for event in events:
                if event['type'] == 'COLLISION':
                    telemetry['collision_count'] += 1
                    avg_damage = (event.get('damage_to_a', 0) + event.get('damage_to_b', 0)) / 2
                    telemetry['collision_damages'].append(avg_damage)

            # Check win conditions
            if not fighter_a.is_alive():
                telemetry['winner'] = fighter_b.name
                telemetry['winner_hp'] = fighter_b.hp
                telemetry['winner_max_hp'] = fighter_b.max_hp
                telemetry['reason'] = 'KO'
                telemetry['tick'] = tick
                break

            if not fighter_b.is_alive():
                telemetry['winner'] = fighter_a.name
                telemetry['winner_hp'] = fighter_a.hp
                telemetry['winner_max_hp'] = fighter_a.max_hp
                telemetry['reason'] = 'KO'
                telemetry['tick'] = tick
                break
        else:
            # Timeout
            if fighter_a.hp > fighter_b.hp:
                telemetry['winner'] = fighter_a.name
                telemetry['winner_hp'] = fighter_a.hp
                telemetry['winner_max_hp'] = fighter_a.max_hp
            else:
                telemetry['winner'] = fighter_b.name
                telemetry['winner_hp'] = fighter_b.hp
                telemetry['winner_max_hp'] = fighter_b.max_hp
            telemetry['reason'] = 'timeout'
            telemetry['tick'] = max_ticks

        # Calculate derived metrics
        telemetry['winner_hp_pct'] = telemetry['winner_hp'] / telemetry['winner_max_hp']
        if telemetry['collision_damages']:
            telemetry['avg_collision_damage'] = sum(telemetry['collision_damages']) / len(telemetry['collision_damages'])
        else:
            telemetry['avg_collision_damage'] = 0

        return telemetry

    finally:
        # Restore original values
        for param_name, original_value in original_values.items():
            if param_name == 'STANCES':
                poc_minimal.STANCES = original_value
            elif param_name == 'calculate_fighter_stats':
                poc_minimal.calculate_fighter_stats = original_value
            else:
                setattr(poc_minimal, param_name, original_value)


# ============================================================================
# PARAMETER SEARCH
# ============================================================================

def sample_parameters(param_space: Dict) -> Dict:
    """Randomly sample a parameter set from the search space."""
    params = {}
    for name, (min_val, max_val) in param_space.items():
        params[name] = random.uniform(min_val, max_val)
    return params


def evaluate_parameter_set(params: Dict, matchups: List[Tuple]) -> Dict:
    """
    Evaluate a parameter set across all test matchups.

    Returns dict with aggregate scores and individual match results.
    """
    match_results = []

    for fighter_a, fighter_b in matchups:
        try:
            telemetry = run_instrumented_match(fighter_a, fighter_b, params)
            quality_scores = calculate_match_quality(telemetry)
            overall_score = aggregate_quality_score(quality_scores)

            match_results.append({
                'matchup': f"{fighter_a.name} vs {fighter_b.name}",
                'quality_scores': quality_scores,
                'overall_score': overall_score,
                'winner': telemetry.get('winner'),
                'duration': telemetry.get('tick')
            })
        except Exception as e:
            # If match fails (e.g., extreme params cause crash), score as 0
            print(f"   ERROR in {fighter_a.name} vs {fighter_b.name}: {e}")
            match_results.append({
                'matchup': f"{fighter_a.name} vs {fighter_b.name}",
                'quality_scores': {},
                'overall_score': 0.0,
                'error': str(e)
            })

    # Aggregate across all matches - PURE SPECTACLE SCORE
    # We don't care WHO wins, only QUALITY of fights!
    avg_score = sum(m['overall_score'] for m in match_results) / len(match_results)

    # Win distribution (for info only, not used in scoring)
    winners = [m.get('winner') for m in match_results if 'winner' in m]
    win_counts = {}
    for winner in winners:
        # Extract archetype (Light/Mid/Heavy)
        if "Light" in winner:
            archetype = "Light"
        elif "Heavy" in winner or "Tank" in winner:
            archetype = "Heavy"
        else:
            archetype = "Mid"

        win_counts[archetype] = win_counts.get(archetype, 0) + 1

    return {
        'params': params,
        'avg_score': avg_score,
        'match_results': match_results,
        'win_distribution': win_counts
    }


def run_parameter_search(num_samples: int = 100, top_n: int = 5) -> List[Dict]:
    """
    Search the parameter space and return the top N parameter sets.
    """
    print(f"Starting parameter search with {num_samples} samples...")
    print(f"Testing {len(TEST_FIGHTERS)} fighter archetypes")
    print()

    # Generate all matchups (each fighter vs each other fighter)
    matchups = []
    for i, fa in enumerate(TEST_FIGHTERS):
        for j, fb in enumerate(TEST_FIGHTERS):
            if i < j:  # Avoid duplicates and self-matches
                matchups.append((fa, fb))

    print(f"Running {len(matchups)} matchups per parameter set")
    print("=" * 70)
    print()

    results = []

    for sample_idx in range(num_samples):
        # Sample random parameters
        params = sample_parameters(PARAM_SPACE)

        # Evaluate
        result = evaluate_parameter_set(params, matchups)
        results.append(result)

        # Progress update
        if (sample_idx + 1) % 10 == 0:
            print(f"Completed {sample_idx + 1}/{num_samples} samples... "
                  f"Best so far: {max(r['avg_score'] for r in results):.3f}")

    # Sort by score
    results.sort(key=lambda r: r['avg_score'], reverse=True)

    print()
    print("=" * 70)
    print(f"Search complete! Top {top_n} parameter sets:")
    print()

    for idx, result in enumerate(results[:top_n], 1):
        print(f"#{idx} - Score: {result['avg_score']:.3f}")
        print(f"   Win Distribution: {result['win_distribution']}")
        print("   Parameters:")
        for param_name, param_value in result['params'].items():
            print(f"      {param_name}: {param_value:.4f}")
        print()

    return results[:top_n]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search for optimal world parameters")
    parser.add_argument("--samples", type=int, default=50, help="Number of parameter sets to try")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Run search
    top_results = run_parameter_search(num_samples=args.samples, top_n=args.top)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(top_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
