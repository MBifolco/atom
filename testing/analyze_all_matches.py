#!/usr/bin/env python3
"""
Comprehensive match-by-match behavior analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

FIGHTERS = ["tank", "rusher", "balanced", "grappler", "berserker", "zoner", "dodger"]

def analyze_match(fighter_a: str, fighter_b: str, seed: int = 42) -> Dict:
    """Run and deeply analyze a single match."""
    # Run the match
    cmd = [
        "python", "atom_fight.py",
        f"fighters/examples/{fighter_a}.py",
        f"fighters/examples/{fighter_b}.py",
        "--seed", str(seed),
        "--save", f"outputs/{fighter_a}_vs_{fighter_b}.json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Load telemetry
    with open(f"outputs/{fighter_a}_vs_{fighter_b}.json", 'r') as f:
        data = json.load(f)

    ticks = data['telemetry']['ticks']

    # Initialize tracking
    stats = {
        'match': f"{fighter_a} vs {fighter_b}",
        'total_ticks': len(ticks),
        'fighter_a': {
            'name': fighter_a,
            'wall_time': 0,
            'wall_stuck': 0,
            'stationary': 0,
            'retreating': 0,
            'advancing': 0,
            'stances': {},
            'min_hp': 100,
            'aggressive_moves': 0,  # extended + advancing
            'defensive_moves': 0,    # defending + retreating
            'position_sum': 0,
            'extreme_positions': []  # Track if stuck at edges
        },
        'fighter_b': {
            'name': fighter_b,
            'wall_time': 0,
            'wall_stuck': 0,
            'stationary': 0,
            'retreating': 0,
            'advancing': 0,
            'stances': {},
            'min_hp': 100,
            'aggressive_moves': 0,
            'defensive_moves': 0,
            'position_sum': 0,
            'extreme_positions': []
        },
        'dynamics': {
            'min_distance': 100,
            'max_distance': 0,
            'close_combat_time': 0,  # <1.5m
            'chase_time': 0,
            'mutual_retreat_time': 0,
            'standoff_time': 0,
            'collisions': 0
        },
        'issues': []
    }

    arena_width = 12.0
    prev_dist = None

    for i, tick in enumerate(ticks):
        fa = tick['fighter_a']
        fb = tick['fighter_b']

        # Distance
        dist = abs(fa['position'] - fb['position'])
        stats['dynamics']['min_distance'] = min(stats['dynamics']['min_distance'], dist)
        stats['dynamics']['max_distance'] = max(stats['dynamics']['max_distance'], dist)

        if dist < 1.5:
            stats['dynamics']['close_combat_time'] += 1

        # Check for collision
        if prev_dist and dist < 0.5 and abs(dist - prev_dist) > 0.3:
            stats['dynamics']['collisions'] += 1
        prev_dist = dist

        # Analyze Fighter A
        analyze_fighter(fa, stats['fighter_a'], arena_width, i)

        # Analyze Fighter B
        analyze_fighter(fb, stats['fighter_b'], arena_width, i)

        # Match dynamics
        va = fa['velocity']
        vb = fb['velocity']

        # Chasing: one advancing fast, other retreating fast
        if (va > 1.5 and vb < -1.5) or (vb > 1.5 and va < -1.5):
            stats['dynamics']['chase_time'] += 1

        # Mutual retreat
        if va < -1.0 and vb < -1.0:
            stats['dynamics']['mutual_retreat_time'] += 1

        # Standoff
        if abs(va) < 0.5 and abs(vb) < 0.5 and dist > 2.0:
            stats['dynamics']['standoff_time'] += 1

    # Identify issues
    identify_issues(stats)

    return stats

def analyze_fighter(fighter_data: Dict, stats: Dict, arena_width: float, tick_num: int):
    """Analyze single fighter's behavior in a tick."""
    pos = fighter_data['position']
    vel = fighter_data['velocity']
    stance = fighter_data['stance']
    hp = fighter_data['hp']

    stats['min_hp'] = min(stats['min_hp'], hp)
    stats['position_sum'] += pos

    # Wall detection
    if pos <= 0.5 or pos >= arena_width - 0.5:
        stats['wall_time'] += 1
        if abs(vel) < 0.1:  # Stuck on wall
            stats['wall_stuck'] += 1

    # Track extreme positions
    if (pos <= 0.1 or pos >= arena_width - 0.1) and tick_num % 100 == 0:
        stats['extreme_positions'].append((tick_num, pos))

    # Movement
    if abs(vel) < 0.3:
        stats['stationary'] += 1
    elif vel < -0.5:
        stats['retreating'] += 1
    elif vel > 0.5:
        stats['advancing'] += 1

    # Stance
    stats['stances'][stance] = stats['stances'].get(stance, 0) + 1

    # Aggressive vs defensive
    if stance == 'extended' and vel > 0.5:
        stats['aggressive_moves'] += 1
    if stance in ['defending', 'retracted'] and vel < -0.5:
        stats['defensive_moves'] += 1

def identify_issues(stats: Dict):
    """Identify problematic behaviors."""
    issues = stats['issues']
    fa = stats['fighter_a']
    fb = stats['fighter_b']
    dyn = stats['dynamics']

    # Wall issues
    if fa['wall_stuck'] > 50:
        issues.append(f"⚠️ {fa['name']} stuck on wall for {fa['wall_stuck']} ticks")
    if fb['wall_stuck'] > 50:
        issues.append(f"⚠️ {fb['name']} stuck on wall for {fb['wall_stuck']} ticks")

    # Passivity issues
    if fa['stationary'] > 400:
        issues.append(f"⚠️ {fa['name']} too passive: {fa['stationary']} stationary ticks")
    if fb['stationary'] > 400:
        issues.append(f"⚠️ {fb['name']} too passive: {fb['stationary']} stationary ticks")

    # Retreat issues
    if fa['retreating'] > 600:
        issues.append(f"⚠️ {fa['name']} retreating excessively: {fa['retreating']} ticks")
    if fb['retreating'] > 600:
        issues.append(f"⚠️ {fb['name']} retreating excessively: {fb['retreating']} ticks")

    # Match dynamics issues
    if dyn['mutual_retreat_time'] > 100:
        issues.append(f"⚠️ Both fighters retreating: {dyn['mutual_retreat_time']} ticks")
    if dyn['standoff_time'] > 500:
        issues.append(f"⚠️ Excessive standoffs: {dyn['standoff_time']} ticks")
    if dyn['close_combat_time'] < 50 and stats['total_ticks'] >= 1000:
        issues.append(f"⚠️ Too little close combat: only {dyn['close_combat_time']} ticks")

    # Check for fighters not engaging
    total_ticks = stats['total_ticks']
    if dyn['collisions'] < 10 and total_ticks >= 500:
        issues.append(f"🚨 CRITICAL: Only {dyn['collisions']} collisions - fighters not engaging!")

def print_detailed_report(stats: Dict):
    """Print detailed match report."""
    print(f"\n{'='*80}")
    print(f"MATCH: {stats['match'].upper()}")
    print(f"Duration: {stats['total_ticks']} ticks")

    fa = stats['fighter_a']
    fb = stats['fighter_b']
    dyn = stats['dynamics']

    # Fighter A
    print(f"\n{fa['name'].upper()} Behavior:")
    print(f"  Movement Distribution:")
    print(f"    Advancing: {fa['advancing']} ticks ({fa['advancing']*100//stats['total_ticks']}%)")
    print(f"    Stationary: {fa['stationary']} ticks ({fa['stationary']*100//stats['total_ticks']}%)")
    print(f"    Retreating: {fa['retreating']} ticks ({fa['retreating']*100//stats['total_ticks']}%)")
    print(f"  Stance Usage: {fa['stances']}")
    print(f"  Wall Contact: {fa['wall_time']} ticks (stuck: {fa['wall_stuck']})")
    print(f"  Combat Style: {fa['aggressive_moves']} aggressive, {fa['defensive_moves']} defensive")
    print(f"  Average Position: {fa['position_sum']/stats['total_ticks']:.1f}")

    # Fighter B
    print(f"\n{fb['name'].upper()} Behavior:")
    print(f"  Movement Distribution:")
    print(f"    Advancing: {fb['advancing']} ticks ({fb['advancing']*100//stats['total_ticks']}%)")
    print(f"    Stationary: {fb['stationary']} ticks ({fb['stationary']*100//stats['total_ticks']}%)")
    print(f"    Retreating: {fb['retreating']} ticks ({fb['retreating']*100//stats['total_ticks']}%)")
    print(f"  Stance Usage: {fb['stances']}")
    print(f"  Wall Contact: {fb['wall_time']} ticks (stuck: {fb['wall_stuck']})")
    print(f"  Combat Style: {fb['aggressive_moves']} aggressive, {fb['defensive_moves']} defensive")
    print(f"  Average Position: {fb['position_sum']/stats['total_ticks']:.1f}")

    # Match Dynamics
    print(f"\nMatch Dynamics:")
    print(f"  Distance Range: {dyn['min_distance']:.1f}m - {dyn['max_distance']:.1f}m")
    print(f"  Close Combat: {dyn['close_combat_time']} ticks ({dyn['close_combat_time']*100//stats['total_ticks']}%)")
    print(f"  Chasing: {dyn['chase_time']} ticks")
    print(f"  Mutual Retreat: {dyn['mutual_retreat_time']} ticks")
    print(f"  Standoffs: {dyn['standoff_time']} ticks")
    print(f"  Estimated Collisions: {dyn['collisions']}")

    # Issues
    if stats['issues']:
        print(f"\n{'🚨 ISSUES DETECTED:'}")
        for issue in stats['issues']:
            print(f"  {issue}")
    else:
        print(f"\n✅ No major issues detected")

# Main execution
if __name__ == "__main__":
    print("COMPREHENSIVE FIGHTER BEHAVIOR ANALYSIS")
    print("="*80)

    # Key matchups to analyze in detail
    key_matchups = [
        ('tank', 'berserker'),    # Should be defensive vs aggressive
        ('rusher', 'dodger'),      # Chase dynamics
        ('grappler', 'zoner'),     # Close vs range
        ('balanced', 'berserker'), # Adaptive vs relentless
        ('zoner', 'rusher'),       # Range control vs pressure
        ('dodger', 'grappler'),    # Evasion vs pursuit
        ('tank', 'tank'),          # Mirror match
    ]

    all_stats = []
    for fa, fb in key_matchups:
        stats = analyze_match(fa, fb)
        all_stats.append(stats)
        print_detailed_report(stats)

    # Summary of all issues
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL ISSUES")
    print("="*80)

    critical_issues = []
    moderate_issues = []

    for stats in all_stats:
        for issue in stats['issues']:
            if '🚨' in issue:
                critical_issues.append(f"{stats['match']}: {issue}")
            else:
                moderate_issues.append(f"{stats['match']}: {issue}")

    if critical_issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"  {issue}")

    if moderate_issues:
        print("\n⚠️ MODERATE ISSUES:")
        for issue in moderate_issues:
            print(f"  {issue}")

    if not critical_issues and not moderate_issues:
        print("\n✅ All fighters performing as expected!")