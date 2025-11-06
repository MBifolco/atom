#!/usr/bin/env python3
"""
Comprehensive fighter testing suite
Tests all fighters and generates a summary report
"""

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import re

@dataclass
class TestResult:
    fighter1: str
    fighter2: str
    winner: str
    collisions: int
    duration: int
    final_hp_a: float
    final_hp_b: float
    spectacle_score: float

def run_fight(fighter1: str, fighter2: str, seed: int = 42) -> TestResult:
    """Run a single fight and parse the results"""
    cmd = [
        sys.executable,
        "atom_fight.py",
        f"fighters/examples/{fighter1}.py",
        f"fighters/examples/{fighter2}.py",
        "--seed", str(seed)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/biff/eng/atom",
            timeout=30
        )

        output = result.stdout

        # Parse output
        winner_match = re.search(r"Winner: (.+)", output)
        duration_match = re.search(r"Duration: (\d+) ticks", output)
        hp_match = re.search(r"Final HP: ([\d.]+) vs ([\d.]+)", output)
        collision_match = re.search(r"Collisions: (\d+)", output)
        spectacle_match = re.search(r"Spectacle Score: ([\d.]+)", output)

        winner = winner_match.group(1) if winner_match else "UNKNOWN"
        duration = int(duration_match.group(1)) if duration_match else 0
        final_hp_a = float(hp_match.group(1)) if hp_match else 0.0
        final_hp_b = float(hp_match.group(2)) if hp_match else 0.0
        collisions = int(collision_match.group(1)) if collision_match else 0
        spectacle = float(spectacle_match.group(1)) if spectacle_match else 0.0

        return TestResult(
            fighter1=fighter1,
            fighter2=fighter2,
            winner=winner,
            collisions=collisions,
            duration=duration,
            final_hp_a=final_hp_a,
            final_hp_b=final_hp_b,
            spectacle_score=spectacle
        )
    except Exception as e:
        print(f"Error running {fighter1} vs {fighter2}: {e}", file=sys.stderr)
        return TestResult(
            fighter1=fighter1,
            fighter2=fighter2,
            winner="ERROR",
            collisions=0,
            duration=0,
            final_hp_a=0.0,
            final_hp_b=0.0,
            spectacle_score=0.0
        )

def main():
    print("=" * 80)
    print("FIGHTER TESTING SUITE".center(80))
    print("=" * 80)
    print()

    # Define test matrix
    tests = [
        # Berserker tests
        ("berserker", "tank"),
        ("berserker", "rusher"),
        ("berserker", "balanced"),
        ("berserker", "grappler"),

        # Counter Puncher tests
        ("counter_puncher", "tank"),
        ("counter_puncher", "rusher"),
        ("counter_puncher", "balanced"),
        ("counter_puncher", "grappler"),

        # Zoner tests
        ("zoner", "tank"),
        ("zoner", "rusher"),
        ("zoner", "balanced"),
        ("zoner", "grappler"),

        # Dodger tests (limited)
        ("dodger", "tank"),
        ("dodger", "rusher"),

        # Hit and Run tests (limited)
        ("hit_and_run", "tank"),
        ("hit_and_run", "rusher"),

        # Stamina Manager tests (limited)
        ("stamina_manager", "tank"),
        ("stamina_manager", "rusher"),
    ]

    results: List[TestResult] = []

    # Run all tests
    total_tests = len(tests)
    for idx, (fighter1, fighter2) in enumerate(tests, 1):
        print(f"[{idx}/{total_tests}] Testing {fighter1} vs {fighter2}...", end=" ", flush=True)
        result = run_fight(fighter1, fighter2)
        results.append(result)
        print(f"Winner: {result.winner} | Collisions: {result.collisions}")

    print()
    print("=" * 80)
    print("RESULTS SUMMARY".center(80))
    print("=" * 80)
    print()

    # Print detailed results table
    print("DETAILED RESULTS")
    print("-" * 80)
    header = f"{'Fighter 1':<18} {'Fighter 2':<12} {'Winner':<18} {'Cols':<6} {'Dur':<6} {'HP A':<8} {'HP B':<8} {'Spec':<6}"
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r.fighter1:<18} {r.fighter2:<12} {r.winner:<18} {r.collisions:<6} {r.duration:<6} {r.final_hp_a:<8.1f} {r.final_hp_b:<8.1f} {r.spectacle_score:<6.3f}"
        print(row)

    print()
    print("=" * 80)
    print("FIGHTER PERFORMANCE ANALYSIS".center(80))
    print("=" * 80)
    print()

    # Analyze each fighter's performance
    fighters = ["berserker", "counter_puncher", "zoner", "dodger", "hit_and_run", "stamina_manager"]

    for fighter in fighters:
        fighter_results = [r for r in results if r.fighter1 == fighter]

        if not fighter_results:
            continue

        wins = sum(1 for r in fighter_results if r.winner == fighter)
        total_matches = len(fighter_results)
        avg_collisions = sum(r.collisions for r in fighter_results) / total_matches
        avg_spectacle = sum(r.spectacle_score for r in fighter_results) / total_matches
        low_collision_matches = sum(1 for r in fighter_results if r.collisions < 30)

        print(f"{fighter.upper().replace('_', ' ')}")
        print(f"  Win Rate: {wins}/{total_matches} ({100*wins/total_matches:.1f}%)")
        print(f"  Avg Collisions: {avg_collisions:.1f}")
        print(f"  Avg Spectacle Score: {avg_spectacle:.3f}")
        print(f"  Low Collision Matches: {low_collision_matches}/{total_matches}")

        # Identify issues
        issues = []
        if avg_collisions < 30:
            issues.append("⚠️  Too few collisions - fighter may be too passive or evasive")
        if wins == 0 and total_matches >= 3:
            issues.append("⚠️  No wins - fighter is underperforming")
        if low_collision_matches > total_matches / 2:
            issues.append("⚠️  Consistently low engagement")
        if avg_spectacle < 0.4:
            issues.append("⚠️  Poor spectacle score - not entertaining")

        if issues:
            print("  Issues:")
            for issue in issues:
                print(f"    {issue}")
        else:
            print("  ✓ Performance appears adequate")

        print()

    print("=" * 80)
    print("RECOMMENDATIONS".center(80))
    print("=" * 80)
    print()

    # Identify fighters needing improvement
    needs_improvement = []

    for fighter in fighters:
        fighter_results = [r for r in results if r.fighter1 == fighter]
        if not fighter_results:
            continue

        avg_collisions = sum(r.collisions for r in fighter_results) / len(fighter_results)
        wins = sum(1 for r in fighter_results if r.winner == fighter)
        total = len(fighter_results)

        priority = 0
        reasons = []

        if avg_collisions < 30:
            priority += 3
            reasons.append(f"Low engagement ({avg_collisions:.1f} avg collisions)")

        if wins == 0 and total >= 3:
            priority += 3
            reasons.append(f"No wins in {total} matches")
        elif wins < total * 0.3:
            priority += 2
            reasons.append(f"Low win rate ({wins}/{total})")

        if priority > 0:
            needs_improvement.append((fighter, priority, reasons))

    # Sort by priority
    needs_improvement.sort(key=lambda x: x[1], reverse=True)

    if needs_improvement:
        print("Fighters requiring improvement (highest priority first):")
        print()
        for fighter, priority, reasons in needs_improvement:
            print(f"{priority}. {fighter.upper().replace('_', ' ')} (Priority: {priority})")
            for reason in reasons:
                print(f"   - {reason}")
            print()
    else:
        print("All fighters performing within acceptable parameters!")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
