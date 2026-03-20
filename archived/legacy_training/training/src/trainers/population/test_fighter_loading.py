#!/usr/bin/env python3
"""
Diagnostic tool to test if hardcoded fighters load and work correctly.
This helps debug the 0% win rate issue in population training.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # atom root
sys.path.insert(0, str(Path(__file__).parent))  # population dir

from fighter_loader import load_hardcoded_fighters, test_fighter_in_combat
from simple_combat_env import SimpleCombatEnv
import numpy as np


def test_fighter_vs_random(decide_func, fighter_name: str, num_matches: int = 5):
    """Test a fighter against a random opponent in SimpleCombatEnv."""
    print(f"\nTesting {fighter_name} vs Random in SimpleCombatEnv ({num_matches} matches):")

    wins = 0
    total_damage_dealt = 0
    total_damage_taken = 0

    for match in range(num_matches):
        # Create environment with random opponent
        env = SimpleCombatEnv(opponent_func=None)
        obs, _ = env.reset()

        done = False
        steps = 0

        while not done:
            # Create snapshot for fighter decision
            snapshot = env._make_snapshot(for_opponent=False)

            # Get fighter decision
            try:
                decision = decide_func(snapshot)
                acceleration = decision["acceleration"]
                stance = decision["stance"]

                # Convert to action format expected by env
                stance_map = {"neutral": 0, "extended": 1, "retracted": 2, "defending": 3}
                stance_idx = stance_map.get(stance, 0)

                # Normalize acceleration to [-1, 1]
                norm_accel = np.clip(acceleration / 4.5, -1, 1)

                action = np.array([norm_accel, stance_idx], dtype=np.float32)

            except Exception as e:
                print(f"    Error getting decision: {e}")
                action = np.array([0, 0], dtype=np.float32)

            # Step environment
            obs, reward, done, _, info = env.step(action)
            steps += 1

        # Record results
        if info["won"]:
            wins += 1
            result = "WIN"
        else:
            result = "LOSS"

        total_damage_dealt += info["damage_dealt"]
        total_damage_taken += info["damage_taken"]

        print(f"  Match {match + 1}: {result} - "
              f"HP: {info['hp']:.1f}, Opp HP: {info['opp_hp']:.1f}, "
              f"Damage dealt: {info['damage_dealt']:.1f}, "
              f"Damage taken: {info['damage_taken']:.1f}, "
              f"Ticks: {info['tick']}")

    win_rate = wins / num_matches
    avg_damage_dealt = total_damage_dealt / num_matches
    avg_damage_taken = total_damage_taken / num_matches

    print(f"\n  Summary for {fighter_name}:")
    print(f"    Win rate: {win_rate:.1%} ({wins}/{num_matches})")
    print(f"    Avg damage dealt: {avg_damage_dealt:.1f}")
    print(f"    Avg damage taken: {avg_damage_taken:.1f}")

    return win_rate


def test_fighter_vs_fighter(fighter1_func, fighter1_name: str,
                           fighter2_func, fighter2_name: str,
                           num_matches: int = 5):
    """Test one fighter against another in SimpleCombatEnv."""
    print(f"\nTesting {fighter1_name} vs {fighter2_name} ({num_matches} matches):")

    wins = 0

    for match in range(num_matches):
        # Create environment with fighter2 as opponent
        env = SimpleCombatEnv(opponent_func=fighter2_func)
        obs, _ = env.reset()

        done = False

        while not done:
            # Get fighter1 decision
            snapshot = env._make_snapshot(for_opponent=False)

            try:
                decision = fighter1_func(snapshot)
                acceleration = decision["acceleration"]
                stance = decision["stance"]

                # Convert to action format
                stance_map = {"neutral": 0, "extended": 1, "retracted": 2, "defending": 3}
                stance_idx = stance_map.get(stance, 0)
                norm_accel = np.clip(acceleration / 4.5, -1, 1)
                action = np.array([norm_accel, stance_idx], dtype=np.float32)

            except Exception as e:
                print(f"    Error: {e}")
                action = np.array([0, 0], dtype=np.float32)

            obs, reward, done, _, info = env.step(action)

        if info["won"]:
            wins += 1
            result = "WIN"
        else:
            result = "LOSS"

        print(f"  Match {match + 1}: {fighter1_name} {result}")

    win_rate = wins / num_matches
    print(f"  {fighter1_name} vs {fighter2_name}: {win_rate:.1%} win rate")

    return win_rate


def main():
    """Main diagnostic routine."""
    print("=" * 70)
    print("FIGHTER LOADING DIAGNOSTIC TOOL")
    print("=" * 70)

    # Step 1: Try to load all hardcoded fighters
    print("\n1. LOADING HARDCODED FIGHTERS")
    print("-" * 40)

    fighters = load_hardcoded_fighters("/home/biff/eng/atom", verbose=True)

    if not fighters:
        print("\n✗ CRITICAL: No fighters could be loaded!")
        print("This explains the 0% win rate in population training.")
        print("\nPossible causes:")
        print("  1. Fighter files not found at expected paths")
        print("  2. Import errors in fighter files")
        print("  3. Missing dependencies")
        return 1

    # Step 2: Test each fighter in simulated combat
    print("\n2. TESTING FIGHTERS IN COMBAT SIMULATION")
    print("-" * 40)

    for name, decide_func in fighters.items():
        print(f"\n{name.upper()}:")
        success = test_fighter_in_combat(decide_func, num_steps=10, verbose=False)
        if success:
            print("  ✓ Passed combat simulation")
        else:
            print("  ✗ Failed combat simulation")

    # Step 3: Test fighters in SimpleCombatEnv
    print("\n3. TESTING IN SIMPLECOMBATENV")
    print("-" * 40)

    win_rates = {}
    for name, decide_func in fighters.items():
        win_rate = test_fighter_vs_random(decide_func, name, num_matches=5)
        win_rates[name] = win_rate

    # Step 4: Test fighters against each other
    print("\n4. TESTING FIGHTER VS FIGHTER")
    print("-" * 40)

    if len(fighters) >= 2:
        fighter_names = list(fighters.keys())
        for i in range(len(fighter_names)):
            for j in range(i + 1, len(fighter_names)):
                name1 = fighter_names[i]
                name2 = fighter_names[j]
                test_fighter_vs_fighter(
                    fighters[name1], name1,
                    fighters[name2], name2,
                    num_matches=3
                )

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"\nFighters loaded: {list(fighters.keys())}")
    print("\nWin rates vs random opponent:")
    for name, rate in win_rates.items():
        print(f"  {name}: {rate:.1%}")

    if all(rate > 0 for rate in win_rates.values()):
        print("\n✓ All fighters can win against random opponents")
        print("  The 0% win rate in training is likely due to:")
        print("  1. Training not using these fighters as opponents")
        print("  2. Trained models not learning effective strategies")
        print("  3. Evaluation code silently failing")
    else:
        print("\n✗ Some fighters cannot beat random opponents")
        print("  This may indicate issues with the SimpleCombatEnv")

    return 0


if __name__ == "__main__":
    sys.exit(main())