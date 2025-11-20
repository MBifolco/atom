#!/usr/bin/env python3
"""
Test a single match with detailed output to debug combat mechanics.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # atom root
sys.path.insert(0, str(Path(__file__).parent))  # population dir

from fighter_loader import load_fighter
from simple_combat_env import SimpleCombatEnv


def run_detailed_match(fighter1_path: str, fighter2_path: str = None, max_ticks: int = 50):
    """Run a match with detailed tick-by-tick output."""

    print("=" * 70)
    print("DETAILED MATCH ANALYSIS")
    print("=" * 70)

    # Load fighters
    print(f"\nLoading fighter 1 from: {fighter1_path}")
    fighter1 = load_fighter(fighter1_path, verbose=True)

    fighter2 = None
    if fighter2_path:
        print(f"\nLoading fighter 2 from: {fighter2_path}")
        fighter2 = load_fighter(fighter2_path, verbose=True)
    else:
        print("\nFighter 2: Random opponent")

    # Create environment
    env = SimpleCombatEnv(opponent_func=fighter2)
    obs, _ = env.reset()

    print("\n" + "-" * 70)
    print("MATCH START")
    print("-" * 70)
    print(f"Fighter 1 Position: {env.pos:.2f}")
    print(f"Fighter 2 Position: {env.opp_pos:.2f}")
    print(f"Initial Distance: {abs(env.opp_pos - env.pos):.2f}")
    print(f"Arena Width: {env.arena_width}")

    # Run match
    done = False
    tick = 0

    print("\n" + "-" * 70)
    print("TICK-BY-TICK COMBAT")
    print("-" * 70)

    while not done and tick < max_ticks:
        # Get fighter decision
        snapshot = env._make_snapshot(for_opponent=False)
        decision = fighter1(snapshot)

        # Convert to action
        acceleration = decision["acceleration"]
        stance = decision["stance"]
        stance_map = {"neutral": 0, "extended": 1, "retracted": 2, "defending": 3}
        stance_idx = stance_map.get(stance, 0)
        norm_accel = np.clip(acceleration / 4.5, -1, 1)
        action = np.array([norm_accel, stance_idx], dtype=np.float32)

        # Store state before step
        prev_hp = env.hp
        prev_opp_hp = env.opp_hp
        prev_dist = abs(env.opp_pos - env.pos)

        # Step
        obs, reward, done, _, info = env.step(action)

        # Calculate what happened
        damage_dealt = prev_opp_hp - env.opp_hp
        damage_taken = prev_hp - env.hp
        new_dist = abs(env.opp_pos - env.pos)

        # Print tick info
        if tick < 20 or damage_dealt > 0 or damage_taken > 0:  # First 20 ticks or combat
            print(f"\nTick {tick:3d}:")
            print(f"  Positions: F1={env.pos:5.2f}, F2={env.opp_pos:5.2f}, Dist={new_dist:5.2f}")
            print(f"  F1 Action: accel={acceleration:5.2f}, stance={stance:10s}, stamina={env.stamina:.2f}")
            print(f"  Health:    F1={env.hp:5.1f}, F2={env.opp_hp:5.1f}")

            if damage_dealt > 0 or damage_taken > 0:
                print(f"  COMBAT! Dealt={damage_dealt:.1f}, Taken={damage_taken:.1f}")
                if new_dist < 1.5:
                    print(f"    -> In range (dist < 1.5)")
                    if env.stamina <= 0:
                        print(f"    -> F1 out of stamina!")
                    if env.opp_stamina <= 0:
                        print(f"    -> F2 out of stamina!")

            print(f"  Reward: {reward:6.2f}")

        tick += 1

    # Final results
    print("\n" + "=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)

    if info["won"]:
        print("WINNER: Fighter 1")
    elif env.hp <= 0:
        print("WINNER: Fighter 2")
    else:
        print("DRAW (timeout)")

    print(f"\nFinal Stats:")
    print(f"  Fighter 1: HP={env.hp:.1f}, Damage Dealt={info['damage_dealt']:.1f}")
    print(f"  Fighter 2: HP={env.opp_hp:.1f}, Damage Taken={info['damage_taken']:.1f}")
    print(f"  Total Ticks: {info['tick']}")
    print(f"  Final Distance: {abs(env.opp_pos - env.pos):.2f}")


if __name__ == "__main__":
    # Test rusher vs tank
    base_path = "/home/biff/eng/atom"

    print("Testing RUSHER vs TANK\n")
    run_detailed_match(
        f"{base_path}/fighters/examples/rusher.py",
        f"{base_path}/fighters/examples/tank.py",
        max_ticks=50
    )

    print("\n" + "=" * 70)
    print("\nTesting BALANCED vs Random\n")
    run_detailed_match(
        f"{base_path}/fighters/examples/balanced.py",
        None,
        max_ticks=30
    )