#!/usr/bin/env python3
"""
Test JIT-compiled JAX Physics

Quick test to ensure JIT-compiled physics works correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.arena.arena_1d import Arena1D
from src.arena.arena_1d_jax_jit import Arena1DJAXJit
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig
import time

def test_jit_correctness():
    """Test that JIT version produces correct results."""
    print("\n" + "="*80)
    print("TESTING JIT CORRECTNESS")
    print("="*80)

    config = WorldConfig()
    fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
    fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

    # Python version
    arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)

    # JAX JIT version
    arena_jax = Arena1DJAXJit(fighter_a, fighter_b, config, seed=42)

    # Run 10 steps
    print("\nRunning 10 steps with same actions...")
    for step in range(10):
        action_a = {"acceleration": 1.0, "stance": "neutral"}
        action_b = {"acceleration": -1.0, "stance": "neutral"}

        arena_py.step(action_a, action_b)
        arena_jax.step(action_a, action_b)

    # Check if results match
    print(f"\nAfter 10 steps:")
    print(f"  Python - Position A: {arena_py.fighter_a.position:.6f}, HP A: {arena_py.fighter_a.hp:.6f}")
    print(f"  JAX    - Position A: {arena_jax.fighter_a.position:.6f}, HP A: {arena_jax.fighter_a.hp:.6f}")

    pos_diff = abs(arena_py.fighter_a.position - arena_jax.fighter_a.position)
    hp_diff = abs(arena_py.fighter_a.hp - arena_jax.fighter_a.hp)

    print(f"\n  Difference - Position: {pos_diff:.2e}, HP: {hp_diff:.2e}")

    if pos_diff < 1e-5 and hp_diff < 1e-5:
        print("\n✅ JIT physics matches Python physics!")
        return True
    else:
        print("\n❌ JIT physics does NOT match Python physics!")
        return False


def test_jit_performance():
    """Benchmark JIT-compiled physics performance."""
    print("\n" + "="*80)
    print("BENCHMARKING JIT PERFORMANCE")
    print("="*80)

    config = WorldConfig()
    NUM_EPISODES = 1000
    MAX_TICKS = 250

    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES:,}")
    print(f"  Max ticks per episode: {MAX_TICKS}")

    # Benchmark JIT version
    print(f"\n{'='*80}")
    print("JAX JIT Physics (Phase 3)")
    print("="*80)

    total_ticks = 0
    start_time = time.time()

    for episode in range(NUM_EPISODES):
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config, seed=episode)

        import numpy as np
        np.random.seed(episode)
        for tick in range(MAX_TICKS):
            accel_a = np.random.uniform(-1, 1)
            accel_b = np.random.uniform(-1, 1)
            stance_a = np.random.choice(["neutral", "extended", "retracted", "defending"])
            stance_b = np.random.choice(["neutral", "extended", "retracted", "defending"])

            action_a = {"acceleration": accel_a, "stance": stance_a}
            action_b = {"acceleration": accel_b, "stance": stance_b}

            arena.step(action_a, action_b)
            total_ticks += 1

            if arena.is_finished():
                break

    end_time = time.time()
    jit_elapsed = end_time - start_time
    jit_tps = total_ticks / jit_elapsed

    print(f"✅ Completed in {jit_elapsed:.2f} seconds")
    print(f"   Throughput: {jit_tps:,.0f} ticks/second")

    return jit_elapsed, jit_tps


def main():
    print("\n" + "="*80)
    print("JAX JIT PHYSICS TEST")
    print("="*80)

    # Test correctness first
    if not test_jit_correctness():
        print("\n❌ Correctness test failed - not running performance test")
        return

    # Test performance
    jit_time, jit_tps = test_jit_performance()

    # Compare with Phase 1 results (from docs)
    print("\n" + "="*80)
    print("COMPARISON WITH PHASE 1")
    print("="*80)

    phase1_tps = 2313  # From Phase 1 benchmark (no JIT)
    phase0_tps = 57088  # Python baseline

    print(f"\nPhase 0 (Python):        {phase0_tps:,} ticks/sec")
    print(f"Phase 1 (JAX, no JIT):   {phase1_tps:,} ticks/sec")
    print(f"Phase 3 (JAX with JIT):  {jit_tps:,.0f} ticks/sec")

    speedup_vs_phase1 = jit_tps / phase1_tps
    speedup_vs_python = jit_tps / phase0_tps

    print(f"\nSpeedup vs Phase 1: {speedup_vs_phase1:.2f}x")
    print(f"Speedup vs Python:  {speedup_vs_python:.2f}x")

    if speedup_vs_python > 1.0:
        print(f"\n🚀 JAX JIT is {speedup_vs_python:.2f}x FASTER than Python!")
    elif speedup_vs_python < 1.0:
        print(f"\n⚠️  JAX JIT is {1/speedup_vs_python:.2f}x SLOWER than Python")
    else:
        print(f"\n🤷 Similar performance to Python")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
