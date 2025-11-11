#!/usr/bin/env python3
"""
Benchmark JAX Physics vs Python Physics

Measures raw physics simulation speed (not RL training).
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.arena.arena_1d import Arena1D
from src.arena.arena_1d_jax import Arena1DJAX
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig
import numpy as np


def benchmark_physics(use_jax: bool, num_episodes: int = 1000, max_ticks: int = 250):
    """
    Benchmark physics simulation speed.

    Args:
        use_jax: Use JAX physics if True, Python if False
        num_episodes: Number of episodes to simulate
        max_ticks: Maximum ticks per episode

    Returns:
        (total_time, ticks_per_second)
    """
    config = WorldConfig()
    ArenaClass = Arena1DJAX if use_jax else Arena1D

    total_ticks = 0
    start_time = time.time()

    for episode in range(num_episodes):
        # Create fresh fighters
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena = ArenaClass(fighter_a, fighter_b, config, seed=episode)

        # Run episode with random actions
        np.random.seed(episode)
        for tick in range(max_ticks):
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
    elapsed = end_time - start_time
    ticks_per_sec = total_ticks / elapsed

    return elapsed, ticks_per_sec


def main():
    print("\n" + "=" * 80)
    print("JAX PHYSICS BENCHMARK")
    print("=" * 80)

    NUM_EPISODES = 1000
    MAX_TICKS = 250

    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES:,}")
    print(f"  Max ticks per episode: {MAX_TICKS}")
    print(f"  Total ticks (approx): {NUM_EPISODES * MAX_TICKS:,}")

    # Benchmark Python physics
    print(f"\n{'=' * 80}")
    print("PYTHON PHYSICS")
    print("=" * 80)
    print("Running...")

    py_time, py_tps = benchmark_physics(use_jax=False, num_episodes=NUM_EPISODES, max_ticks=MAX_TICKS)

    print(f"✅ Completed in {py_time:.2f} seconds")
    print(f"   Throughput: {py_tps:,.0f} ticks/second")

    # Benchmark JAX physics
    print(f"\n{'=' * 80}")
    print("JAX PHYSICS")
    print("=" * 80)
    print("Running...")

    jax_time, jax_tps = benchmark_physics(use_jax=True, num_episodes=NUM_EPISODES, max_ticks=MAX_TICKS)

    print(f"✅ Completed in {jax_time:.2f} seconds")
    print(f"   Throughput: {jax_tps:,.0f} ticks/second")

    # Compare
    speedup = py_time / jax_time
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)

    print(f"\nPython Physics:")
    print(f"  Time: {py_time:.2f}s")
    print(f"  Throughput: {py_tps:,.0f} ticks/sec")

    print(f"\nJAX Physics:")
    print(f"  Time: {jax_time:.2f}s")
    print(f"  Throughput: {jax_tps:,.0f} ticks/sec")

    print(f"\nSpeedup:")
    if speedup > 1.0:
        print(f"  🚀 JAX is {speedup:.2f}x FASTER ({(speedup-1)*100:.1f}% improvement)")
    elif speedup < 1.0:
        print(f"  ⚠️  JAX is {1/speedup:.2f}x SLOWER ({(1-speedup)*100:.1f}% slower)")
    else:
        print(f"  🤷 Equivalent performance")

    print("=" * 80)

    # Analysis
    print("\nANALYSIS:")
    if speedup > 1.5:
        print("✅ JAX provides significant speedup - recommend using use_jax=True")
    elif speedup > 1.1:
        print("✅ JAX provides modest speedup - consider using use_jax=True")
    elif speedup > 0.9:
        print("🤷 Performance is roughly equivalent - either option is fine")
    else:
        print("⚠️  Python is faster - stick with use_jax=False")
        print("   (This may be due to compilation overhead on small batches)")

    print("\nNOTE: This benchmarks raw physics simulation, not full RL training.")
    print("      Training speed also depends on neural network and data collection.")


if __name__ == "__main__":
    main()
