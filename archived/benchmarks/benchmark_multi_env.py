#!/usr/bin/env python3
"""
Benchmark Multi-Environment Training (Level 1 Optimization)

Tests how training scales with number of parallel environments.
This is the EASIEST optimization - just increase n_envs!
"""

import sys
from pathlib import Path
import time
import argparse

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.gym_env import AtomCombatEnv
from src.arena import WorldConfig
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sbx import PPO
import importlib.util


def load_opponent(filepath: str):
    """Load opponent decision function."""
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def benchmark_n_envs(n_envs: int, timesteps: int = 50000, use_subproc: bool = True):
    """
    Benchmark training with N parallel environments.

    Args:
        n_envs: Number of parallel environments
        timesteps: Training timesteps
        use_subproc: Use SubprocVecEnv (true parallelism) vs DummyVecEnv

    Returns:
        elapsed_time, steps_per_sec, episodes_per_sec
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {n_envs} Parallel Environments")
    print(f"  VecEnv: {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}")
    print("="*80)

    # Load opponent
    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    # Create environment factory
    def make_env(seed: int):
        def _init():
            env = AtomCombatEnv(
                opponent_decision_func=opponent_func,
                config=config,
                max_ticks=250,
                seed=seed
            )
            return Monitor(env)
        return _init

    # Create vectorized environment
    env_fns = [make_env(42 + i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"✅ Created {n_envs} subprocess environments")
    else:
        vec_env = DummyVecEnv(env_fns)
        print(f"✅ Created {n_envs} dummy environments")

    # Create SBX model
    print("Creating SBX PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512 // n_envs,  # Adjust for number of envs
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="auto"
    )

    # Train
    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    # Estimate episodes per second (assuming ~250 steps per episode)
    episodes_per_sec = (steps_per_sec * n_envs) / 250

    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"   Throughput: {steps_per_sec:,.0f} steps/sec")
    print(f"   Episodes: ~{episodes_per_sec:.1f} episodes/sec")

    vec_env.close()

    return elapsed, steps_per_sec, episodes_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                       help="Number of environments to test")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Training timesteps per test")
    parser.add_argument("--use-dummy", action="store_true",
                       help="Use DummyVecEnv instead of SubprocVecEnv")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTI-ENVIRONMENT SCALING BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Timesteps per test: {args.timesteps:,}")
    print(f"  VecEnv type: {'DummyVecEnv' if args.use_dummy else 'SubprocVecEnv'}")
    print(f"  Testing: {args.envs} environments")

    results = []

    for n_envs in args.envs:
        elapsed, sps, eps = benchmark_n_envs(
            n_envs,
            timesteps=args.timesteps,
            use_subproc=not args.use_dummy
        )

        results.append({
            "n_envs": n_envs,
            "elapsed": elapsed,
            "steps_per_sec": sps,
            "episodes_per_sec": eps
        })

    # Analysis
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Envs':<10} {'Time (s)':<12} {'Steps/sec':<15} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 70)

    baseline = results[0]
    for r in results:
        speedup = baseline["elapsed"] / r["elapsed"]
        efficiency = speedup / r["n_envs"]  # How close to linear scaling
        print(f"{r['n_envs']:<10} {r['elapsed']:<12.2f} {r['steps_per_sec']:<15,.0f} {speedup:<10.2f}x {efficiency:<10.1%}")

    # Best configuration
    best = max(results, key=lambda x: x["steps_per_sec"])
    print(f"\n🏆 Best Configuration: {best['n_envs']} environments")
    print(f"   Throughput: {best['steps_per_sec']:,.0f} steps/sec")
    print(f"   Speedup: {baseline['elapsed'] / best['elapsed']:.2f}x over 1 environment")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    optimal_envs = best["n_envs"]
    print(f"\n✅ Use {optimal_envs} parallel environments for maximum throughput")
    print(f"   Expected training time reduction: {(1 - 1/(baseline['elapsed']/best['elapsed']))*100:.0f}%")
    print(f"   10 hour training → {10 * best['elapsed'] / baseline['elapsed']:.1f} hours")
    print(f"   1 hour training → {60 * best['elapsed'] / baseline['elapsed']:.0f} minutes")

    # Scaling analysis
    print(f"\nScaling Efficiency:")
    if efficiency > 0.8:
        print(f"  🚀 EXCELLENT ({efficiency:.0%}) - nearly linear scaling")
    elif efficiency > 0.6:
        print(f"  ✅ GOOD ({efficiency:.0%}) - good parallelization")
    elif efficiency > 0.4:
        print(f"  🤷 OKAY ({efficiency:.0%}) - some overhead")
    else:
        print(f"  ⚠️  POOR ({efficiency:.0%}) - high overhead, consider fewer envs")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
