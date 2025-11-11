#!/usr/bin/env python3
"""
Benchmark SBX vs SB3 Training Speed

Measures training speed for PPO using:
1. SBX (JAX-accelerated)
2. SB3 (PyTorch)

Tests with same environment, same hyperparameters, same timesteps.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.gym_env import AtomCombatEnv
from src.arena import WorldConfig
import numpy as np


def load_opponent(filepath: str):
    """Load opponent decision function from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def benchmark_ppo(use_sbx: bool, timesteps: int = 50_000, n_envs: int = 4) -> dict:
    """
    Benchmark PPO training speed.

    Args:
        use_sbx: Use SBX (True) or SB3 (False)
        timesteps: Total training timesteps
        n_envs: Number of parallel environments

    Returns:
        dict with timing results
    """
    # Import the appropriate PPO
    if use_sbx:
        from sbx import PPO
        label = "SBX (JAX)"
    else:
        from stable_baselines3 import PPO
        label = "SB3 (PyTorch)"

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {label}")
    print(f"{'='*80}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Environments: {n_envs}")

    # Load opponent
    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    # Create environments
    def make_env(seed: int):
        def _init():
            env = AtomCombatEnv(
                opponent_decision_func=opponent_func,
                config=config,
                max_ticks=250,
                fighter_mass=70.0,
                opponent_mass=70.0,
                seed=seed
            )
            return Monitor(env)
        return _init

    env_fns = [make_env(42 + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Create PPO model
    # Force CPU for PyTorch (SB3) due to ROCm compatibility issues
    # JAX (SBX) handles GPU automatically
    device = "cpu" if not use_sbx else "auto"
    print(f"\nInitializing {label} model (device: {device})...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device=device
    )

    # Benchmark training
    print(f"Training for {timesteps:,} timesteps...")
    start_time = time.time()

    model.learn(
        total_timesteps=timesteps,
        progress_bar=False
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate metrics
    steps_per_sec = timesteps / elapsed
    episodes_per_sec = steps_per_sec / 250  # Approximate (episodes ~250 steps)

    results = {
        "label": label,
        "timesteps": timesteps,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_sec,
        "episodes_per_second": episodes_per_sec,
    }

    print(f"\n✅ Completed in {elapsed:.2f} seconds")
    print(f"   Throughput: {steps_per_sec:,.0f} steps/sec ({episodes_per_sec:.1f} episodes/sec)")

    vec_env.close()

    return results


def main():
    print("\n" + "="*80)
    print("SBX vs SB3 TRAINING SPEED BENCHMARK")
    print("="*80)

    TIMESTEPS = 50_000  # 50k steps for quick benchmark
    N_ENVS = 4

    print(f"\nConfiguration:")
    print(f"  Timesteps: {TIMESTEPS:,}")
    print(f"  Parallel Environments: {N_ENVS}")
    print(f"  Opponent: Training Dummy (stationary)")

    # Benchmark SB3 (baseline)
    print("\n" + "-"*80)
    print("BASELINE: Stable-Baselines3 (PyTorch)")
    print("-"*80)
    sb3_results = benchmark_ppo(use_sbx=False, timesteps=TIMESTEPS, n_envs=N_ENVS)

    # Small pause between benchmarks
    time.sleep(2)

    # Benchmark SBX (JAX)
    print("\n" + "-"*80)
    print("OPTIMIZED: SBX (JAX)")
    print("-"*80)
    sbx_results = benchmark_ppo(use_sbx=True, timesteps=TIMESTEPS, n_envs=N_ENVS)

    # Compare results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\nStable-Baselines3 (PyTorch):")
    print(f"  Time: {sb3_results['elapsed_seconds']:.2f}s")
    print(f"  Throughput: {sb3_results['steps_per_second']:,.0f} steps/sec")
    print(f"  Episodes: {sb3_results['episodes_per_second']:.1f} episodes/sec")

    print(f"\nSBX (JAX):")
    print(f"  Time: {sbx_results['elapsed_seconds']:.2f}s")
    print(f"  Throughput: {sbx_results['steps_per_second']:,.0f} steps/sec")
    print(f"  Episodes: {sbx_results['episodes_per_second']:.1f} episodes/sec")

    # Calculate speedup
    speedup = sb3_results['elapsed_seconds'] / sbx_results['elapsed_seconds']

    print(f"\nSpeedup:")
    if speedup > 1.0:
        print(f"  🚀 SBX is {speedup:.2f}x FASTER ({(speedup-1)*100:.1f}% improvement)")
        print(f"     Training that took {sb3_results['elapsed_seconds']/60:.1f} minutes with SB3")
        print(f"     now takes {sbx_results['elapsed_seconds']/60:.1f} minutes with SBX")
    elif speedup < 1.0:
        print(f"  ⚠️  SBX is {1/speedup:.2f}x SLOWER ({(1-speedup)*100:.1f}% slower)")
    else:
        print(f"  🤷 Equivalent performance")

    print("\n" + "="*80)

    # Analysis
    print("\nANALYSIS:")
    if speedup > 10:
        print(f"✅ Excellent speedup! SBX provides {speedup:.0f}x faster training.")
        print(f"   This matches the expected 10-20x speedup from JAX acceleration.")
    elif speedup > 5:
        print(f"✅ Good speedup! SBX provides {speedup:.1f}x faster training.")
        print(f"   This is substantial and will significantly reduce training time.")
    elif speedup > 1.5:
        print(f"✅ Modest speedup. SBX provides {speedup:.1f}x faster training.")
        print(f"   Worthwhile for long training runs.")
    elif speedup > 0.8:
        print(f"🤷 Similar performance. SBX and SB3 are roughly equivalent.")
    else:
        print(f"⚠️  SBX is slower in this configuration.")
        print(f"   This may be due to JIT compilation overhead on small workloads.")

    print("\nNOTE: This benchmarks RL training (including neural network updates).")
    print("      For full speedup, combine with JAX physics (Phase 3).")


if __name__ == "__main__":
    main()
