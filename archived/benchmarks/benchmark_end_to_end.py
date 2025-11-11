#!/usr/bin/env python3
"""
End-to-End Training Benchmark: SBX + JAX Physics

Compares:
1. SB3 (PyTorch) + Python physics (baseline)
2. SBX (JAX) + Python physics (Phase 2)
3. SBX (JAX) + JAX JIT physics (Phase 2 + Phase 3)

This shows the TOTAL combined speedup.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.gym_env import AtomCombatEnv
from src.arena import WorldConfig
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import importlib.util


def load_opponent(filepath: str):
    """Load opponent decision function."""
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def benchmark_training(use_sbx: bool, use_jax_jit: bool, timesteps: int = 50000):
    """
    Benchmark training with different configurations.

    Args:
        use_sbx: Use SBX (JAX training) vs SB3 (PyTorch)
        use_jax_jit: Use JAX JIT physics vs Python physics
        timesteps: Number of training timesteps

    Returns:
        elapsed_time, steps_per_sec
    """
    # Import appropriate PPO
    if use_sbx:
        from sbx import PPO
        label = "SBX (JAX)"
    else:
        from stable_baselines3 import PPO
        label = "SB3 (PyTorch)"

    physics_label = "JAX JIT" if use_jax_jit else "Python"

    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {label} + {physics_label} Physics")
    print("="*80)
    print(f"Timesteps: {timesteps:,}")

    # Load opponent
    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    # Create environment
    def make_env():
        env = AtomCombatEnv(
            opponent_decision_func=opponent_func,
            config=config,
            max_ticks=250,
            use_jax_jit=use_jax_jit,
            seed=42
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    # Create model
    device = "cpu" if not use_sbx else "auto"
    print(f"Creating model (device: {device})...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device=device
    )

    # Train
    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    print(f"✅ Completed in {elapsed:.2f}s")
    print(f"   Throughput: {steps_per_sec:,.0f} steps/sec")

    env.close()

    return elapsed, steps_per_sec


def main():
    print("\n" + "="*80)
    print("END-TO-END TRAINING SPEEDUP BENCHMARK")
    print("="*80)
    print("\nConfiguration:")
    print("  Timesteps: 50,000")
    print("  Opponent: Training Dummy")
    print("  Environment: 1 (DummyVecEnv)")

    TIMESTEPS = 50000

    # Benchmark 1: SB3 + Python (baseline)
    print("\n" + "-"*80)
    print("BASELINE: SB3 (PyTorch) + Python Physics")
    print("-"*80)
    sb3_py_time, sb3_py_sps = benchmark_training(
        use_sbx=False,
        use_jax_jit=False,
        timesteps=TIMESTEPS
    )

    # Benchmark 2: SBX + Python (Phase 2 only)
    print("\n" + "-"*80)
    print("PHASE 2: SBX (JAX) + Python Physics")
    print("-"*80)
    sbx_py_time, sbx_py_sps = benchmark_training(
        use_sbx=True,
        use_jax_jit=False,
        timesteps=TIMESTEPS
    )

    # Benchmark 3: SBX + JAX JIT (Phase 2 + Phase 3)
    print("\n" + "-"*80)
    print("PHASE 2+3: SBX (JAX) + JAX JIT Physics")
    print("-"*80)
    sbx_jax_time, sbx_jax_sps = benchmark_training(
        use_sbx=True,
        use_jax_jit=True,
        timesteps=TIMESTEPS
    )

    # Results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nSB3 + Python (baseline):")
    print(f"  Time: {sb3_py_time:.2f}s")
    print(f"  Throughput: {sb3_py_sps:,.0f} steps/sec")

    print(f"\nSBX + Python (Phase 2):")
    print(f"  Time: {sbx_py_time:.2f}s")
    print(f"  Throughput: {sbx_py_sps:,.0f} steps/sec")
    print(f"  Speedup: {sb3_py_time / sbx_py_time:.2f}x")

    print(f"\nSBX + JAX JIT (Phase 2+3):")
    print(f"  Time: {sbx_jax_time:.2f}s")
    print(f"  Throughput: {sbx_jax_sps:,.0f} steps/sec")
    print(f"  Speedup: {sb3_py_time / sbx_jax_time:.2f}x")

    # Analysis
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS")
    print("="*80)

    phase2_speedup = sb3_py_time / sbx_py_time
    phase23_speedup = sb3_py_time / sbx_jax_time
    jax_physics_contribution = sbx_py_time / sbx_jax_time

    print(f"\nPhase 2 (SBX only):        {phase2_speedup:.2f}x faster")
    print(f"Phase 2+3 (SBX + JAX JIT): {phase23_speedup:.2f}x faster")
    print(f"JAX Physics contribution:  {jax_physics_contribution:.2f}x")

    print(f"\nCombined Achievement:")
    if phase23_speedup >= 8:
        print(f"  🚀 EXCELLENT! {phase23_speedup:.1f}x total speedup achieved!")
    elif phase23_speedup >= 5:
        print(f"  ✅ GREAT! {phase23_speedup:.1f}x total speedup")
    elif phase23_speedup >= 3:
        print(f"  ✅ GOOD! {phase23_speedup:.1f}x total speedup")
    else:
        print(f"  🤷 {phase23_speedup:.1f}x speedup")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
