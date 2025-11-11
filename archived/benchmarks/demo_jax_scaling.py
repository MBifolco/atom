#!/usr/bin/env python3
"""
JAX Optimization Scaling Demonstration

Shows the performance progression through all optimization levels:
- Level 0: Baseline (SB3 + Python)
- Level 1: SBX Training
- Level 2: Multi-Environment
- Level 3: vmap Integration
- Level 4: GPU (if available)

Run this to see the full potential of JAX optimization!
"""

import sys
from pathlib import Path
import time
import argparse
import numpy as np

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.gym_env import AtomCombatEnv
from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena import WorldConfig
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sbx import PPO as SBX_PPO
import importlib.util
import jax


def load_opponent(filepath: str):
    """Load opponent decision function."""
    spec = importlib.util.spec_from_file_location("opponent", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def print_header(title: str):
    """Print a nice header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_result(level: str, time_sec: float, steps_per_sec: float, speedup: float):
    """Print benchmark result."""
    print(f"\n{'Level':<12} {'Time':<12} {'Steps/sec':<15} {'Speedup':<10}")
    print("-" * 60)
    print(f"{level:<12} {time_sec:<12.2f} {steps_per_sec:<15,.0f} {speedup:<10.2f}x")


def level_0_baseline(timesteps: int = 10000):
    """
    Level 0: Baseline (SB3 + Python Physics)

    This is the original configuration before any optimization.
    """
    print_header("Level 0: Baseline (SB3 + Python Physics)")

    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    def make_env():
        env = AtomCombatEnv(
            opponent_decision_func=opponent_func,
            config=config,
            max_ticks=250,
            use_jax=False,
            use_jax_jit=False,
            seed=42
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    print("Creating SB3 PPO model...")
    model = SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=0
    )

    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    print(f"\n✅ Baseline: {steps_per_sec:,.0f} steps/sec in {elapsed:.2f}s")

    env.close()
    return elapsed, steps_per_sec


def level_1_sbx(timesteps: int = 10000):
    """
    Level 1: SBX Training (JAX NN + Python Physics)

    Replace PyTorch with JAX for neural network operations.
    Expected speedup: 2-3x
    """
    print_header("Level 1: SBX Training (JAX NN + Python Physics)")

    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    def make_env():
        env = AtomCombatEnv(
            opponent_decision_func=opponent_func,
            config=config,
            max_ticks=250,
            use_jax=False,
            use_jax_jit=False,
            seed=42
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])

    print("Creating SBX PPO model...")
    model = SBX_PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="auto"
    )

    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    print(f"\n✅ SBX Training: {steps_per_sec:,.0f} steps/sec in {elapsed:.2f}s")

    env.close()
    return elapsed, steps_per_sec


def level_2_multi_env(n_envs: int = 16, timesteps: int = 10000):
    """
    Level 2: Multi-Environment (SBX + Parallel Envs)

    Use multiple parallel environments to maximize CPU utilization.
    Expected speedup: 2-3x additional (6-8x total vs baseline)
    """
    print_header(f"Level 2: Multi-Environment ({n_envs} parallel envs)")

    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    def make_env(seed: int):
        def _init():
            env = AtomCombatEnv(
                opponent_decision_func=opponent_func,
                config=config,
                max_ticks=250,
                use_jax=False,
                use_jax_jit=False,
                seed=seed
            )
            return Monitor(env)
        return _init

    env_fns = [make_env(42 + i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    print(f"Created {n_envs} subprocess environments")
    print("Creating SBX PPO model...")

    model = SBX_PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512 // n_envs,  # Adjust for multiple envs
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="auto"
    )

    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    print(f"\n✅ Multi-Env ({n_envs} envs): {steps_per_sec:,.0f} steps/sec in {elapsed:.2f}s")

    env.close()
    return elapsed, steps_per_sec


def level_3_vmap(n_envs: int = 100, timesteps: int = 10000):
    """
    Level 3: vmap Integration (SBX + JAX vmap Physics)

    Use JAX vmap to run 100+ episodes in parallel with JIT-compiled physics.
    Expected speedup: 2-5x additional (12-20x total vs baseline)
    """
    print_header(f"Level 3: vmap Integration ({n_envs} parallel JAX envs)")

    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")
    config = WorldConfig()

    print(f"Creating VmapEnvWrapper with {n_envs} parallel environments...")
    env = VmapEnvWrapper(
        n_envs=n_envs,
        opponent_decision_func=opponent_func,
        config=config,
        max_ticks=250,
        seed=42
    )

    # Wrap for SBX (it expects certain attributes)
    class VmapEnvAdapter:
        """Adapter to make VmapEnvWrapper compatible with SBX."""
        def __init__(self, vmap_env):
            self.vmap_env = vmap_env
            self.observation_space = vmap_env.observation_space
            self.action_space = vmap_env.action_space
            self.num_envs = vmap_env.n_envs

        def reset(self):
            obs, info = self.vmap_env.reset()
            return obs

        def step(self, actions):
            obs, rewards, dones, truncated, infos = self.vmap_env.step(actions)
            # SBX expects combined done
            dones = np.logical_or(dones, truncated)
            return obs, rewards, dones, infos

    adapted_env = VmapEnvAdapter(env)

    print("Creating SBX PPO model...")
    model = SBX_PPO(
        "MlpPolicy",
        adapted_env,
        learning_rate=3e-4,
        n_steps=512 // n_envs,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="auto"
    )

    print(f"Training for {timesteps:,} timesteps...")
    start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start

    steps_per_sec = timesteps / elapsed

    print(f"\n✅ vmap Integration ({n_envs} envs): {steps_per_sec:,.0f} steps/sec in {elapsed:.2f}s")

    return elapsed, steps_per_sec


def check_gpu():
    """Check if GPU is available."""
    devices = jax.devices()
    backend = jax.default_backend()

    has_gpu = backend in ['gpu', 'rocm', 'cuda']

    print("\nGPU Status:")
    print(f"  Devices: {devices}")
    print(f"  Backend: {backend}")
    print(f"  GPU Available: {'✅ Yes' if has_gpu else '❌ No (CPU only)'}")

    return has_gpu


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate JAX optimization scaling across all levels"
    )
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Training timesteps per level")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip Level 0 (baseline) - it's slow")
    parser.add_argument("--multi-env-count", type=int, default=16,
                       help="Number of environments for Level 2")
    parser.add_argument("--vmap-count", type=int, default=100,
                       help="Number of parallel envs for Level 3 (vmap)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick demo with fewer timesteps (5000)")

    args = parser.parse_args()

    if args.quick:
        timesteps = 5000
        print("\n🚀 Quick Demo Mode: 5,000 timesteps per level\n")
    else:
        timesteps = args.timesteps

    print("\n" + "="*80)
    print("  JAX OPTIMIZATION SCALING DEMONSTRATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Timesteps per level: {timesteps:,}")
    print(f"  Multi-env count: {args.multi_env_count}")
    print(f"  vmap count: {args.vmap_count}")

    # Check GPU
    has_gpu = check_gpu()

    results = []

    # Level 0: Baseline
    if not args.skip_baseline:
        try:
            elapsed_0, sps_0 = level_0_baseline(timesteps)
            results.append(("Level 0: Baseline (SB3)", elapsed_0, sps_0))
        except Exception as e:
            print(f"\n⚠️  Level 0 failed: {e}")
            print("Skipping baseline, will use Level 1 as reference")
            elapsed_0, sps_0 = None, None
    else:
        print("\n⏭️  Skipping Level 0 (baseline) - using Level 1 as reference")
        elapsed_0, sps_0 = None, None

    # Level 1: SBX
    try:
        elapsed_1, sps_1 = level_1_sbx(timesteps)
        results.append(("Level 1: SBX", elapsed_1, sps_1))
        baseline_time, baseline_sps = elapsed_1, sps_1
    except Exception as e:
        print(f"\n❌ Level 1 failed: {e}")
        return

    # Use Level 0 as baseline if available
    if elapsed_0 is not None:
        baseline_time, baseline_sps = elapsed_0, sps_0

    # Level 2: Multi-Environment
    try:
        elapsed_2, sps_2 = level_2_multi_env(args.multi_env_count, timesteps)
        results.append((f"Level 2: Multi-Env ({args.multi_env_count})", elapsed_2, sps_2))
    except Exception as e:
        print(f"\n⚠️  Level 2 failed: {e}")

    # Level 3: vmap
    try:
        elapsed_3, sps_3 = level_3_vmap(args.vmap_count, timesteps)
        results.append((f"Level 3: vmap ({args.vmap_count})", elapsed_3, sps_3))
    except Exception as e:
        print(f"\n⚠️  Level 3 failed: {e}")
        print("This is expected - vmap integration is experimental")

    # Results Summary
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Level':<30} {'Time (s)':<12} {'Steps/sec':<15} {'Speedup':<10}")
    print("-" * 80)

    for level_name, elapsed, sps in results:
        speedup = baseline_sps / sps if baseline_sps else 1.0
        speedup_vs_time = baseline_time / elapsed if baseline_time else 1.0
        print(f"{level_name:<30} {elapsed:<12.2f} {sps:<15,.0f} {speedup_vs_time:<10.2f}x")

    # Analysis
    print("\n" + "="*80)
    print("  ANALYSIS")
    print("="*80)

    if len(results) >= 2:
        best = max(results, key=lambda x: x[2])
        print(f"\n🏆 Best Configuration: {best[0]}")
        print(f"   Throughput: {best[2]:,.0f} steps/sec")
        print(f"   Speedup: {baseline_time / best[1]:.2f}x over baseline")
        print(f"   Time saved: {(1 - best[1]/baseline_time)*100:.0f}%")

        # Projections
        print(f"\n📊 Training Time Projections:")
        print(f"   1M timesteps:")
        print(f"     Baseline: {baseline_time * 1_000_000 / timesteps / 3600:.1f} hours")
        print(f"     Best:     {best[1] * 1_000_000 / timesteps / 3600:.1f} hours")
        print(f"     Saved:    {(baseline_time - best[1]) * 1_000_000 / timesteps / 3600:.1f} hours")

    # Recommendations
    print("\n" + "="*80)
    print("  RECOMMENDATIONS")
    print("="*80)

    print("\n✅ Production Ready:")
    print("   - Level 1 (SBX): 2-3x speedup, zero risk")
    print("   - Level 2 (Multi-Env): Additional 2-3x, minimal risk")

    print("\n🔬 Experimental:")
    print("   - Level 3 (vmap): Additional 2-5x, needs testing")
    print("   - Level 4 (GPU): 10-100x total, high setup cost")

    if not has_gpu:
        print("\n💡 GPU Acceleration:")
        print("   JAX GPU not detected. See docs/GPU_SETUP_GUIDE.md")
        print("   Potential speedup: 10-100x with ROCm + JAX")

    print("\n📚 For more details:")
    print("   - docs/JAX_OPTIMIZATION_ROADMAP.md")
    print("   - docs/GPU_SETUP_GUIDE.md")
    print("   - docs/INTEGRATION_AND_GPU_RESULTS.md")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
