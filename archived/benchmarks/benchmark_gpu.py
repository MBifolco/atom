#!/usr/bin/env python3
"""
GPU Benchmark: Test JAX ROCm GPU acceleration
Compares CPU vs GPU performance for physics and training
"""

import sys
from pathlib import Path
import time
import os

# Ensure GPU environment is set
if 'HSA_OVERRIDE_GFX_VERSION' not in os.environ:
    print("❌ GPU environment not configured!")
    print("Run: source setup_gpu.sh")
    sys.exit(1)

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from src.arena.arena_1d_jax_jit import Arena1DJAXJit
from src.arena import WorldConfig
import importlib.util


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


print_header("GPU BENCHMARK: JAX ROCm Acceleration Test")

# Verify GPU
print(f"\n🎮 GPU Status:")
print(f"   JAX version: {jax.__version__}")
print(f"   Devices: {jax.devices()}")
print(f"   Backend: {jax.default_backend()}")

if jax.default_backend() != 'gpu':
    print("\n⚠️  WARNING: GPU not detected, running on CPU")
    print("   Results will not show GPU acceleration")
else:
    print(f"   ✅ GPU detected: {jax.devices()[0]}")

# =============================================================================
# Test 1: Matrix Multiplication (GPU warmup)
# =============================================================================

print_header("Test 1: Matrix Multiplication (GPU Warmup)")

size = 4096
print(f"Computing 10 x {size}x{size} matrix multiplications...")

a = jnp.ones((size, size))
b = jnp.ones((size, size))

# Warm up
_ = jnp.dot(a, b).block_until_ready()

# Benchmark
start = time.time()
for _ in range(10):
    c = jnp.dot(a, b).block_until_ready()
elapsed = time.time() - start

print(f"\n✅ Completed in {elapsed:.3f}s")
print(f"   Average: {elapsed/10*1000:.1f}ms per operation")
print(f"   (Expected: ~30ms on GPU, ~500ms on CPU)")

# =============================================================================
# Test 2: Setup for vmap tests
# =============================================================================

print_header("Test 2: Setup")

from src.arena.arena_1d_jax_jit import FighterStateJAX

config = WorldConfig()

# Create fighter states
fighter_a = FighterStateJAX(
    mass=70.0,
    position=0.0,
    velocity=0.0,
    hp=100.0,
    max_hp=100.0,
    stamina=100.0,
    max_stamina=100.0,
    stance=0  # NEUTRAL
)

fighter_b = FighterStateJAX(
    mass=75.0,
    position=10.0,
    velocity=0.0,
    hp=100.0,
    max_hp=100.0,
    stamina=100.0,
    max_stamina=100.0,
    stance=0  # NEUTRAL
)

print("\n✅ Fighter states created")
print(f"   Fighter A: {fighter_a.mass}kg at position {fighter_a.position}")
print(f"   Fighter B: {fighter_b.mass}kg at position {fighter_b.position}")

# For comparison with CPU baseline
tps_single = 10065  # From previous CPU benchmarks (JAX JIT single episode)

# =============================================================================
# Test 3: vmap Parallelization - Batch Processing
# =============================================================================

print_header("Test 3: vmap Batch Processing (GPU Parallelization)")

from src.arena.arena_1d_jax_jit import create_stance_arrays

# Test different batch sizes
batch_sizes = [50, 100, 250, 500]

print("\nBatch Size | Throughput (tps) | Speedup vs Single | GPU Utilization")
print("-" * 75)

baseline_tps = tps_single

for batch_size in batch_sizes:
    # Create initial states
    states = []
    for _ in range(batch_size):
        arena_temp = Arena1DJAXJit(fighter_a, fighter_b, config, seed=42)
        states.append(arena_temp.state)

    # Stack states for vmap
    import jax.tree_util as tree_util
    batched_state = tree_util.tree_map(lambda *args: jnp.stack(args), *states)

    # Create vmap'd step function
    stance_reach, stance_defense, stance_drain = create_stance_arrays(config)

    def single_step(state):
        action_a = {"acceleration": 0.0, "stance": 0}
        action_b = {"acceleration": 0.0, "stance": 0}
        new_state, _ = Arena1DJAXJit._jax_step_jit(
            state, action_a, action_b,
            config.dt, config.max_acceleration, config.max_velocity,
            config.friction, config.arena_width,
            config.stamina_accel_cost, config.stamina_base_regen,
            config.stamina_neutral_bonus,
            stance_reach, stance_defense, stance_drain
        )
        return new_state

    batched_step = jit(vmap(single_step))

    # Warm up
    _ = batched_step(batched_state)

    # Benchmark
    num_steps = 250
    start = time.time()
    current_state = batched_state
    for _ in range(num_steps):
        current_state = batched_step(current_state)
    # Block until complete
    jax.block_until_ready(current_state)
    elapsed = time.time() - start

    total_ticks = batch_size * num_steps
    tps = total_ticks / elapsed
    speedup_vs_single = tps / baseline_tps
    gpu_util = (speedup_vs_single / batch_size) * 100

    print(f"{batch_size:>10} | {tps:>16,.0f} | {speedup_vs_single:>17.2f}x | {gpu_util:>14.1f}%")

# =============================================================================
# Test 4: Memory Stress Test
# =============================================================================

print_header("Test 4: GPU Memory Stress Test")

print("\nTesting maximum batch size before OOM...")

test_sizes = [500, 1000, 2000, 4000]
max_working_batch = 0

for test_batch in test_sizes:
    try:
        print(f"\nTrying batch size: {test_batch}...", end=" ")

        # Create states
        states = []
        for _ in range(test_batch):
            arena_temp = Arena1DJAXJit(fighter_a, fighter_b, config, seed=42)
            states.append(arena_temp.state)

        batched_state = tree_util.tree_map(lambda *args: jnp.stack(args), *states)

        # Try one step
        _ = batched_step(batched_state)
        jax.block_until_ready(_)

        max_working_batch = test_batch
        print("✅ Success")

    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}")
        break

print(f"\n✅ Maximum working batch size: {max_working_batch}")

# =============================================================================
# Summary
# =============================================================================

print_header("BENCHMARK SUMMARY")

print(f"""
Hardware:
  GPU: {jax.devices()[0]}
  Backend: {jax.default_backend()}

Performance Results:
  Single Episode: {tps_single:,.0f} ticks/sec
  Best vmap (batch={batch_sizes[-1]}): {tps:,.0f} ticks/sec
  GPU Speedup: {speedup_vs_single:.2f}x
  Max Batch Size: {max_working_batch}

Comparison to CPU Baseline (from previous benchmarks):
  Python Physics: ~57,107 tps (single episode)
  JAX CPU vmap (500): ~122,947 tps
  JAX GPU vmap (500): {tps:,.0f} tps

GPU vs CPU Speedup: {tps / 122947:.2f}x faster than CPU vmap

Next Steps:
  1. Test with real training (SBX + GPU)
  2. Benchmark full curriculum training
  3. Compare wall-clock training time vs CPU
""")

print("="*80)
