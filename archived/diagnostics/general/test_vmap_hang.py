#!/usr/bin/env python3
"""
Simple test to diagnose vmap environment hanging issue.
"""

import sys
import time
from pathlib import Path

# Force output to be unbuffered
sys.stdout = sys.__class__(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = sys.__class__(sys.stderr.fileno(), mode='w', buffering=1)

print("=" * 80)
print("VMAP ENVIRONMENT HANG DIAGNOSTIC TEST")
print("=" * 80)
print(flush=True)

# Test 1: Import JAX
print("Test 1: Importing JAX...", flush=True)
start = time.time()
try:
    import jax
    import jax.numpy as jnp
    print(f"✅ JAX imported successfully in {time.time() - start:.2f}s", flush=True)
    print(f"   JAX version: {jax.__version__}", flush=True)
    print(f"   Default backend: {jax.default_backend()}", flush=True)
    print(f"   Available devices: {jax.devices()}", flush=True)
except Exception as e:
    print(f"❌ Failed to import JAX: {e}", flush=True)
    sys.exit(1)

# Test 2: Simple JAX compilation
print("\nTest 2: Simple JAX JIT compilation...", flush=True)
start = time.time()
try:
    @jax.jit
    def simple_func(x):
        return x * 2 + 1

    result = simple_func(jnp.array([1.0, 2.0, 3.0]))
    print(f"✅ JIT compilation successful in {time.time() - start:.2f}s", flush=True)
    print(f"   Result: {result}", flush=True)
except Exception as e:
    print(f"❌ JIT compilation failed: {e}", flush=True)

# Test 3: Import arena components
print("\nTest 3: Importing arena components...", flush=True)
start = time.time()
try:
    from src.arena import WorldConfig
    from src.arena.fighter import FighterState
    print(f"✅ Arena components imported in {time.time() - start:.2f}s", flush=True)
except Exception as e:
    print(f"❌ Failed to import arena: {e}", flush=True)
    sys.exit(1)

# Test 4: Import vmap wrapper
print("\nTest 4: Importing VmapEnvWrapper...", flush=True)
start = time.time()
try:
    from src.training.vmap_env_wrapper import VmapEnvWrapper
    print(f"✅ VmapEnvWrapper imported in {time.time() - start:.2f}s", flush=True)
except Exception as e:
    print(f"❌ Failed to import VmapEnvWrapper: {e}", flush=True)
    sys.exit(1)

# Test 5: Create small vmap environment
print("\nTest 5: Creating small vmap environment (10 envs)...", flush=True)
start = time.time()
try:
    test_dummy_dir = Path("fighters/test_dummies")
    opponent_paths = [
        str(test_dummy_dir / "atomic/stationary_neutral.py"),
        str(test_dummy_dir / "atomic/stationary_extended.py"),
    ]

    print(f"   Using opponents: {[Path(p).stem for p in opponent_paths]}", flush=True)
    print("   Creating environment...", flush=True)

    env = VmapEnvWrapper(
        n_envs=10,  # Small number for testing
        opponent_paths=opponent_paths,
        config=WorldConfig(),
        max_ticks=100,
        fighter_mass=70.0,
        opponent_mass=70.0,
        seed=42,
        debug=True  # Enable debug output
    )

    print(f"✅ Small vmap environment created in {time.time() - start:.2f}s", flush=True)
except Exception as e:
    print(f"❌ Failed to create vmap environment: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Reset environment
print("\nTest 6: Resetting environment...", flush=True)
start = time.time()
try:
    obs, info = env.reset()
    print(f"✅ Environment reset successful in {time.time() - start:.2f}s", flush=True)
    print(f"   Observation shape: {obs.shape}", flush=True)
except Exception as e:
    print(f"❌ Failed to reset environment: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Single step
print("\nTest 7: Taking a single step...", flush=True)
start = time.time()
try:
    import numpy as np
    actions = np.random.uniform(-1, 1, size=(10, 2)).astype(np.float32)
    actions[:, 1] = np.random.uniform(0, 2.99, size=10)  # Stance selector

    obs, rewards, dones, truncated, infos = env.step(actions)
    print(f"✅ Step successful in {time.time() - start:.2f}s", flush=True)
    print(f"   Rewards: min={rewards.min():.2f}, max={rewards.max():.2f}, mean={rewards.mean():.2f}", flush=True)
except Exception as e:
    print(f"❌ Failed to step environment: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Create large vmap environment (250 envs)
print("\nTest 8: Creating large vmap environment (250 envs)...", flush=True)
print("⚠️  This is where training typically hangs", flush=True)
start = time.time()
try:
    opponent_paths = [
        str(test_dummy_dir / "atomic/stationary_neutral.py"),
        str(test_dummy_dir / "atomic/stationary_extended.py"),
        str(test_dummy_dir / "atomic/stationary_defending.py"),
    ]

    print(f"   Using opponents: {[Path(p).stem for p in opponent_paths]}", flush=True)
    print("   Creating environment with 250 parallel envs...", flush=True)

    env_large = VmapEnvWrapper(
        n_envs=250,  # Full size
        opponent_paths=opponent_paths,
        config=WorldConfig(),
        max_ticks=250,
        fighter_mass=70.0,
        opponent_mass=70.0,
        seed=42,
        debug=True
    )

    print(f"✅ Large vmap environment created in {time.time() - start:.2f}s", flush=True)

    # Try to reset it
    print("   Resetting large environment...", flush=True)
    reset_start = time.time()
    obs, info = env_large.reset()
    print(f"✅ Large environment reset in {time.time() - reset_start:.2f}s", flush=True)

except Exception as e:
    print(f"❌ Failed with large environment: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nNo hanging detected. The issue might be:", flush=True)
print("1. Specific to the training loop integration", flush=True)
print("2. Related to memory allocation patterns", flush=True)
print("3. Caused by interaction with other components", flush=True)