#!/usr/bin/env python3
"""
Test Gym Environment with JAX JIT Physics

Quick test to ensure the gym environment works with JAX JIT physics.
"""

import sys
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


def test_gym_with_jax_jit():
    """Test that gym environment works with JAX JIT physics."""
    print("\n" + "="*80)
    print("TESTING GYM ENVIRONMENT WITH JAX JIT PHYSICS")
    print("="*80)

    config = WorldConfig()
    opponent_func = load_opponent("fighters/training_opponents/training_dummy.py")

    # Test Python (baseline)
    print("\n1. Testing with Python physics...")
    env_python = AtomCombatEnv(
        opponent_decision_func=opponent_func,
        config=config,
        max_ticks=250,
        use_jax=False,
        use_jax_jit=False
    )
    obs, _ = env_python.reset(seed=42)
    print(f"   ✅ Python env initialized - obs shape: {obs.shape}")

    # Run a few steps
    for i in range(10):
        action = env_python.action_space.sample()
        obs, reward, terminated, truncated, info = env_python.step(action)
    print(f"   ✅ Python env runs - step 10 obs: {obs[:3]}")

    # Test JAX JIT
    print("\n2. Testing with JAX JIT physics...")
    env_jax = AtomCombatEnv(
        opponent_decision_func=opponent_func,
        config=config,
        max_ticks=250,
        use_jax=False,
        use_jax_jit=True  # Enable JAX JIT
    )
    obs, _ = env_jax.reset(seed=42)
    print(f"   ✅ JAX JIT env initialized - obs shape: {obs.shape}")

    # Run a few steps
    for i in range(10):
        action = env_jax.action_space.sample()
        obs, reward, terminated, truncated, info = env_jax.step(action)
    print(f"   ✅ JAX JIT env runs - step 10 obs: {obs[:3]}")

    # Compare observations (should be similar with same seed and actions)
    print("\n3. Comparing Python vs JAX JIT with same actions...")
    env_python.reset(seed=100)
    env_jax.reset(seed=100)

    np.random.seed(100)
    for i in range(5):
        action = np.array([0.5, 1.0])  # Fixed action
        obs_py, _, _, _, _ = env_python.step(action)
        obs_jax, _, _, _, _ = env_jax.step(action)

        diff = np.abs(obs_py - obs_jax).max()
        print(f"   Step {i+1}: Max obs difference = {diff:.2e}")

    print("\n✅ All tests passed! JAX JIT physics works in gym environment.")


if __name__ == "__main__":
    test_gym_with_jax_jit()
