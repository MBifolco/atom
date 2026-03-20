#!/usr/bin/env python3
"""
Test the population model creation to understand observation shape issues.
"""

# Set GPU environment variables BEFORE any imports
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Testing population model observation spaces...")
print("=" * 60)

# Import required modules
from src.training.gym_env import AtomCombatEnv
from src.arena.world_config import WorldConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Create config
config = WorldConfig()

# Test 1: Check AtomCombatEnv observation space
print("\n1. AtomCombatEnv observation space:")
env = AtomCombatEnv(
    opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
    config=config,
    max_ticks=500,
    fighter_mass=75.0,
    opponent_mass=75.0
)
print(f"   Shape: {env.observation_space.shape}")
print(f"   Low: {env.observation_space.low}")
print(f"   High: {env.observation_space.high}")

# Test 2: Create a PPO model and check its observation space
print("\n2. Creating PPO model with AtomCombatEnv...")
vec_env = DummyVecEnv([lambda: Monitor(env)])
model = PPO("MlpPolicy", vec_env, verbose=0)
print(f"   Model observation space: {model.observation_space}")
print(f"   Model expects shape: {model.observation_space.shape}")

# Test 3: Try to predict with different observation shapes
print("\n3. Testing predictions with different observation shapes...")

# Test with correct 13-dim observation
obs_13 = np.random.randn(13).astype(np.float32)
try:
    action, _ = model.predict(obs_13, deterministic=True)
    print(f"   ✅ 13-dim observation works: action = {action}")
except Exception as e:
    print(f"   ❌ 13-dim observation failed: {e}")

# Test with incorrect 9-dim observation
obs_9 = np.random.randn(9).astype(np.float32)
try:
    action, _ = model.predict(obs_9, deterministic=True)
    print(f"   ✅ 9-dim observation works: action = {action}")
except Exception as e:
    print(f"   ❌ 9-dim observation failed: {e}")

# Test 4: Check VmapEnvWrapper with opponent models
print("\n4. Testing VmapEnvWrapper with PPO opponent models...")
from src.training.vmap_env_wrapper import VmapEnvWrapper

# Create multiple PPO models as opponents
opponent_models = []
for i in range(3):
    opp_env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
        opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
        config=config,
        fighter_mass=75.0,
        opponent_mass=75.0
    ))])
    opp_model = PPO("MlpPolicy", opp_env, verbose=0)
    opponent_models.append(opp_model)

# Create VmapEnvWrapper with opponent models
vmap_env = VmapEnvWrapper(
    n_envs=45,
    opponent_models=opponent_models,
    config=config,
    max_ticks=500,
    fighter_mass=75.0,
    opponent_mass=75.0
)

print(f"   VmapEnvWrapper observation space: {vmap_env.observation_space}")
print(f"   Expected shape: {vmap_env.observation_space.shape}")

# Reset and check actual observation shape
obs, _ = vmap_env.reset()
print(f"   Actual observation shape after reset: {obs.shape}")

# Test 5: Check if opponent models are getting wrong observations
print("\n5. Checking opponent model predictions in VmapEnvWrapper...")

# Take a step to trigger opponent predictions
action = np.random.uniform(-1, 1, (45, 2))
action[:, 1] = np.random.uniform(0, 2.99, 45)

try:
    obs2, reward, terminated, truncated, info = vmap_env.step(action)
    print(f"   ✅ Step succeeded, observation shape: {obs2.shape}")
except Exception as e:
    print(f"   ❌ Step failed with error: {type(e).__name__}: {str(e)[:200]}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Analysis complete!")

# Clean up
del model
del vmap_env
import jax
jax.clear_caches()