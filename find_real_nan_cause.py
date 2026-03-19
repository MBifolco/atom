#!/usr/bin/env python3
"""
Find the REAL cause of NaN in specific environments during training.
The error shows NaN in specific environments (10, 18) out of 64, not a dimension issue.
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Finding the REAL cause of NaN in specific environments...")
print("=" * 60)

# The error showed:
# - Shape was correct (64, 2)
# - But specific rows (10, 18) had NaN values
# - This happened DURING training, not immediately

print("\n1. Simulating what happens during population training...")

from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig
from stable_baselines3 import PPO

config = WorldConfig()

# Create VmapEnvWrapper with population training setup
print("\nCreating environment with 64 parallel envs (like in error)...")

# Simulate population training with opponent models
dummy_opponents = []
for i in range(8):  # 8 opponents like population training
    class DummyModel:
        def predict(self, obs, deterministic=False):
            # Return random actions
            if len(obs.shape) == 1:
                return np.array([np.random.uniform(-1, 1), np.random.randint(0, 3)]), None
            else:
                batch_size = obs.shape[0]
                return np.column_stack([
                    np.random.uniform(-1, 1, batch_size),
                    np.random.randint(0, 3, batch_size)
                ]), None
    dummy_opponents.append(DummyModel())

vmap_env = VmapEnvWrapper(
    n_envs=64,
    opponent_models=dummy_opponents,
    config=config
)

print(f"Environment created with observation space: {vmap_env.observation_space.shape}")

# Reset and run for many steps to see if NaN appears
print("\n2. Running many steps to find when NaN appears...")

obs, _ = vmap_env.reset()
print(f"Initial observation shape: {obs.shape}")

# Check initial observations
if np.isnan(obs).any():
    print("⚠️ NaN in initial observations!")
    nan_envs = np.where(np.isnan(obs).any(axis=1))[0]
    print(f"   Environments with NaN: {nan_envs}")

# Run many steps
nan_found = False
for step in range(1000):
    # Random actions
    actions = np.random.randn(64, 2).astype(np.float32)
    actions[:, 0] = np.clip(actions[:, 0], -1, 1)
    actions[:, 1] = np.clip(actions[:, 1], 0, 2.99)

    obs, rewards, dones, truncated, info = vmap_env.step(actions)

    # Check for NaN in observations
    if np.isnan(obs).any():
        print(f"\n⚠️ NaN detected at step {step}!")
        nan_envs = np.where(np.isnan(obs).any(axis=1))[0]
        print(f"   Environments with NaN: {nan_envs}")
        print(f"   First NaN env observations: {obs[nan_envs[0]]}")

        # Check what's special about these environments
        for env_idx in nan_envs[:3]:
            print(f"\n   Environment {env_idx}:")
            print(f"     Observation: {obs[env_idx]}")
            print(f"     Reward: {rewards[env_idx]}")
            print(f"     Done: {dones[env_idx]}")
            print(f"     Truncated: {truncated[env_idx]}")

        nan_found = True
        break

    # Check for extreme values that might lead to NaN
    if step % 100 == 0:
        max_val = np.abs(obs).max()
        if max_val > 1000:
            print(f"   Step {step}: Large values detected (max: {max_val})")
            large_envs = np.where(np.abs(obs).max(axis=1) > 1000)[0]
            print(f"     Environments with large values: {large_envs}")

if not nan_found:
    print("✅ No NaN found in 1000 steps")

# Check for specific patterns that cause NaN
print("\n3. Checking for specific patterns that might cause NaN...")

# The error message showed tensor values before NaN appeared
# Let's check if certain value combinations cause issues

# Check reward scaling
print("\nReward statistics:")
print(f"  Mean: {rewards.mean():.2f}")
print(f"  Std: {rewards.std():.2f}")
print(f"  Min: {rewards.min():.2f}")
print(f"  Max: {rewards.max():.2f}")

# Check if the issue is with specific observation values
print("\nObservation statistics (last step):")
for i in range(obs.shape[1]):
    col = obs[:, i]
    print(f"  Dim {i:2d}: mean={col.mean():7.2f}, std={col.std():7.2f}, "
          f"min={col.min():7.2f}, max={col.max():7.2f}")

# Check if it's related to episode resets
print("\n4. Checking episode resets...")
reset_envs = np.where(dones | truncated)[0]
if len(reset_envs) > 0:
    print(f"Environments that reset: {reset_envs}")

    # Force a reset and check
    obs, _ = vmap_env.reset()
    if np.isnan(obs).any():
        print("⚠️ NaN after reset!")
    else:
        print("✅ No NaN after reset")

# The real issue might be in the PPO model itself
print("\n5. Checking if it's a model issue...")

# Load the checkpoint that's causing issues
checkpoint_path = project_root / "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip"
if checkpoint_path.exists():
    print(f"\nLoading problematic checkpoint...")
    model = PPO.load(checkpoint_path, device="cpu")

    # Check model parameters for NaN
    import torch
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️ NaN in model parameter: {name}")
        if torch.isinf(param).any():
            print(f"⚠️ Inf in model parameter: {name}")

        # Check for very large values
        max_val = param.abs().max().item()
        if max_val > 100:
            print(f"⚠️ Large values in {name}: max={max_val}")

    # Test prediction with actual observations
    print("\nTesting model predictions...")
    for i in range(5):
        test_obs = obs[i]
        try:
            action, _ = model.predict(test_obs, deterministic=True)
            print(f"  Env {i}: ✅ Prediction OK")
        except Exception as e:
            print(f"  Env {i}: ❌ Prediction failed: {e}")

print("\n" + "=" * 60)
print("Analysis complete!")