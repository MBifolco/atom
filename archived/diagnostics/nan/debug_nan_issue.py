#!/usr/bin/env python3
"""
Debug the actual NaN issue - find out WHY certain environments produce NaN.
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Debugging NaN issue with 13-dimensional observations...")
print("=" * 60)

# First, let's understand what the 13 dimensions are supposed to be
print("\n1. Checking observation space bounds...")

from src.training.gym_env import AtomCombatEnv
from src.arena.world_config import WorldConfig

config = WorldConfig()

# Create a dummy environment to check observation space
env = AtomCombatEnv(
    opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
    config=config
)

obs_space = env.observation_space
print(f"Observation space shape: {obs_space.shape}")
print(f"Low bounds:  {obs_space.low}")
print(f"High bounds: {obs_space.high}")

# Now let's check what each dimension represents
obs_dims = [
    "position",          # 0: 0-15
    "velocity",          # 1: -3 to 3
    "hp_norm",          # 2: 0-1
    "stamina_norm",     # 3: 0-1
    "distance",         # 4: 0-15
    "rel_velocity",     # 5: -5 to 5
    "opp_hp_norm",      # 6: 0-1
    "opp_stamina_norm", # 7: 0-1
    "arena_width"       # 8: 0-15 (should be constant 12 actually)
]

if len(obs_space.low) > 9:
    obs_dims.extend([
        "wall_dist_left",    # 9: 0-15
        "wall_dist_right",   # 10: 0-15
        "opp_stance_int",    # 11: 0-2
        "recent_damage"      # 12: 0-100
    ])

print("\nObservation dimensions:")
for i, (name, low, high) in enumerate(zip(obs_dims, obs_space.low, obs_space.high)):
    print(f"  [{i:2d}] {name:20s}: {low:6.1f} to {high:6.1f}")

# Test with extreme values
print("\n2. Testing edge cases that might cause NaN...")

# Reset environment
obs, _ = env.reset()
print(f"\nInitial observation shape: {obs.shape}")
print(f"Initial observation: {obs}")

# Check for any issues with the observation
if np.isnan(obs).any():
    print("⚠️  NaN detected in initial observation!")
    print(f"   NaN indices: {np.where(np.isnan(obs))}")

if np.isinf(obs).any():
    print("⚠️  Inf detected in initial observation!")
    print(f"   Inf indices: {np.where(np.isinf(obs))}")

# Now test with PPO to see if we can reproduce the NaN
print("\n3. Testing with PPO model...")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create vectorized environment
vec_env = DummyVecEnv([lambda: env])

# Create PPO model
model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")

# Test prediction with various observations
print("\n4. Testing model predictions with edge cases...")

# Test normal observation
try:
    obs_test = np.random.randn(len(obs_dims)).astype(np.float32)
    # Clip to valid range
    obs_test = np.clip(obs_test, obs_space.low, obs_space.high)

    action, _ = model.predict(obs_test, deterministic=True)
    print(f"✅ Normal observation works: action = {action}")
except Exception as e:
    print(f"❌ Normal observation failed: {e}")

# Test with zero observation
try:
    obs_zero = np.zeros(len(obs_dims), dtype=np.float32)
    action, _ = model.predict(obs_zero, deterministic=True)
    print(f"✅ Zero observation works: action = {action}")
except Exception as e:
    print(f"❌ Zero observation failed: {e}")

# Test with maximum values
try:
    obs_max = obs_space.high.copy()
    action, _ = model.predict(obs_max, deterministic=True)
    print(f"✅ Max observation works: action = {action}")
except Exception as e:
    print(f"❌ Max observation failed: {e}")

# Now let's check what happens during training
print("\n5. Checking model internals...")

# Get the policy network
policy = model.policy

# Check weight statistics
print("\nModel weight statistics:")
for name, param in policy.named_parameters():
    if param.requires_grad:
        data = param.data
        print(f"  {name:30s}: mean={data.mean().item():7.4f}, std={data.std().item():7.4f}, "
              f"min={data.min().item():7.4f}, max={data.max().item():7.4f}")

        if torch.isnan(data).any():
            print(f"    ⚠️  NaN detected in {name}!")
        if torch.isinf(data).any():
            print(f"    ⚠️  Inf detected in {name}!")

# Test batch prediction to simulate training
print("\n6. Testing batch prediction (simulating training)...")

batch_size = 64
obs_batch = np.random.randn(batch_size, len(obs_dims)).astype(np.float32)

# Important: Make sure observations are within bounds
for i in range(batch_size):
    obs_batch[i] = np.clip(obs_batch[i], obs_space.low, obs_space.high)

# Convert to torch tensor
obs_tensor = torch.FloatTensor(obs_batch)

# Forward pass through policy
try:
    with torch.no_grad():
        features = policy.extract_features(obs_tensor)
        latent_pi, latent_vf = policy.mlp_extractor(features)

        # Check for NaN in intermediate values
        if torch.isnan(features).any():
            print("⚠️  NaN in features!")
            nan_envs = torch.where(torch.isnan(features).any(dim=1))[0]
            print(f"   Environments with NaN: {nan_envs.tolist()}")

        if torch.isnan(latent_pi).any():
            print("⚠️  NaN in policy latent!")
            nan_envs = torch.where(torch.isnan(latent_pi).any(dim=1))[0]
            print(f"   Environments with NaN: {nan_envs.tolist()}")

        # Get action distribution
        mean_actions = policy.action_net(latent_pi)

        if torch.isnan(mean_actions).any():
            print("⚠️  NaN in action output!")
            nan_envs = torch.where(torch.isnan(mean_actions).any(dim=1))[0]
            print(f"   Environments with NaN: {nan_envs.tolist()}")
            print(f"   Problem observations:")
            for env_idx in nan_envs:
                print(f"     Env {env_idx}: {obs_batch[env_idx]}")
        else:
            print("✅ Batch prediction successful - no NaN detected")

except Exception as e:
    print(f"❌ Batch prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Analysis complete!")