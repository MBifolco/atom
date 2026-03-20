#!/usr/bin/env python3
"""
Test the actual 13-dimensional observation setup to find the problem.
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

print("Testing 13-dimensional observation configuration...")
print("=" * 60)

# Check what's actually happening in VmapEnvWrapper with 13 dims
print("\n1. Checking VmapEnvWrapper with population training setup...")

from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.training.gym_env import AtomCombatEnv

config = WorldConfig()

# First, let's see what happens if we manually create 13-dim observations
print("\n2. Testing 13-dimensional observation bounds...")

# The 13-dim space should be:
obs_13_dims = [
    ("position",          0.0, 15.0),   # 0
    ("velocity",         -3.0,  3.0),   # 1
    ("hp_norm",           0.0,  1.0),   # 2
    ("stamina_norm",      0.0,  1.0),   # 3
    ("distance",          0.0, 15.0),   # 4
    ("rel_velocity",     -5.0,  5.0),   # 5
    ("opp_hp_norm",       0.0,  1.0),   # 6
    ("opp_stamina_norm",  0.0,  1.0),   # 7
    ("arena_width",       0.0, 15.0),   # 8
    ("wall_dist_left",    0.0, 15.0),   # 9
    ("wall_dist_right",   0.0, 15.0),   # 10
    ("opp_stance_int",    0.0,  2.0),   # 11
    ("recent_damage",     0.0, 100.0),  # 12
]

print("13-dimensional observation space:")
for i, (name, low, high) in enumerate(obs_13_dims):
    print(f"  [{i:2d}] {name:20s}: {low:6.1f} to {high:6.1f}")

# Create a PPO model expecting 13 dimensions
print("\n3. Creating PPO model with 13-dim observation space...")

from gymnasium import spaces

obs_space_13 = spaces.Box(
    low=np.array([d[1] for d in obs_13_dims], dtype=np.float32),
    high=np.array([d[2] for d in obs_13_dims], dtype=np.float32),
    dtype=np.float32
)

action_space = spaces.Box(
    low=np.array([-1.0, 0.0], dtype=np.float32),
    high=np.array([1.0, 2.99], dtype=np.float32),
    dtype=np.float32
)

# Create a dummy environment with 13-dim observations
class Dummy13DimEnv:
    def __init__(self):
        self.observation_space = obs_space_13
        self.action_space = action_space

    def reset(self):
        # Return a valid 13-dim observation
        obs = np.array([
            7.5,   # position (middle)
            0.0,   # velocity
            1.0,   # hp_norm (full)
            1.0,   # stamina_norm (full)
            5.0,   # distance
            0.0,   # rel_velocity
            1.0,   # opp_hp_norm (full)
            1.0,   # opp_stamina_norm (full)
            12.0,  # arena_width
            7.5,   # wall_dist_left
            4.5,   # wall_dist_right (12 - 7.5)
            0.0,   # opp_stance_int (neutral)
            0.0,   # recent_damage
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = self.reset()[0]
        return obs, 0.0, False, False, {}

# Create vectorized environment
dummy_env = Dummy13DimEnv()
vec_env = DummyVecEnv([lambda: dummy_env])

# Create PPO model
model_13 = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")

print(f"Model created with observation shape: {model_13.observation_space.shape}")

# Test predictions
print("\n4. Testing predictions with different observation values...")

test_cases = [
    ("Normal", [7.5, 0.0, 1.0, 1.0, 5.0, 0.0, 1.0, 1.0, 12.0, 7.5, 4.5, 0.0, 0.0]),
    ("Zero HP", [7.5, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0, 12.0, 7.5, 4.5, 0.0, 0.0]),
    ("Max damage", [7.5, 0.0, 0.5, 0.5, 5.0, 0.0, 0.5, 0.5, 12.0, 7.5, 4.5, 1.0, 100.0]),
    ("At wall", [0.0, 0.0, 1.0, 1.0, 12.0, 0.0, 1.0, 1.0, 12.0, 0.0, 12.0, 0.0, 0.0]),
    ("High stance", [7.5, 0.0, 1.0, 1.0, 5.0, 0.0, 1.0, 1.0, 12.0, 7.5, 4.5, 2.0, 0.0]),
]

for name, obs_values in test_cases:
    obs_test = np.array(obs_values, dtype=np.float32)
    try:
        action, _ = model_13.predict(obs_test, deterministic=True)
        print(f"✅ {name:12s}: action = {action}")
    except Exception as e:
        print(f"❌ {name:12s}: {e}")

# Now test with a batch to simulate training
print("\n5. Testing batch predictions (simulating parallel environments)...")

batch_size = 64
obs_batch = np.zeros((batch_size, 13), dtype=np.float32)

for i in range(batch_size):
    # Create varied but valid observations
    pos = np.random.uniform(0, 12)
    obs_batch[i] = [
        pos,                          # position
        np.random.uniform(-1, 1),     # velocity
        np.random.uniform(0.5, 1.0),  # hp_norm
        np.random.uniform(0.5, 1.0),  # stamina_norm
        np.random.uniform(0, 12),     # distance
        np.random.uniform(-2, 2),     # rel_velocity
        np.random.uniform(0.5, 1.0),  # opp_hp_norm
        np.random.uniform(0.5, 1.0),  # opp_stamina_norm
        12.0,                         # arena_width (constant)
        pos,                          # wall_dist_left
        12.0 - pos,                   # wall_dist_right
        np.random.randint(0, 3),      # opp_stance_int
        np.random.uniform(0, 50),     # recent_damage
    ]

# Test with the batch
import torch
obs_tensor = torch.FloatTensor(obs_batch)

try:
    with torch.no_grad():
        # Get policy
        policy = model_13.policy

        # Extract features
        features = policy.extract_features(obs_tensor)

        # Check for NaN
        if torch.isnan(features).any():
            print("⚠️  NaN detected in features!")
            nan_mask = torch.isnan(features).any(dim=1)
            nan_envs = torch.where(nan_mask)[0]
            print(f"   Environments with NaN: {nan_envs.tolist()}")
            print(f"   Problem observations:")
            for idx in nan_envs[:3]:  # Show first 3
                print(f"     Env {idx}: {obs_batch[idx]}")
        else:
            print(f"✅ Batch of {batch_size} predictions successful - no NaN")

except Exception as e:
    print(f"❌ Batch prediction failed: {e}")

# Check if the issue is with specific observation patterns
print("\n6. Testing edge cases that might cause NaN...")

edge_cases = [
    ("Zero stamina", [7.5, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 12.0, 7.5, 4.5, 0.0, 0.0]),
    ("Dead fighter", [7.5, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 1.0, 12.0, 7.5, 4.5, 0.0, 0.0]),
    ("Both dead", [7.5, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 12.0, 7.5, 4.5, 0.0, 0.0]),
    ("Max values", obs_space_13.high.tolist()),
    ("Min values", obs_space_13.low.tolist()),
]

for name, obs_values in edge_cases:
    obs_test = np.array(obs_values, dtype=np.float32)
    try:
        action, _ = model_13.predict(obs_test, deterministic=True)
        print(f"✅ {name:12s}: OK")
    except Exception as e:
        print(f"❌ {name:12s}: {e}")

print("\n" + "=" * 60)
print("Analysis complete!")