#!/usr/bin/env python3
"""
Test that the NaN fixes work properly:
1. Rewards are clipped
2. Model can train without NaN
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

print("Testing NaN prevention fixes...")
print("=" * 60)

# Test 1: Verify reward clipping works
print("\n1. Testing reward clipping in gym_env...")

from src.training.gym_env import AtomCombatEnv
from src.arena.world_config import WorldConfig

config = WorldConfig()
env = AtomCombatEnv(
    opponent_decision_func=lambda s: {"acceleration": 0.0, "stance": "neutral"},
    config=config
)

obs, _ = env.reset()
max_reward = -float('inf')
min_reward = float('inf')

# Run some steps and check rewards
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    max_reward = max(max_reward, reward)
    min_reward = min(min_reward, reward)

    if done or truncated:
        obs, _ = env.reset()

print(f"  Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
if max_reward <= 10.0 and min_reward >= -10.0:
    print("  ✅ Rewards are properly clipped!")
else:
    print(f"  ⚠️ Rewards exceed clipping range!")

# Test 2: Verify VmapEnvWrapper reward clipping
print("\n2. Testing reward clipping in VmapEnvWrapper...")

from src.training.vmap_env_wrapper import VmapEnvWrapper

vmap_env = VmapEnvWrapper(
    n_envs=64,
    opponent_decision_func=lambda s: np.array([0.0, 1.0]),  # VmapEnvWrapper uses array format
    config=config
)

obs, _ = vmap_env.reset()
max_reward = -float('inf')
min_reward = float('inf')

for _ in range(100):
    actions = np.random.randn(64, 2).astype(np.float32)
    actions[:, 0] = np.clip(actions[:, 0], -1, 1)
    actions[:, 1] = np.clip(actions[:, 1], 0, 2.99)

    obs, rewards, dones, truncated, info = vmap_env.step(actions)

    max_reward = max(max_reward, rewards.max())
    min_reward = min(min_reward, rewards.min())

    # Check for NaN
    if np.isnan(rewards).any():
        print("  ❌ NaN detected in rewards!")
        break

print(f"  Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
if max_reward <= 10.0 and min_reward >= -10.0:
    print("  ✅ VmapEnvWrapper rewards are properly clipped!")
else:
    print(f"  ⚠️ VmapEnvWrapper rewards exceed clipping range!")

# Test 3: Quick PPO training test
print("\n3. Testing PPO training stability...")

from stable_baselines3 import PPO
from src.training.utils.stable_ppo_config import get_stable_ppo_config

# Create a small model with stable config
stable_config = get_stable_ppo_config()
stable_config['device'] = 'cpu'  # Use CPU for test

from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", vec_env, **stable_config, verbose=0)

# Train for a few steps and check for NaN
print("  Training for 1000 steps...")
try:
    model.learn(total_timesteps=1000, progress_bar=False)
    print("  ✅ Training completed without NaN!")

    # Check model parameters for NaN
    import torch
    has_nan = False
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ NaN found in parameter: {name}")
            has_nan = True

    if not has_nan:
        print("  ✅ All model parameters are valid!")

except ValueError as e:
    if "nan" in str(e).lower():
        print(f"  ❌ NaN error during training: {e}")
    else:
        print(f"  ❌ Training error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("\nSummary:")
print("- Reward clipping prevents extreme values")
print("- Stable hyperparameters reduce gradient explosion risk")
print("- Together these fixes should prevent NaN errors during training")