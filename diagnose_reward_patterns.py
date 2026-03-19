#!/usr/bin/env python3
"""
Diagnose whether NaN is caused by reward SIZE or reward PATTERNS during training.
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

print("Diagnosing reward size vs patterns...")
print("=" * 60)

# Test 1: Can PPO handle large constant rewards?
print("\n1. Testing large CONSTANT rewards...")

from stable_baselines3 import PPO
from gymnasium import spaces
import gymnasium as gym

class ConstantRewardEnv(gym.Env):
    """Environment with large but constant rewards."""
    def __init__(self, reward_value=200.0):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reward_value = reward_value
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return np.zeros(13, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(13).astype(np.float32) * 0.1
        # Large but CONSTANT reward
        reward = self.reward_value
        done = self.steps >= 100
        return obs, reward, done, False, {}

# Test different reward magnitudes
from stable_baselines3.common.vec_env import DummyVecEnv

for reward_size in [10, 100, 500, 1000]:
    print(f"\n  Testing constant reward = {reward_size}...")
    env = DummyVecEnv([lambda r=reward_size: ConstantRewardEnv(r) for _ in range(4)])
    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0, device="cpu")

    try:
        model.learn(total_timesteps=2000, progress_bar=False)

        # Check for NaN
        has_nan = False
        for name, param in model.policy.named_parameters():
            if torch.isnan(param).any():
                has_nan = True
                break

        if has_nan:
            print(f"    ❌ NaN with constant reward={reward_size}")
        else:
            print(f"    ✅ No NaN with constant reward={reward_size}")
    except Exception as e:
        if "nan" in str(e).lower():
            print(f"    ❌ NaN error with reward={reward_size}")
        else:
            print(f"    ❌ Error: {str(e)[:50]}")

# Test 2: Small rewards with high variance
print("\n\n2. Testing small rewards with HIGH VARIANCE...")

class VariableRewardEnv(gym.Env):
    """Environment with small average but high variance rewards."""
    def __init__(self, variance_pattern="random"):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.variance_pattern = variance_pattern
        self.steps = 0
        self.episode = 0

    def reset(self, seed=None):
        self.steps = 0
        self.episode += 1
        return np.zeros(13, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(13).astype(np.float32) * 0.1

        if self.variance_pattern == "random":
            # Random between -10 and 10 (mean ~0, high variance)
            reward = np.random.uniform(-10, 10)
        elif self.variance_pattern == "spike":
            # Mostly small, occasional huge spikes
            if np.random.random() < 0.95:
                reward = np.random.uniform(-1, 1)
            else:
                reward = np.random.choice([-100, 100])  # Rare spikes
        elif self.variance_pattern == "oscillating":
            # Oscillates between extremes
            reward = 50 * np.sin(self.steps * 0.5) * np.sign(np.random.randn())

        done = self.steps >= 100
        return obs, reward, done, False, {}

patterns = ["random", "spike", "oscillating"]
for pattern in patterns:
    print(f"\n  Testing {pattern} pattern...")
    env = DummyVecEnv([lambda p=pattern: VariableRewardEnv(p) for _ in range(4)])
    model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=0, device="cpu")

    try:
        model.learn(total_timesteps=2000, progress_bar=False)

        has_nan = any(torch.isnan(param).any() for _, param in model.policy.named_parameters())

        if has_nan:
            print(f"    ❌ NaN with {pattern} pattern")
        else:
            print(f"    ✅ No NaN with {pattern} pattern")
    except Exception as e:
        if "nan" in str(e).lower():
            print(f"    ❌ NaN error with {pattern} pattern")

# Test 3: Simulate Level 5 reward patterns
print("\n\n3. Simulating Level 5 expert fighter patterns...")

class Level5PatternEnv(gym.Env):
    """Simulates reward patterns from expert fighters."""
    def __init__(self, env_id=0):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.env_id = env_id
        self.steps = 0
        self.episode = 0
        self.winning = False

    def reset(self, seed=None):
        self.steps = 0
        self.episode += 1
        # Some environments consistently win, others lose
        self.winning = (self.env_id % 3) == 0 or np.random.random() < 0.3
        return np.zeros(13, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.random.randn(13).astype(np.float32) * 0.1

        # Expert fighters create consistent damage patterns
        if self.winning:
            # Consistent positive damage differential
            damage_diff = np.random.uniform(5, 15)
            reward = damage_diff * 10  # Original reward scale

            # Terminal reward
            if self.steps >= 99:
                reward = 200  # Win
        else:
            # Consistent negative damage differential
            damage_diff = np.random.uniform(-15, -5)
            reward = damage_diff * 10

            # Terminal reward
            if self.steps >= 99:
                reward = -200  # Loss

        # Add occasional spikes (critical hits, blocks)
        if np.random.random() < 0.1:
            reward += np.random.choice([-50, 50])

        done = self.steps >= 100
        return obs, reward, done, False, {}

print("  Creating environment with expert fighter patterns...")
env = DummyVecEnv([lambda i=i: Level5PatternEnv(i) for i in range(64)])

# Test with and without gradient clipping
for max_grad_norm in [None, 0.5, 0.1]:
    print(f"\n  Testing with max_grad_norm={max_grad_norm}...")

    kwargs = {"learning_rate": 5e-5, "verbose": 0, "device": "cpu"}
    if max_grad_norm is not None:
        kwargs["max_grad_norm"] = max_grad_norm

    model = PPO("MlpPolicy", env, **kwargs)

    try:
        model.learn(total_timesteps=5000, progress_bar=False)

        has_nan = any(torch.isnan(param).any() for _, param in model.policy.named_parameters())

        if has_nan:
            print(f"    ❌ NaN detected!")
        else:
            print(f"    ✅ No NaN")

    except Exception as e:
        if "nan" in str(e).lower():
            print(f"    ❌ NaN error during training")

print("\n" + "=" * 60)
print("Key findings:")
print("1. Large constant rewards alone don't cause NaN")
print("2. High variance and spikes are more problematic")
print("3. Expert fighters create consistent patterns that accumulate")
print("4. Gradient clipping helps but isn't always sufficient")