#!/usr/bin/env python3
"""
Explain and demonstrate how VecNormalize handles unknown reward ranges.
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

print("How VecNormalize handles unknown reward ranges")
print("=" * 60)

# Create a simple env with unpredictable rewards
class UnpredictableRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.step_count = 0
        self.episode = 0

    def reset(self, seed=None):
        self.step_count = 0
        self.episode += 1
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1

        # Rewards that grow over time - no fixed maximum!
        if self.episode < 10:
            reward = np.random.uniform(-10, 10)
        elif self.episode < 20:
            reward = np.random.uniform(-100, 100)
        elif self.episode < 30:
            reward = np.random.uniform(-500, 500)
        else:
            # Can even have outliers
            if np.random.random() < 0.95:
                reward = np.random.uniform(-100, 100)
            else:
                reward = np.random.choice([-10000, 10000])  # Rare huge rewards!

        done = self.step_count >= 20
        return np.random.randn(4).astype(np.float32) * 0.1, reward, done, False, {}

# Create environment
env = DummyVecEnv([UnpredictableRewardEnv])
normalized_env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

print("\n1. VecNormalize uses RUNNING statistics (Welford's algorithm):")
print("   - Starts with mean=0, std=1")
print("   - Updates with each new reward it sees")
print("   - No need to know min/max ahead of time!\n")

# Track how statistics evolve
print("2. Watch how statistics adapt as rewards change:\n")
print("Episode | Raw Reward Range | Running Mean | Running Std | Normalized Range")
print("-" * 80)

for episode in range(40):
    obs = normalized_env.reset()
    raw_rewards = []
    normalized_rewards = []

    done = False
    while not done:
        action = [env.action_space.sample()]
        obs, norm_reward, done, info = normalized_env.step(action)

        # Get the raw reward (before normalization)
        raw_reward = float(normalized_env.unnormalize_reward(norm_reward)[0])
        raw_rewards.append(raw_reward)
        normalized_rewards.append(float(norm_reward[0]))

    # Get current running statistics
    running_mean = float(normalized_env.ret_rms.mean)
    running_std = float(np.sqrt(normalized_env.ret_rms.var))

    if episode % 5 == 0 or episode < 3:
        print(f"{episode:7d} | [{min(raw_rewards):8.1f}, {max(raw_rewards):8.1f}] | "
              f"{running_mean:11.2f} | {running_std:10.2f} | "
              f"[{min(normalized_rewards):5.2f}, {max(normalized_rewards):5.2f}]")

print("\n3. Key insights:")
print("   - Early episodes: Statistics are still adapting")
print("   - Later episodes: Statistics stabilize around the data")
print("   - Outliers: Get clipped to ±10 standard deviations")
print("   - No maximum needed: Adapts to whatever it sees!")

print("\n4. The normalization formula:")
print("   normalized_reward = (reward - running_mean) / (running_std + epsilon)")
print("   - epsilon (1e-8) prevents division by zero")
print("   - Then clips to [-10, 10] to prevent extreme values")

print("\n5. Why this prevents NaN:")
print("   - Gradients are computed on normalized rewards (small, stable values)")
print("   - Original reward differences are preserved (200 is still 2x of 100)")
print("   - Outliers can't explode gradients (clipped to ±10 std)")
print("   - Adapts to any reward scale automatically!")

# Show what happens with the actual fighter rewards
print("\n6. With fighter rewards (-200 to +200 with spikes):")
print("   - Initial episodes: Might see ±5 normalized range")
print("   - After adaptation: Most rewards in ±2 range")
print("   - Rare wins/losses: Might hit ±5 to ±10")
print("   - Gradients stay stable throughout!")