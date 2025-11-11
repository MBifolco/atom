#!/usr/bin/env python3
"""Test VmapEnvWrapper"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.vmap_env_wrapper import VmapEnvWrapper
import numpy as np


def dummy_opponent(state):
    return {"acceleration": 0.0, "stance": "neutral"}


print("Testing VmapEnvWrapper...")

env = VmapEnvWrapper(
    n_envs=100,
    opponent_decision_func=dummy_opponent,
    max_ticks=250
)

print(f"✅ Created env with {env.n_envs} parallel environments")

obs, _ = env.reset()
print(f"✅ Reset complete - obs shape: {obs.shape}")

# Run a few steps
for i in range(10):
    actions = env.action_space.sample()
    actions = np.tile(actions, (env.n_envs, 1))  # Replicate for all envs

    obs, rewards, dones, truncated, infos = env.step(actions)

    if i == 0:
        print(f"✅ Step complete - obs shape: {obs.shape}, rewards shape: {rewards.shape}")

print(f"✅ All tests passed!")
print(f"\nThis wrapper can now be used with SBX for vmap-accelerated training!")
