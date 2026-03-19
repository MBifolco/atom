#!/usr/bin/env python3
"""
Test VmapEnvWrapper with comprehensive NaN checking.
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

from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig

print("Testing NaN protection in VmapEnvWrapper...")
print("=" * 60)

# Create config
config = WorldConfig()

# Test edge cases
test_cases = [
    ("Normal mass", 75.0, 75.0),
    ("Minimum mass", config.min_mass, 75.0),
    ("Maximum mass", config.max_mass, 75.0),
    ("Very small mass", 1.0, 75.0),  # Could cause issues
    ("Very large mass", 1000.0, 75.0),  # Could cause issues
]

for name, fighter_mass, opponent_mass in test_cases:
    print(f"\nTesting: {name} (fighter={fighter_mass}, opponent={opponent_mass})")

    try:
        # Create env
        env = VmapEnvWrapper(
            n_envs=10,
            opponent_decision_func=lambda s: np.array([0.0, 1.0]),
            config=config,
            fighter_mass=fighter_mass,
            opponent_mass=opponent_mass
        )

        # Reset
        obs, _ = env.reset()

        # Check for NaN in initial observations
        if np.isnan(obs).any():
            print(f"   ❌ NaN in initial observations!")
            print(f"      NaN locations: {np.where(np.isnan(obs))}")
        else:
            print(f"   ✅ No NaN in initial observations")

        # Take a few steps
        for step in range(3):
            action = np.random.uniform(-1, 1, (10, 2))
            action[:, 1] = np.random.uniform(0, 2.99, 10)

            obs, reward, done, truncated, info = env.step(action)

            if np.isnan(obs).any():
                print(f"   ❌ NaN in observations at step {step}")
            elif np.isnan(reward).any():
                print(f"   ❌ NaN in rewards at step {step}")
            else:
                print(f"   ✅ Step {step} OK")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
