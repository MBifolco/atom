#!/usr/bin/env python3
"""
Test VmapEnvWrapper observation shape to debug the (45, 9) issue.
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

print("Testing VmapEnvWrapper observation shape...")
print("=" * 60)

# Import VmapEnvWrapper
from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig

# Create config
config = WorldConfig()

# Create dummy opponent function
def dummy_opponent(snapshot):
    return np.array([0.0, 1.0])

# Create VmapEnvWrapper
print("\n1. Creating VmapEnvWrapper with 45 environments...")
env = VmapEnvWrapper(
    config=config,
    opponent_decision_func=dummy_opponent,
    n_envs=45
)

# Reset the environment
print("\n2. Resetting environment...")
result = env.reset()
if isinstance(result, tuple):
    obs = result[0]  # New gym API returns (obs, info)
else:
    obs = result

# Check observation shape
print(f"\n3. Observation shape: {obs.shape}")
print(f"   Expected shape: (45, 13)")
print(f"   Actual shape: {obs.shape}")

if obs.shape[1] != 13:
    print(f"\n⚠️  Wrong number of features! Got {obs.shape[1]} instead of 13")

    # Debug the _get_observations method
    print("\n4. Debugging observation components...")

    # Let's check what's in the observation
    print(f"   First env observation: {obs[0]}")
    print(f"   Observation values:")
    for i, val in enumerate(obs[0]):
        print(f"     [{i}]: {val:.3f}")

    # Check if episode_damage_dealt is initialized
    print(f"\n   episode_damage_dealt shape: {env.episode_damage_dealt.shape if hasattr(env, 'episode_damage_dealt') else 'NOT FOUND'}")
    print(f"   tick_counts shape: {env.tick_counts.shape if hasattr(env, 'tick_counts') else 'NOT FOUND'}")

    # Let's manually check what _get_observations returns
    print("\n5. Manually calling _get_observations...")
    obs_manual = env._get_observations()
    print(f"   Manual observation shape: {obs_manual.shape}")

    # Check the stacking operation
    print("\n6. Checking individual components...")
    fighter_pos = np.array(env.jax_states.fighter_a.position)
    print(f"   fighter_pos shape: {fighter_pos.shape}")

else:
    print("\n✅ Observation shape is correct!")

# Take a step to see if it changes
print("\n7. Taking a step...")
action = np.random.uniform(-1, 1, (45, 2))
action[:, 1] = np.random.uniform(0, 2.99, 45)  # Stance
result = env.step(action)
if len(result) == 5:
    obs2, reward, terminated, truncated, info = result
else:
    obs2, reward, done, info = result
print(f"   After step, observation shape: {obs2.shape}")

print("\n" + "=" * 60)