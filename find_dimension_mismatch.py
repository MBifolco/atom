#!/usr/bin/env python3
"""
Find the dimension mismatch issue - when a model trained on one dimension gets another.
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

print("Finding dimension mismatch issue...")
print("=" * 60)

# The key insight: Check what dimensions the SAVED models expect
print("\n1. Checking saved model dimensions...")

checkpoint_paths = [
    "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip",
    "outputs/progressive_20251125_183751/curriculum/models/level_0.zip",
    "outputs/progressive_20251125_183751/curriculum/models/level_1.zip",
]

for path in checkpoint_paths:
    full_path = project_root / path
    if full_path.exists():
        print(f"\nChecking: {path}")

        from stable_baselines3 import PPO

        try:
            # Load model without environment
            model = PPO.load(full_path, device="cpu")

            # Check observation space
            obs_space = model.observation_space
            print(f"  Observation space shape: {obs_space.shape}")
            print(f"  Observation dims: {obs_space.shape[0]}")

            # Check the input layer of the neural network
            policy = model.policy

            # Get the first layer to see input dimensions
            if hasattr(policy, 'mlp_extractor'):
                if hasattr(policy.mlp_extractor, 'policy_net'):
                    first_layer = policy.mlp_extractor.policy_net[0]
                    if hasattr(first_layer, 'in_features'):
                        print(f"  Neural network input size: {first_layer.in_features}")

            # Test what happens with wrong dimensions
            print(f"  Testing with different input dimensions...")

            # Test with 9 dimensions
            obs_9 = np.random.randn(9).astype(np.float32)
            try:
                action_9, _ = model.predict(obs_9, deterministic=True)
                print(f"    ✅ 9-dim input works")
            except Exception as e:
                print(f"    ❌ 9-dim input failed: {str(e)[:50]}...")

            # Test with 13 dimensions
            obs_13 = np.random.randn(13).astype(np.float32)
            try:
                action_13, _ = model.predict(obs_13, deterministic=True)
                print(f"    ✅ 13-dim input works")
            except Exception as e:
                print(f"    ❌ 13-dim input failed: {str(e)[:50]}...")

        except Exception as e:
            print(f"  Could not load: {e}")
    else:
        print(f"\n{path} not found")

# Now check VmapEnvWrapper's actual output
print("\n2. Checking VmapEnvWrapper actual output dimensions...")

from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig

config = WorldConfig()

# Create VmapEnvWrapper
vmap_env = VmapEnvWrapper(
    n_envs=64,  # Same as error message showed
    opponent_decision_func=lambda s: np.array([0.0, 1.0]),
    config=config
)

print(f"VmapEnvWrapper observation space: {vmap_env.observation_space.shape}")

# Reset and check actual output
obs, _ = vmap_env.reset()
print(f"Actual observation shape: {obs.shape}")
print(f"First environment obs: {obs[0]}")

# Check if there's a mismatch between what the wrapper says and what it produces
expected_dims = vmap_env.observation_space.shape[0]
actual_dims = obs.shape[1]

if expected_dims != actual_dims:
    print(f"\n⚠️  DIMENSION MISMATCH DETECTED!")
    print(f"  Observation space claims: {expected_dims} dimensions")
    print(f"  Actually produces: {actual_dims} dimensions")
else:
    print(f"\n✅ Dimensions match: {expected_dims}")

# The REAL test: What happens when curriculum trainer loads a model
print("\n3. Simulating what happens during curriculum training...")

# Curriculum trainer does this:
# 1. Creates environment with certain dimensions
# 2. Loads a checkpoint that might have different dimensions
# 3. Continues training

if (project_root / "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip").exists():
    print("\nSimulating curriculum trainer loading checkpoint...")

    # Load the saved model
    saved_model = PPO.load(
        project_root / "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip",
        device="cpu"
    )

    saved_dims = saved_model.observation_space.shape[0]
    current_dims = obs.shape[1]

    print(f"  Saved model expects: {saved_dims} dimensions")
    print(f"  Current env produces: {current_dims} dimensions")

    if saved_dims != current_dims:
        print(f"\n⚠️  CRITICAL MISMATCH!")
        print(f"  This will cause NaN errors when the model tries to process observations!")

        # Test what happens
        print(f"\n  Testing mismatched prediction...")
        try:
            # Use actual observation from environment
            test_obs = obs[0]  # First environment's observation
            action, _ = saved_model.predict(test_obs, deterministic=True)
            print(f"    Somehow worked?? Action: {action}")
        except Exception as e:
            print(f"    Failed as expected: {e}")
    else:
        print(f"\n✅ Dimensions match - no issue here")

print("\n" + "=" * 60)
print("Root cause analysis:")
print("1. Check if saved models expect different dimensions than current env")
print("2. Check if VmapEnvWrapper observation_space matches actual output")
print("3. Check if the issue happens when loading checkpoints")