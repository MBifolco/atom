#!/usr/bin/env python3
"""
Diagnose why NaN appears specifically at Level 5 of curriculum training.
Focus on numerical stability issues that accumulate over time.
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

print("Diagnosing Level 5 NaN issue...")
print("=" * 60)

# The error shows NaN in specific environments (10, 18) during PPO training
# This suggests the issue is with specific observation/action combinations

print("\n1. Checking for numerical instabilities in PPO...")

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution

# Load the checkpoint from just before Level 5
checkpoint_path = project_root / "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip"
if checkpoint_path.exists():
    model = PPO.load(checkpoint_path, device="cpu")

    # Check the log_std parameter which controls action distribution
    log_std = model.policy.log_std
    print(f"Current log_std: {log_std}")
    print(f"  Actual std: {torch.exp(log_std)}")

    # Check if log_std is too small (can cause numerical issues)
    if (log_std < -10).any():
        print("  ⚠️ WARNING: log_std very negative - can cause underflow!")

    # Check gradient clipping settings
    print(f"\nGradient clipping: {model.max_grad_norm}")

    # Check learning rate
    print(f"Learning rate: {model.learning_rate}")
    if model.learning_rate > 1e-3:
        print("  ⚠️ WARNING: High learning rate can cause instability!")

print("\n2. Simulating Level 5 training conditions...")

# Level 5 uses expert fighters - let's check if they produce extreme observations
from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena.world_config import WorldConfig
import importlib.util

# Load one of the Level 5 fighters
fighter_path = project_root / "fighters/examples/swarmer.py"
spec = importlib.util.spec_from_file_location("swarmer", fighter_path)
swarmer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swarmer_module)

# Create environment with Level 5-like conditions
config = WorldConfig()

# Mock the expert fighters as opponents
class ExpertOpponent:
    def __init__(self, decide_func):
        self.decide_func = decide_func

    def predict(self, obs, deterministic=False):
        # Convert observation to state dict (simplified)
        # This is where numerical issues might occur
        if len(obs.shape) == 1:
            # Single observation
            state = {
                "you": {
                    "position": float(obs[6] if obs[0] < 6 else obs[0]),
                    "velocity": float(obs[7] if len(obs) > 7 else 0),
                    "hp": float(obs[2] * 100),
                    "max_hp": 100,
                    "stamina": float(obs[3] * 100),
                    "max_stamina": 100
                },
                "opponent": {
                    "position": float(obs[0] if obs[0] < 6 else 12 - obs[0]),
                    "distance": float(obs[4]),
                    "direction": 1 if obs[5] > 0 else -1,
                    "hp": float(obs[6] * 100) if len(obs) > 6 else 50,
                    "stance": "neutral"
                }
            }

            try:
                decision = self.decide_func(state)
                action = np.array([
                    float(decision.get("acceleration", 0)),
                    0 if decision.get("stance") == "neutral" else
                    1 if decision.get("stance") == "extended" else 2
                ], dtype=np.float32)

                # Check for NaN in expert's decision
                if np.isnan(action).any():
                    print(f"  ⚠️ Expert fighter produced NaN action!")
                    print(f"     State: {state}")
                    print(f"     Decision: {decision}")

            except Exception as e:
                print(f"  ⚠️ Expert fighter error: {e}")
                action = np.array([0.0, 0.0], dtype=np.float32)

            return action, None
        else:
            # Batch - simplified
            batch_size = obs.shape[0]
            return np.zeros((batch_size, 2), dtype=np.float32), None

# Create opponents like Level 5
expert_opponents = [ExpertOpponent(swarmer_module.decide) for _ in range(8)]

vmap_env = VmapEnvWrapper(
    n_envs=64,
    opponent_models=expert_opponents,
    config=config
)

print(f"\nEnvironment setup with {len(expert_opponents)} expert opponents")

# Run some steps and check for extreme values
obs, _ = vmap_env.reset()
print(f"Initial observation range: [{obs.min():.2f}, {obs.max():.2f}]")

# Track value ranges over steps
value_history = {
    'obs_max': [],
    'obs_min': [],
    'reward_max': [],
    'reward_min': []
}

print("\n3. Running steps to detect value explosion...")
for step in range(100):
    # Use the PPO model's actions
    with torch.no_grad():
        actions, _ = model.predict(obs, deterministic=False)

    obs, rewards, dones, truncated, info = vmap_env.step(actions)

    # Track extremes
    value_history['obs_max'].append(np.abs(obs).max())
    value_history['obs_min'].append(np.abs(obs).min())
    value_history['reward_max'].append(rewards.max())
    value_history['reward_min'].append(rewards.min())

    # Check for concerning patterns
    if np.abs(obs).max() > 100:
        print(f"  Step {step}: Large observation value: {np.abs(obs).max():.2f}")

    if np.isnan(obs).any():
        print(f"  Step {step}: NaN detected in observations!")
        nan_envs = np.where(np.isnan(obs).any(axis=1))[0]
        print(f"    Affected environments: {nan_envs}")
        break

# Analyze trends
print("\n4. Analyzing value trends...")
obs_max_trend = np.array(value_history['obs_max'])
if len(obs_max_trend) > 10:
    growth_rate = (obs_max_trend[-1] / obs_max_trend[0]) if obs_max_trend[0] != 0 else 0
    print(f"Observation max growth rate: {growth_rate:.2f}x")
    if growth_rate > 10:
        print("  ⚠️ WARNING: Rapid value growth detected!")

print(f"Final observation range: [{obs.min():.2f}, {obs.max():.2f}]")
print(f"Reward range: [{min(value_history['reward_min']):.2f}, {max(value_history['reward_max']):.2f}]")

print("\n5. Checking PPO's numerical stability with these values...")

# Test if PPO can handle the observations
with torch.no_grad():
    test_batch = torch.from_numpy(obs).float()

    # Extract features
    features = model.policy.extract_features(test_batch)
    if torch.isnan(features).any():
        print("  ⚠️ NaN in feature extraction!")

    # Get latent representations
    latent_pi, latent_vf = model.policy.mlp_extractor(features)
    if torch.isnan(latent_pi).any():
        print("  ⚠️ NaN in policy latent!")
    if torch.isnan(latent_vf).any():
        print("  ⚠️ NaN in value latent!")

    # Get mean actions
    mean_actions = model.policy.action_net(latent_pi)
    if torch.isnan(mean_actions).any():
        print("  ⚠️ NaN in mean actions!")
        nan_rows = torch.where(torch.isnan(mean_actions).any(dim=1))[0]
        print(f"    Affected rows: {nan_rows.tolist()}")

    # Try to create distribution (this is where the error occurs)
    try:
        std = torch.exp(model.policy.log_std)
        dist = torch.distributions.Normal(mean_actions, std)
        print("  ✅ Distribution creation successful")
    except Exception as e:
        print(f"  ❌ Distribution creation failed: {e}")

        # Debug which specific values caused the issue
        for i in range(mean_actions.shape[0]):
            if torch.isnan(mean_actions[i]).any():
                print(f"    Row {i}: mean_actions={mean_actions[i]}, obs={obs[i]}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("\nKey findings:")
print("1. Check if log_std is too negative (causes numerical underflow)")
print("2. Check if observations grow unbounded over time")
print("3. Check if expert fighters produce extreme actions")
print("4. Check if specific observation patterns trigger NaN in the network")