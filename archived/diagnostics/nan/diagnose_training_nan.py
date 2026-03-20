#!/usr/bin/env python3
"""
Diagnose NaN during actual PPO training - the issue happens during backpropagation.
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

print("Diagnosing NaN during PPO training/backpropagation...")
print("=" * 60)

# The error occurred during:
# 1. PPO training after collecting rollouts
# 2. In the policy network during distribution creation
# 3. After MANY successful episodes

print("\n1. Checking learning rate and gradient issues...")

checkpoint_path = project_root / "outputs/progressive_20251125_183751/curriculum/models/checkpoint_0.zip"
if not checkpoint_path.exists():
    print("No checkpoint found, creating new model...")
    from stable_baselines3 import PPO
    from gymnasium import spaces

    # Create dummy env
    class DummyEnv:
        def __init__(self):
            self.observation_space = spaces.Box(
                low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
                dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 2.99], dtype=np.float32),
                dtype=np.float32
            )

    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: DummyEnv()])
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")
else:
    from stable_baselines3 import PPO
    model = PPO.load(checkpoint_path, device="cpu")

# Check current learning rate
print(f"Current learning rate: {model.learning_rate}")
if callable(model.learning_rate):
    print(f"  (It's a schedule function)")

# Check the actual optimizer learning rate
for param_group in model.policy.optimizer.param_groups:
    print(f"Optimizer LR: {param_group['lr']}")

print("\n2. Checking for gradient explosion patterns...")

# The error showed the network was outputting reasonable values EXCEPT for specific envs
# This suggests cumulative numerical instability

# Check value function scale
print("\nValue function predictions:")
test_obs = torch.randn(64, 13)  # Batch of 64 like in error
with torch.no_grad():
    values = model.policy.predict_values(test_obs)
    print(f"  Mean: {values.mean().item():.2f}")
    print(f"  Std: {values.std().item():.2f}")
    print(f"  Min: {values.min().item():.2f}")
    print(f"  Max: {values.max().item():.2f}")

print("\n3. Checking policy network stability...")

# Simulate what happens with extreme inputs
extreme_cases = [
    ("Normal", torch.randn(64, 13)),
    ("Large values", torch.randn(64, 13) * 10),
    ("Small values", torch.randn(64, 13) * 0.01),
    ("Mixed extreme", torch.cat([
        torch.randn(32, 13) * 0.01,
        torch.randn(32, 13) * 10
    ])),
]

for name, test_input in extreme_cases:
    print(f"\nTesting {name}:")
    try:
        with torch.no_grad():
            # Forward pass through policy
            features = model.policy.extract_features(test_input)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            mean_actions = model.policy.action_net(latent_pi)

            # Check for NaN/Inf
            if torch.isnan(mean_actions).any():
                nan_rows = torch.where(torch.isnan(mean_actions).any(dim=1))[0]
                print(f"  ⚠️ NaN in output! Rows: {nan_rows.tolist()}")
            elif torch.isinf(mean_actions).any():
                print(f"  ⚠️ Inf in output!")
            else:
                print(f"  ✅ No NaN/Inf")
                print(f"     Action mean range: [{mean_actions.min():.3f}, {mean_actions.max():.3f}]")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n4. Checking cumulative training effects...")

# The error happened after thousands of steps - suggests accumulation
print("\nModel weight statistics:")
for name, param in model.policy.named_parameters():
    data = param.data
    print(f"  {name}:")
    print(f"    Shape: {data.shape}")
    print(f"    Mean: {data.mean().item():.6f}, Std: {data.std().item():.6f}")
    print(f"    Min: {data.min().item():.6f}, Max: {data.max().item():.6f}")

    # Check for concerning patterns
    if data.std().item() < 1e-6:
        print(f"    ⚠️ Very low variance - might be dead neurons")
    if data.abs().max().item() > 10:
        print(f"    ⚠️ Large weights - might explode")

print("\n5. Checking specific failure pattern from error...")

# The error showed specific environments (10, 18) producing NaN
# Let's create a batch with problematic patterns

batch = torch.randn(64, 13)
# Make some environments have extreme but valid values
batch[10] = torch.tensor([12.0, 2.0, 0.1, 0.1, 10.0, -3.0, 0.1, 0.1, 12.0, 12.0, 0.0, 2.0, 50.0])
batch[18] = torch.tensor([0.0, -2.0, 0.1, 0.0, 12.0, 3.0, 0.1, 0.0, 12.0, 0.0, 12.0, 0.0, 0.0])

print("\nTesting with specific problematic patterns...")
with torch.no_grad():
    features = model.policy.extract_features(batch)
    print(f"Features - any NaN: {torch.isnan(features).any()}")

    latent_pi, latent_vf = model.policy.mlp_extractor(features)
    print(f"Latent PI - any NaN: {torch.isnan(latent_pi).any()}")
    print(f"Latent VF - any NaN: {torch.isnan(latent_vf).any()}")

    mean_actions = model.policy.action_net(latent_pi)
    print(f"Mean actions - any NaN: {torch.isnan(mean_actions).any()}")

    # Check log_std
    log_std = model.policy.log_std
    print(f"Log std: {log_std}")
    print(f"Log std - any NaN: {torch.isnan(log_std).any()}")

print("\n6. The REAL issue - checking distribution creation...")

# The error occurred in Normal distribution creation
# This happens when log_std becomes too negative or mean_actions has NaN

with torch.no_grad():
    # Get action distribution
    from torch.distributions import Normal

    mean_actions = model.policy.action_net(latent_pi)

    # This is where the error happens!
    try:
        # Create distribution like PPO does
        std = torch.exp(model.policy.log_std)
        print(f"\nStd values: {std}")
        print(f"Std range: [{std.min().item():.6f}, {std.max().item():.6f}]")

        if (std <= 0).any():
            print("⚠️ Non-positive std values!")

        distribution = Normal(mean_actions, std)
        print("✅ Distribution created successfully")

    except Exception as e:
        print(f"❌ Distribution creation failed: {e}")
        print(f"Mean actions shape: {mean_actions.shape}")
        print(f"Mean actions: {mean_actions}")
        print(f"Std: {std}")

print("\n" + "=" * 60)
print("Root cause analysis:")
print("1. NaN appears during Normal distribution creation")
print("2. Likely caused by numerical instability in policy network")
print("3. Accumulates over training until specific inputs trigger it")
print("4. Check log_std becoming too negative or weights exploding")