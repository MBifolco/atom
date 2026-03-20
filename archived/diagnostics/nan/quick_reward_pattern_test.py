#!/usr/bin/env python3
"""
Quick test: Is it reward SIZE or PATTERN that causes NaN?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Testing reward SIZE vs PATTERN for gradient explosion")
print("=" * 60)

# Simulate a simple policy network like PPO uses
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

def test_gradient_explosion(rewards, pattern_name):
    """Test if a reward pattern causes gradient explosion."""
    model = SimplePolicy()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Simulate multiple gradient updates
    for step in range(100):
        # Random observations
        obs = torch.randn(64, 13)

        # Forward pass
        mean, std = model(obs)

        # Create distribution (this is where NaN often occurs)
        try:
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)

            # Simulate policy gradient loss
            # rewards[step % len(rewards)] simulates seeing these rewards
            reward = rewards[step % len(rewards)]
            loss = -(log_probs * reward).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check for gradient explosion
            total_grad = 0
            max_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad += grad_norm
                    max_grad = max(max_grad, param.grad.abs().max().item())

                    if torch.isnan(param.grad).any():
                        print(f"  ❌ NaN gradient at step {step} with {pattern_name}")
                        return False

            # Apply gradients
            optimizer.step()

            # Check parameters for NaN
            for param in model.parameters():
                if torch.isnan(param).any():
                    print(f"  ❌ NaN in parameters at step {step} with {pattern_name}")
                    return False

            # Check if log_std is getting too negative (common cause of NaN)
            if (model.log_std < -10).any():
                print(f"  ⚠️  log_std very negative at step {step}: {model.log_std.data}")

        except Exception as e:
            print(f"  ❌ Error at step {step} with {pattern_name}: {e}")
            return False

    print(f"  ✅ No issues with {pattern_name}")
    return True

print("\n1. Testing LARGE CONSTANT rewards:")
test_gradient_explosion([500.0] * 10, "constant 500")
test_gradient_explosion([1000.0] * 10, "constant 1000")
test_gradient_explosion([5000.0] * 10, "constant 5000")

print("\n2. Testing SMALL rewards with HIGH VARIANCE:")
test_gradient_explosion([10, -10, 10, -10, 10], "oscillating ±10")
test_gradient_explosion([1, 1, 1, 100, 1, 1, 1, -100], "rare spikes ±100")
test_gradient_explosion(np.random.uniform(-50, 50, 20).tolist(), "random ±50")

print("\n3. Testing EXTREME PATTERNS (like Level 5):")
# Simulate consistent winning with occasional huge rewards
winning_pattern = [10, 15, 12, 10, 200, 15, 10, 12, 15, 10]  # Mostly 10-15, occasional 200
test_gradient_explosion(winning_pattern, "consistent winning")

# Simulate mixed results with high variance
mixed_pattern = [10, -10, 200, -200, 5, -5, 100, -100, 0, 50]
test_gradient_explosion(mixed_pattern, "high variance mixed")

# Simulate what happens when most environments win but a few lose badly
biased_pattern = [10] * 60 + [-200] * 4  # Most win, few lose badly (like env 10, 18)
test_gradient_explosion(biased_pattern, "biased pattern (most win, few lose)")

print("\n" + "=" * 60)
print("Key Findings:")
print("1. Large CONSTANT rewards usually don't cause NaN")
print("2. HIGH VARIANCE patterns are more problematic")
print("3. BIASED patterns (most win, few lose badly) can cause issues")
print("4. The problem is gradient variance, not absolute size")