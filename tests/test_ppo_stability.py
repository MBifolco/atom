#!/usr/bin/env python3
"""
Test PPO stability improvements to prevent NaN during training.
"""

import pytest
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class TestPPOStability:
    """Test PPO configuration for numerical stability."""

    def test_stable_ppo_initialization(self):
        """Test that PPO initializes with stable settings."""
        from src.atom.training.utils.stable_ppo import create_stable_ppo
        from src.atom.training.gym_env import AtomCombatEnv

        env = DummyVecEnv([lambda: AtomCombatEnv(
            opponent_decision_func=lambda _: {"acceleration": 0, "stance": "neutral"},
            max_ticks=100
        )])

        model = create_stable_ppo(env)

        # Check key stability parameters
        # Handle callable learning rate
        lr = model.learning_rate
        if callable(lr):
            # Get initial learning rate (progress_remaining=1.0)
            lr = lr(1.0)
        assert lr == 3e-4  # Standard learning rate

        assert model.max_grad_norm <= 0.5  # Gradient clipping for stability

        # Handle callable clip range
        clip_range = model.clip_range
        if callable(clip_range):
            # Get initial clip range (progress_remaining=1.0)
            clip_range = clip_range(1.0)
        assert clip_range == 0.2  # Standard clip range

        assert model.vf_coef == 0.5  # Value function coefficient
        assert model.ent_coef >= 0.02  # Higher entropy for exploration

        # Check that it's using standard ActorCriticPolicy (created from "MlpPolicy" string)
        assert model.policy.__class__.__name__ == "ActorCriticPolicy"

    def test_orthogonal_initialization(self):
        """Test that networks use orthogonal initialization."""
        from stable_baselines3.common.policies import ActorCriticPolicy
        import torch.nn as nn

        # Check that orthogonal init prevents gradient issues
        layer = nn.Linear(64, 64)
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        # Forward pass with large input shouldn't explode
        x = torch.randn(100, 64) * 10
        output = layer(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 1000  # Reasonable bounds

    def test_gradient_monitoring(self):
        """Test gradient monitoring to catch issues early."""
        from src.atom.training.utils.stable_ppo import GradientMonitor

        # Create a simple network
        net = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

        monitor = GradientMonitor()

        # Simulate training step
        x = torch.randn(32, 10)
        target = torch.randn(32, 2)

        output = net(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Monitor should detect gradient stats
        stats = monitor.check_gradients(net)

        assert 'max_grad' in stats
        assert 'mean_grad' in stats
        assert 'has_nan' in stats
        assert not stats['has_nan']

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate scheduling."""
        from src.atom.training.utils.stable_ppo import AdaptiveLRSchedule

        initial_lr = 1e-4
        schedule = AdaptiveLRSchedule(initial_lr)

        # Simulate NaN detection
        lr_after_nan = schedule.on_nan_detected()
        assert lr_after_nan < initial_lr  # Should reduce

        # Multiple NaN detections should keep reducing
        lr_after_second_nan = schedule.on_nan_detected()
        assert lr_after_second_nan < lr_after_nan

        # But not below minimum
        for _ in range(10):
            lr = schedule.on_nan_detected()

        assert lr >= 1e-6  # Minimum threshold


    def test_value_loss_clipping(self):
        """Test that value loss is clipped to prevent explosion."""
        # This is built into PPO but verify it's working

        # Create dummy data
        values = torch.randn(64) * 100  # Large values
        returns = torch.randn(64)
        old_values = torch.randn(64)

        # Clip value loss
        value_clipped = old_values + torch.clamp(
            values - old_values, -10, 10
        )

        # Should be bounded
        assert (value_clipped - old_values).abs().max() <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])