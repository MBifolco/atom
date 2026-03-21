"""
Stable PPO configuration and utilities to prevent NaN during training.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StableMlpPolicy(ActorCriticPolicy):
    """
    MLP policy with improved initialization for stability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only apply orthogonal init to non-output layers
        self._stable_init()

    def _stable_init(self):
        """Apply stable initialization to all layers except action output."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip the action_net output layer to avoid breaking asymmetric action spaces
                if 'action_net' not in name:
                    # Orthogonal initialization for hidden layers
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


def create_stable_ppo(
    env,
    learning_rate: float = 3e-4,  # Standard learning rate
    device: str = "cpu"
) -> PPO:
    """
    Create a PPO model with stable configuration to prevent NaN.

    Args:
        env: The environment
        learning_rate: Initial learning rate (will be reduced if needed)
        device: Device to use

    Returns:
        Configured PPO model
    """
    # Use standard MlpPolicy instead of custom policy
    # Higher entropy for better exploration of action space
    model = PPO(
        "MlpPolicy",  # Standard policy works better
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        max_grad_norm=0.5,  # Keep gradient clipping for stability
        ent_coef=0.02,  # Increased entropy for better stance exploration
        vf_coef=0.5,
        normalize_advantage=True,
        use_sde=False,
        verbose=0,
        device=device
    )

    return model


class AdaptiveLRSchedule:
    """
    Adaptive learning rate that reduces on NaN detection.
    """

    def __init__(self, initial_lr: float = 1e-4):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.nan_count = 0
        self.min_lr = 1e-6

    def on_nan_detected(self) -> float:
        """Reduce learning rate when NaN is detected."""
        self.nan_count += 1
        # Reduce by 50% each time
        self.current_lr = max(self.current_lr * 0.5, self.min_lr)
        return self.current_lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class GradientMonitor:
    """Monitor gradients for debugging NaN issues."""

    def check_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """
        Check gradient statistics.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        has_nan = False

        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())

                if torch.isnan(p.grad).any():
                    has_nan = True

        total_norm = total_norm ** 0.5

        return {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'has_nan': has_nan,
            'mean_grad': total_norm / sum(1 for _ in model.parameters())
        }


def linear_schedule(initial_value: float, final_value: float = 1e-5):
    """
    Linear learning rate schedule.

    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate

    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        Args:
            progress_remaining: Fraction of training remaining

        Returns:
            Current learning rate
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return func


def warmup_schedule(initial_value: float, warmup_steps: int = 1000):
    """
    Learning rate warmup schedule.

    Args:
        initial_value: Target learning rate after warmup
        warmup_steps: Number of warmup steps

    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Warmup then constant learning rate.

        Args:
            progress_remaining: Fraction of training remaining

        Returns:
            Current learning rate
        """
        progress = 1.0 - progress_remaining
        if progress * warmup_steps < warmup_steps:
            # During warmup
            return initial_value * (progress * warmup_steps / warmup_steps)
        else:
            # After warmup
            return initial_value

    return func