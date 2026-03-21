"""
Stable PPO configuration to prevent NaN during training.
"""

import torch.nn as nn

def get_stable_ppo_config():
    """Get stable PPO hyperparameters to prevent NaN."""
    return {
        "learning_rate": 3e-5,  # Reduced from 5e-5
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,  # Increased for more exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,  # Gradient clipping
        "target_kl": 0.01,  # Early stopping for PPO updates
        "tensorboard_log": None,
        "policy_kwargs": {
            "activation_fn": nn.ReLU,  # Pass the class, not a string
            "net_arch": [64, 64],
            "ortho_init": True,  # More stable initialization
            "log_std_init": -0.5,  # Start with reasonable exploration
        }
    }
