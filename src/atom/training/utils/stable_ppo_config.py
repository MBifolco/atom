"""
Stable PPO configuration to prevent NaN during training.
"""

import torch.nn as nn


def get_shared_policy_kwargs():
    """Policy architecture shared by curriculum and population training.

    Only controls the network shape — training hyperparameters (LR, batch size,
    etc.) are set separately by each training path.
    """
    return {
        "activation_fn": nn.ReLU,
        "net_arch": [256, 256],
        "ortho_init": True,
        "log_std_init": -0.5,
    }


def get_stable_ppo_config():
    """Get stable PPO hyperparameters for curriculum training."""
    return {
        "learning_rate": 3e-5,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "tensorboard_log": None,
        "policy_kwargs": get_shared_policy_kwargs(),
    }
