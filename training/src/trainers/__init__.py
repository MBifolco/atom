"""
Training algorithms for Atom Combat.

Available trainers:
- ppo: Proximal Policy Optimization (default, best for curriculum)
- sac: Soft Actor-Critic (better exploration, more sample efficient)
- population: Population-based training for diverse strategies
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'train_fighter_ppo',
    'train_curriculum_ppo',
    'train_fighter_sac',
    'train_curriculum_sac',
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'train_fighter_ppo':
        from .ppo.trainer import train_fighter as train_fighter_ppo
        return train_fighter_ppo
    elif name == 'train_curriculum_ppo':
        from .ppo.trainer import train_curriculum as train_curriculum_ppo
        return train_curriculum_ppo
    elif name == 'train_fighter_sac':
        from .sac.trainer import train_fighter as train_fighter_sac
        return train_fighter_sac
    elif name == 'train_curriculum_sac':
        from .sac.trainer import train_curriculum as train_curriculum_sac
        return train_curriculum_sac
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
