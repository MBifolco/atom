"""
Training algorithms for Atom Combat.

Available trainers:
- ppo: Proximal Policy Optimization (default, best for curriculum)
- sac: Soft Actor-Critic (better exploration, more sample efficient)
"""

from .ppo.trainer import train_fighter as train_fighter_ppo
from .ppo.trainer import train_curriculum as train_curriculum_ppo
from .sac.trainer import train_fighter as train_fighter_sac
from .sac.trainer import train_curriculum as train_curriculum_sac

__all__ = [
    'train_fighter_ppo',
    'train_curriculum_ppo',
    'train_fighter_sac',
    'train_curriculum_sac',
]
