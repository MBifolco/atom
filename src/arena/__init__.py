"""
Atom Combat - Arena

The source of truth for physics, collisions, and damage.
"""

from .arena_1d import Arena1D
from .world_config import WorldConfig, StanceConfig
from .fighter import FighterState

__all__ = ['Arena1D', 'WorldConfig', 'StanceConfig', 'FighterState']
