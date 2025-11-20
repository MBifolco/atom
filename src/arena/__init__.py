"""
Atom Combat - Arena

The source of truth for physics, collisions, and damage.
"""

from .arena_1d_jax_jit import Arena1DJAXJit
from .world_config import WorldConfig, StanceConfig
from .fighter import FighterState

__all__ = ['Arena1DJAXJit', 'WorldConfig', 'StanceConfig', 'FighterState']
