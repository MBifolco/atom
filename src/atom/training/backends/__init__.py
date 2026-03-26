"""Pluggable training backends for Atom Combat.

Each backend wraps an RL algorithm (PPO, SAC, etc.) behind a common
protocol so the curriculum and population trainers don't need to know
which algorithm is running.
"""

from .protocol import BackendCapabilities, TrainingBackend
from .sb3_ppo import SB3PPOBackend
from .sbx_sac import SBXSACBackend

__all__ = ["BackendCapabilities", "TrainingBackend", "SB3PPOBackend", "SBXSACBackend"]
