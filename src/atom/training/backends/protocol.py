"""Training backend protocol — the interface all RL backends must satisfy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes what a training backend supports."""

    name: str  # e.g. "sb3_ppo", "sbx_sac"
    framework: str  # "pytorch" or "jax"
    on_policy: bool  # True = rollout-based (PPO), False = replay buffer (SAC)
    flush_mode: str  # "rollout_boundary" or "step_interval"


@runtime_checkable
class TrainingBackend(Protocol):
    """Interface for pluggable RL training algorithms.

    Curriculum and population trainers code against this protocol.
    Each backend owns its training hyperparameters internally —
    only network architecture is shared via get_policy_arch().
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend metadata (name, framework, on/off-policy, flush mode)."""
        ...

    def create_model(self, envs: Any, seed: int) -> Any:
        """Create a new untrained model for the given environment."""
        ...

    def load_model(self, path: str, envs: Any | None = None) -> Any:
        """Load a trained model from disk."""
        ...

    def save_model(self, model: Any, path: str) -> None:
        """Save a model to disk."""
        ...

    def predict(self, model: Any, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from observation."""
        ...

    def learn(
        self,
        model: Any,
        total_timesteps: int,
        callback: Any | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Any:
        """Train the model. Returns the model (may be same object or new)."""
        ...

    def replace_env(self, model: Any, envs: Any) -> None:
        """Replace the model's environment.

        Called on level transitions and opponent pool refresh. The backend
        handles any internal fixups (e.g. resetting cached observations).
        """
        ...

    def get_policy_arch(self) -> dict:
        """Return the shared network architecture config.

        Only architecture (net_arch, activation). Training hyperparameters
        (LR, batch size, buffer size) are backend-internal.
        """
        ...

    def reduce_learning_rate(self, model: Any, factor: float) -> None:
        """Reduce learning rate by factor (for NaN recovery)."""
        ...

    def clone_and_mutate(self, model: Any, envs: Any, mutation_rate: float) -> Any:
        """Clone a model and apply random weight mutation.

        Used by population evolution. Returns a new model instance.
        """
        ...

    def create_dummy_env(self, config: Any, max_ticks: int, fighter_mass: float) -> Any:
        """Create a minimal env for model loading/cloning."""
        ...

    def handle_distribution_shift(self, model: Any, kind: str) -> None:
        """Called when the environment distribution changes.

        kind: "level_transition" or "pool_refresh"

        On-policy backends (PPO): no-op (no replay buffer to clear).
        Off-policy backends (SAC): clear replay buffer on level transition,
        keep buffer on pool refresh.
        """
        ...
