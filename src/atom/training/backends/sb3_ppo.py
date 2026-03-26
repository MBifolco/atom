"""SB3 PPO training backend — wraps Stable Baselines3 PPO."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .protocol import BackendCapabilities


_POLICY_ARCH = {
    "activation_fn": nn.ReLU,
    "net_arch": [256, 256],
    "ortho_init": True,
    "log_std_init": -0.5,
}

_CURRICULUM_TRAINING_CONFIG = {
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
}

_POPULATION_TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}


class SB3PPOBackend:
    """Stable Baselines 3 PPO training backend.

    Device is owned by the backend at construction time.
    Training hyperparams are backend-internal; only architecture is shared.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="sb3_ppo",
            framework="pytorch",
            on_policy=True,
            flush_mode="rollout_boundary",
        )

    def create_model(
        self,
        envs: Any,
        seed: int,
        *,
        mode: str = "curriculum",
        tensorboard_log: str | None = None,
        n_envs_per_fighter: int = 1,
        verbose: int = 0,
    ) -> Any:
        """Create a new PPO model.

        Args:
            mode: "curriculum" or "population" — selects training hyperparams.
            tensorboard_log: Optional path for TB logging.
            n_envs_per_fighter: For population mode, scales n_steps.
            verbose: SB3 verbosity level.
        """
        config = dict(_CURRICULUM_TRAINING_CONFIG) if mode == "curriculum" else dict(_POPULATION_TRAINING_CONFIG)

        if mode == "population":
            config["n_steps"] = 2048 // max(1, n_envs_per_fighter)

        model = PPO(
            "MlpPolicy",
            envs,
            device=self._device,
            seed=seed,
            policy_kwargs=dict(_POLICY_ARCH),
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **config,
        )
        return model

    def load_model(self, path: str, envs: Any | None = None) -> Any:
        kwargs: dict[str, Any] = {"device": self._device}
        if envs is not None:
            kwargs["env"] = envs
        return PPO.load(path, **kwargs)

    def save_model(self, model: Any, path: str) -> None:
        model.save(path)

    def predict(self, model: Any, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    def learn(
        self,
        model: Any,
        total_timesteps: int,
        callback: Any | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Any:
        return model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def replace_env(self, model: Any, envs: Any) -> None:
        """Replace environment and reset cached observations.

        SB3's set_env(force_reset=True) sets _last_obs=None. We immediately
        reset the new env to populate _last_obs so the next collect_rollouts
        doesn't crash.
        """
        model.set_env(envs)
        model._last_obs = envs.reset()
        model._last_episode_starts = np.ones((envs.num_envs,), dtype=bool)

    def get_policy_arch(self) -> dict:
        return dict(_POLICY_ARCH)

    def reduce_learning_rate(self, model: Any, factor: float) -> None:
        model.learning_rate *= factor

    def clone_and_mutate(
        self,
        model: Any,
        envs: Any,
        mutation_rate: float,
    ) -> Any:
        """Clone model weights into a new env and apply random mutation."""
        # Save parent to temp file, load into new env
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "parent.zip"
            model.save(tmp_path)
            new_model = PPO.load(tmp_path, env=envs, device=self._device)

        # Mutate learning rate
        new_model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))

        # Mutate weights
        with torch.no_grad():
            for param in new_model.policy.parameters():
                noise = torch.randn_like(param) * mutation_rate * 0.1
                param.data.add_(noise)

        return new_model

    def handle_distribution_shift(self, model: Any, kind: str) -> None:
        """No-op for PPO — on-policy, no replay buffer to clear."""
        pass

    def create_dummy_env(self, config: Any, max_ticks: int, fighter_mass: float) -> Any:
        """Create a minimal env for model loading/cloning in population evolution."""
        from src.atom.training.gym_env import AtomCombatEnv
        return DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            config=config,
            max_ticks=max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=70.0,
        ))])
