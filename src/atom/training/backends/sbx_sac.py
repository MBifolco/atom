"""SBX SAC training backend — wraps SBX (JAX-based) Soft Actor-Critic."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sbx import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .protocol import BackendCapabilities


_POLICY_ARCH = {
    "net_arch": [256, 256],
}

_CURRICULUM_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 300_000,
    "learning_starts": 2000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto_0.1",  # Auto-tune with floor of 0.1 (prevents entropy collapse)
    "train_freq": 4,  # Train every 4 steps (reduces overhead with 250 envs)
    "gradient_steps": 2,  # 2 updates per train call (compensates for lower train_freq)
}

_POPULATION_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto_0.1",
    "train_freq": 4,
    "gradient_steps": 2,
}


class SBXSACBackend:
    """SBX (JAX) Soft Actor-Critic training backend.

    All-JAX pipeline: physics (vmap) and policy on same device.
    Off-policy with replay buffer — gets 10-100x more learning per env step.
    """

    def __init__(self):
        # SBX/JAX handles device automatically
        pass

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="sbx_sac",
            framework="jax",
            on_policy=False,
            flush_mode="step_interval",
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
        """Create a new SAC model."""
        config = dict(_CURRICULUM_TRAINING_CONFIG) if mode == "curriculum" else dict(_POPULATION_TRAINING_CONFIG)

        # Cap verbose at 1 for SAC — verbose=2 prints stats every 4 episodes
        # which creates 50K+ lines of output per level
        sac_verbose = min(verbose, 1)
        model = SAC(
            "MlpPolicy",
            envs,
            seed=seed,
            policy_kwargs=dict(_POLICY_ARCH),
            tensorboard_log=tensorboard_log,
            verbose=sac_verbose,
            **config,
        )
        return model

    def load_model(self, path: str, envs: Any | None = None) -> Any:
        kwargs: dict[str, Any] = {}
        if envs is not None:
            kwargs["env"] = envs
        return SAC.load(path, **kwargs)

    def save_model(self, model: Any, path: str) -> None:
        model.save(path)

    def predict(self, model: Any, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action)

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
            log_interval=100,  # Log SB3 stats every 100 episodes, not every 4
        )

    def replace_env(self, model: Any, envs: Any) -> None:
        """Replace environment for SAC by saving/loading into new env.

        SAC's replay buffer is tied to the original env's shape. Simple
        set_env causes shape mismatches on the next step. Instead, save
        the model weights and load into a fresh model with the new env.
        The replay buffer is preserved so off-policy phases can accumulate
        experience across level transitions.
        """
        saved_buffer = model.replay_buffer

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "sac_transfer.zip"
            model.save(tmp_path)
            loaded = SAC.load(tmp_path, env=envs)

        # Transfer the loaded model's internals back to the original object
        # so callers that hold a reference to model still work.
        model.policy = loaded.policy
        model.env = loaded.env
        model.replay_buffer = saved_buffer  # Preserve old buffer!
        model._last_obs = envs.reset()
        model._last_episode_starts = np.ones((envs.num_envs,), dtype=bool)
        model.num_timesteps = 0

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
        """Clone model weights into a new env and apply random JAX mutation."""
        # Save parent to temp file, load into new env
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "parent.zip"
            model.save(tmp_path)
            new_model = SAC.load(tmp_path, env=envs)

        # Mutate actor params via JAX pytree
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        actor_state = new_model.policy.actor_state

        def add_noise(leaf, subkey):
            noise = jax.random.normal(subkey, leaf.shape) * mutation_rate * 0.1
            return leaf + noise

        leaves, treedef = jax.tree.flatten(actor_state.params)
        keys = jax.random.split(key, len(leaves))
        mutated_leaves = [add_noise(leaf, k) for leaf, k in zip(leaves, keys)]
        mutated_params = treedef.unflatten(mutated_leaves)

        new_model.policy.actor_state = actor_state.replace(params=mutated_params)

        return new_model

    def handle_distribution_shift(self, model: Any, kind: str) -> None:
        """Off-policy phases accumulate opponents — old data is still valid.

        Only clear buffer if explicitly requested (future "hard_reset" kind).
        """
        pass

    def create_dummy_env(self, config: Any, max_ticks: int, fighter_mass: float) -> Any:
        """Create a minimal env for model loading/cloning."""
        from src.atom.training.gym_env import AtomCombatEnv
        return DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            config=config,
            max_ticks=max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=70.0,
        ))])
