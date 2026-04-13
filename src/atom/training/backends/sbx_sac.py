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

# Low-temperature stochastic sampling for deterministic evaluation.
# SAC's mean (tanh(mean)) is never directly optimized, so pure deterministic
# can be arbitrarily bad. Low-temp sampling (mean + std*temp*noise) preserves
# the distribution shape while being much more consistent.
DETERMINISTIC_TEMPERATURE = 0.2


_POLICY_ARCH = {
    "net_arch": [512, 512, 256],
}

_CURRICULUM_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": 0.005,  # Fixed low: 0.02 swamped shaping rewards (O(0.001)) causing deterministic collapse
    "train_freq": 1,
    "gradient_steps": 2,
}

_POPULATION_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 500_000,
    "learning_starts": 5000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": 0.005,  # Fixed low: 0.02 swamped shaping rewards (O(0.001)) causing deterministic collapse
    "train_freq": 1,
    "gradient_steps": 2,
}


_EMA_DECAY = 0.995  # Polyak averaging for evaluation actor (smooths oscillation)


class SBXSACBackend:
    """SBX (JAX) Soft Actor-Critic training backend.

    All-JAX pipeline: physics (vmap) and policy on same device.
    Off-policy with replay buffer — gets 10-100x more learning per env step.
    """

    def __init__(self):
        # SBX/JAX handles device automatically
        self._ema_actor_params = None  # Polyak-averaged actor for stable deterministic eval
        self._eval_key = jax.random.PRNGKey(42)  # RNG key for low-temp sampling

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

        policy_kwargs = dict(_POLICY_ARCH)

        # Use CNN history encoder when observation space is 666D
        from ..signal_engine import OBS_DIM_WITH_HISTORY
        obs_dim = envs.observation_space.shape[0]
        if obs_dim == OBS_DIM_WITH_HISTORY:
            from ..history_encoder import (
                SquashedGaussianActorWithHistory,
                VectorCriticWithHistory,
            )
            policy_kwargs["actor_class"] = SquashedGaussianActorWithHistory
            policy_kwargs["vector_critic_class"] = VectorCriticWithHistory

        # Force verbose=0 for SAC — SBX logs rollout/train stats every few
        # episodes even at verbose=1, creating massive output. Our own
        # progress logging (every 100 episodes) provides all needed info.
        sac_verbose = 0
        model = SAC(
            "MlpPolicy",
            envs,
            seed=seed,
            policy_kwargs=policy_kwargs,
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
        if deterministic:
            # Low-temperature stochastic: mean + std * 0.2 * noise.
            # SAC's pure deterministic (tanh(mean)) is never directly optimized
            # and can be arbitrarily bad. Low-temp sampling preserves the learned
            # distribution while being much more consistent than full stochastic.
            actor_state = model.policy.actor_state
            if self._ema_actor_params is not None:
                actor_state = actor_state.replace(params=self._ema_actor_params)
            obs_2d = np.atleast_2d(obs)
            self._eval_key, subkey = jax.random.split(self._eval_key)
            action = self._low_temp_sample(actor_state, obs_2d, subkey)
            action = np.asarray(action).reshape((-1, *model.action_space.shape))
            action = np.clip(action, -1, 1)
            action = model.policy.unscale_action(action)
            if obs.ndim == 1:
                action = action.squeeze(axis=0)
            return action
        action, _ = model.predict(obs, deterministic=False)
        return np.asarray(action)

    @staticmethod
    @jax.jit
    def _low_temp_sample(actor_state, observations, key):
        """Sample from the policy with reduced temperature (std * 0.2)."""
        dist = actor_state.apply_fn(actor_state.params, observations)
        # Get the underlying normal distribution (before tanh transform)
        normal = dist.distribution
        mean = normal.loc
        std = normal.scale.diag  # LinearOperatorDiag → plain array
        noise = jax.random.normal(key, mean.shape)
        # Low-temp: sample close to mean but not exactly at it
        raw_action = mean + std * DETERMINISTIC_TEMPERATURE * noise
        return jnp.tanh(raw_action)

    def update_ema_actor(self, model: Any) -> None:
        """Update Polyak-averaged actor params from current training actor."""
        current_params = model.policy.actor_state.params
        if self._ema_actor_params is None:
            self._ema_actor_params = jax.tree.map(lambda p: p.copy(), current_params)
        else:
            self._ema_actor_params = jax.tree.map(
                lambda ema, cur: _EMA_DECAY * ema + (1 - _EMA_DECAY) * cur,
                self._ema_actor_params,
                current_params,
            )

    def learn(
        self,
        model: Any,
        total_timesteps: int,
        callback: Any | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Any:
        # Initialize EMA from current actor params before training starts
        if self._ema_actor_params is None:
            self._ema_actor_params = jax.tree.map(
                lambda p: p.copy(), model.policy.actor_state.params
            )

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

        # Re-initialize EMA from the (preserved) actor params so the
        # evaluation policy tracks the new level's training trajectory.
        self._ema_actor_params = jax.tree.map(
            lambda p: p.copy(), model.policy.actor_state.params
        )

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
