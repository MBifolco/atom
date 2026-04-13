"""
Custom SBX actor/critic modules with 1D CNN history encoder.

Splits the 666D observation into:
- current (26D): base snapshot + EMA temporal features
- history (640D): 64 previous ticks × 10D features

The CNN encoder compresses 640D history → 64D context vector.
The MLP then processes 26D + 64D = 90D instead of raw 666D.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import Flatten

from .signal_engine import OBS_DIM, HISTORY_OBS_DIM, HISTORY_FEATURE_DIM, HISTORY_WINDOW

tfd = tfp.distributions

# Observation split point: first OBS_DIM (26) is current, rest is history
CURRENT_OBS_DIM = OBS_DIM       # 26
HISTORY_INPUT_DIM = HISTORY_OBS_DIM  # 640
ENCODER_OUTPUT_DIM = 64          # CNN output context vector size


class HistoryEncoder(nn.Module):
    """1D CNN that compresses 640D fight history → 64D context vector.

    Input: (batch, 640) flattened history
    Reshape to: (batch, 64, 10) — 64 timesteps, 10 features per tick
    Conv1d layers extract temporal patterns, Dense projects to context.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        # Reshape: (batch, 640) → (batch, 64 timesteps, 10 features)
        x = x.reshape(batch_size, HISTORY_WINDOW, HISTORY_FEATURE_DIM)

        # Conv1d layers (Flax Conv expects (batch, spatial, channels))
        # x is already (batch, time=64, channels=10)
        x = nn.Conv(features=32, kernel_size=(8,), strides=(4,))(x)  # → (batch, 15, 32)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4,), strides=(2,))(x)  # → (batch, 6, 64)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,), strides=(1,))(x)  # → (batch, 4, 64)
        x = nn.relu(x)

        # Flatten and project to context vector
        x = x.reshape(batch_size, -1)  # → (batch, 256)
        x = nn.Dense(ENCODER_OUTPUT_DIM)(x)  # → (batch, 64)
        x = nn.relu(x)

        return x


class SquashedGaussianActorWithHistory(nn.Module):
    """SAC actor that uses CNN encoder for fight history.

    Compatible with SBX's SACPolicy.build() which passes:
        action_dim, net_arch, activation_fn
    """
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def get_std(self):
        # Required for gSDE compatibility
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = Flatten()(x)

        # Split observation: current (26D) + history (640D)
        current_obs = x[:, :CURRENT_OBS_DIM]
        history_obs = x[:, CURRENT_OBS_DIM:]

        # Encode history through CNN
        context = HistoryEncoder()(history_obs)  # (batch, 64)

        # Concatenate current + context → 90D input to MLP
        x = jnp.concatenate([current_obs, context], axis=-1)

        # Standard MLP actor
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class ContinuousCriticWithHistory(nn.Module):
    """SAC critic that uses CNN encoder for fight history."""
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)

        # Split observation: current (26D) + history (640D)
        current_obs = x[:, :CURRENT_OBS_DIM]
        history_obs = x[:, CURRENT_OBS_DIM:]

        # Encode history through CNN
        context = HistoryEncoder()(history_obs)  # (batch, 64)

        # Concatenate current + context + action
        x = jnp.concatenate([current_obs, context, action], axis=-1)

        # Standard MLP critic
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class VectorCriticWithHistory(nn.Module):
    """Vectorized critic (multiple Q-networks) with CNN history encoder."""
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: float | None = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        vmap_critic = nn.vmap(
            ContinuousCriticWithHistory,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)
        return q_values
