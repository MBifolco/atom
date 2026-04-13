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

import jax
import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import Flatten

from .signal_engine import OBS_DIM, HISTORY_OBS_DIM, HISTORY_FEATURE_DIM, HISTORY_WINDOW

tfd = tfp.distributions

# Mixture of Gaussians config
NUM_MIXTURE_COMPONENTS = 4

class TanhMixtureDistribution:
    """Mixture of K TanhTransformed Gaussians for multi-modal SAC policy.

    Implements .sample(seed=), .log_prob(), and .mode() to be compatible
    with SBX's SAC training loop.
    """

    def __init__(self, means, log_stds, logits, log_std_min=-20, log_std_max=2):
        """
        Args:
            means: (batch, K, action_dim) — per-component means
            log_stds: (batch, K, action_dim) — per-component log stds
            logits: (batch, K) — mixture gating logits (unnormalized)
        """
        self.means = means
        self.log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        self.logits = logits
        self.K = means.shape[1]
        self.action_dim = means.shape[2]

    def sample(self, seed):
        """Sample from the mixture: pick component, then sample from it."""
        key1, key2 = jax.random.split(seed)
        # Sample component indices from categorical(logits)
        component_idx = jax.random.categorical(key1, self.logits)  # (batch,)
        batch_size = self.means.shape[0]
        batch_idx = jnp.arange(batch_size)

        # Get selected component's mean and std
        mean = self.means[batch_idx, component_idx]      # (batch, action_dim)
        log_std = self.log_stds[batch_idx, component_idx]  # (batch, action_dim)
        std = jnp.exp(log_std)

        # Sample from the selected Gaussian, then tanh
        noise = jax.random.normal(key2, mean.shape)
        raw_action = mean + std * noise
        return jnp.tanh(raw_action)

    def log_prob(self, action):
        """Log probability under the mixture (log-sum-exp over components).

        For each component k:
          log p_k(action) = log_prob_normal(atanh(action)) - log(1 - action^2)
        Then:
          log p(action) = log_sum_exp(log_weights + log_p_k) over k
        """
        # Inverse tanh to get raw action
        # Clip to avoid atanh(±1) = ±inf
        action_clipped = jnp.clip(action, -0.999, 0.999)
        raw_action = jnp.arctanh(action_clipped)  # (batch, action_dim)

        # Expand for K components: (batch, 1, action_dim)
        raw_expanded = raw_action[:, None, :]

        # Per-component log probs under Normal
        stds = jnp.exp(self.log_stds)  # (batch, K, action_dim)
        var = stds ** 2
        log_normal = -0.5 * (
            jnp.sum((raw_expanded - self.means) ** 2 / var, axis=-1)
            + jnp.sum(jnp.log(var), axis=-1)
            + self.action_dim * jnp.log(2 * jnp.pi)
        )  # (batch, K)

        # Tanh correction: -sum(log(1 - tanh(raw)^2)) per dimension
        tanh_correction = jnp.sum(
            jnp.log(jnp.maximum(1.0 - action_clipped ** 2, 1e-6)), axis=-1
        )  # (batch,)

        # Per-component log prob = normal_log_prob - tanh_correction
        component_log_probs = log_normal - tanh_correction[:, None]  # (batch, K)

        # Mixture log prob: log_sum_exp(log_weights + component_log_probs)
        log_weights = jax.nn.log_softmax(self.logits)  # (batch, K)
        mixture_log_prob = jax.nn.logsumexp(
            log_weights + component_log_probs, axis=-1
        )  # (batch,)

        return mixture_log_prob

    def mode(self):
        """Deterministic output: mean of the highest-weight component."""
        best_idx = jnp.argmax(self.logits, axis=-1)  # (batch,)
        batch_idx = jnp.arange(self.means.shape[0])
        best_mean = self.means[batch_idx, best_idx]  # (batch, action_dim)
        return jnp.tanh(best_mean)


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
    """SAC actor with CNN history encoder and Mixture of Gaussians output.

    K=4 mixture components allow the policy to maintain distinct strategies
    for different opponent archetypes (aggressive, defensive, counter-punch,
    distance management). The gating network selects components based on
    the CNN context vector (opponent behavioral fingerprint).

    Compatible with SBX's SACPolicy.build() which passes:
        action_dim, net_arch, activation_fn
    """
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_components: int = NUM_MIXTURE_COMPONENTS

    def get_std(self):
        # Required for gSDE compatibility
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> TanhMixtureDistribution:
        x = Flatten()(x)

        # Split observation: current (26D) + history (640D)
        current_obs = x[:, :CURRENT_OBS_DIM]
        history_obs = x[:, CURRENT_OBS_DIM:]

        # Encode history through CNN
        context = HistoryEncoder()(history_obs)  # (batch, 64)

        # Concatenate current + context → 90D input to MLP
        x = jnp.concatenate([current_obs, context], axis=-1)

        # Shared MLP backbone
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        # K mixture components: each has (mean, log_std) for action_dim
        K = self.n_components
        means = nn.Dense(K * self.action_dim)(x)       # (batch, K*action_dim)
        log_stds = nn.Dense(K * self.action_dim)(x)    # (batch, K*action_dim)
        logits = nn.Dense(K)(x)                         # (batch, K)

        # Reshape to (batch, K, action_dim)
        batch_size = x.shape[0]
        means = means.reshape(batch_size, K, self.action_dim)
        log_stds = log_stds.reshape(batch_size, K, self.action_dim)

        return TanhMixtureDistribution(
            means, log_stds, logits,
            log_std_min=self.log_std_min, log_std_max=self.log_std_max,
        )


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
