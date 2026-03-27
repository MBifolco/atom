"""
Action encoding/decoding for the 2D binned action space.

Action layout: [acceleration, stance_selector]
  - acceleration: continuous in [-1, 1], scaled to max_acceleration by the env
  - stance_selector: continuous in [-1, 1], mapped to 3 equal bins:
      [-1, -1/3)  → defending  (index 2)
      [-1/3, 1/3) → neutral    (index 0) — center bin, safe default
      [1/3, 1]    → extended   (index 1)

This module is the single source of truth for action space bounds and stance
extraction. All environments, trainers, and export paths should use these
helpers instead of inline logic.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Action space bounds (2D: acceleration + stance selector)
# ---------------------------------------------------------------------------

ACTION_SPACE_LOW = np.array([-1.0, -1.0], dtype=np.float32)
ACTION_SPACE_HIGH = np.array([1.0, 1.0], dtype=np.float32)

# Bin edges for stance selector: [-1, -1/3) = defending, [-1/3, 1/3) = neutral, [1/3, 1] = extended
STANCE_BIN_EDGES = np.array([-1 / 3, 1 / 3], dtype=np.float64)

# Maps digitize bin index (0, 1, 2) → stance index (defending=2, neutral=0, extended=1)
_BIN_TO_STANCE = np.array([2, 0, 1], dtype=np.int32)

# ---------------------------------------------------------------------------
# Stance names (single source of truth for training paths)
# ---------------------------------------------------------------------------

STANCE_NAMES = ["neutral", "extended", "defending"]


def stance_idx_to_str(idx: int) -> str:
    """Convert stance index (0/1/2) to canonical name string."""
    return STANCE_NAMES[idx]


def stance_str_to_idx(name: str) -> int:
    """Convert canonical stance name to index. Defaults to 0 (neutral)."""
    try:
        return STANCE_NAMES.index(name)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Scale + validate (single action from model.predict)
# ---------------------------------------------------------------------------


def scale_and_validate_action(
    action: np.ndarray,
    max_acceleration: float,
) -> tuple[float, int]:
    """Clip, scale, and decode a raw 2D action from model.predict().

    Args:
        action: shape (2,) — [acceleration, stance_selector]
        max_acceleration: world-config max acceleration (e.g. 4.3751)

    Returns:
        (acceleration, stance_idx) where acceleration is in world units.
    """
    acceleration = float(np.clip(action[0], -1.0, 1.0)) * max_acceleration
    stance_idx = extract_stance(action)
    return acceleration, stance_idx


def scale_acceleration_batch(
    actions: np.ndarray,
    max_acceleration: float,
) -> np.ndarray:
    """Clip and scale a batch of raw accelerations.

    Args:
        actions: shape (N, 2) raw actions from the model.
        max_acceleration: world-config max acceleration.

    Returns:
        float32 array of shape (N,) with scaled accelerations.
    """
    return np.clip(actions[:, 0], -1.0, 1.0).astype(np.float32) * max_acceleration


# ---------------------------------------------------------------------------
# NumPy helpers (single env / CPU paths)
# ---------------------------------------------------------------------------


def extract_stance(action: np.ndarray) -> int:
    """Extract stance index from a single 2D action via bin lookup.

    Args:
        action: shape (2,) — [acceleration, stance_selector]

    Returns:
        Stance index: 0 (neutral), 1 (extended), or 2 (defending).
    """
    selector = float(np.clip(action[1], -1.0, 1.0))
    if selector < -1 / 3:
        return 2  # defending
    elif selector < 1 / 3:
        return 0  # neutral
    else:
        return 1  # extended


def extract_stance_batch(actions: np.ndarray) -> np.ndarray:
    """Extract stance indices from a batch of 2D actions.

    Args:
        actions: shape (N, 2)

    Returns:
        int32 array of shape (N,) with values in {0, 1, 2}.
    """
    selectors = np.clip(actions[:, 1], -1.0, 1.0)
    bin_indices = np.digitize(selectors, STANCE_BIN_EDGES)  # 0, 1, or 2
    return _BIN_TO_STANCE[bin_indices]


# ---------------------------------------------------------------------------
# JAX helper (vmap / GPU paths)
# ---------------------------------------------------------------------------


def extract_stance_jax(action):
    """Extract stance index from a single JAX action array.

    Suitable for use inside ``jax.vmap``-ed step functions.

    Args:
        action: JAX array of shape (2,)

    Returns:
        JAX int32 scalar.
    """
    import jax.numpy as jnp

    selector = jnp.clip(action[1], -1.0, 1.0)
    # [-1, -1/3) → defending (2), [-1/3, 1/3) → neutral (0), [1/3, 1] → extended (1)
    return jnp.where(
        selector < -1 / 3,
        jnp.int32(2),  # defending
        jnp.where(selector < 1 / 3, jnp.int32(0), jnp.int32(1)),  # neutral / extended
    )
