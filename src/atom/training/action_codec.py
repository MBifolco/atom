"""
Action encoding/decoding for the 4D logit-based action space.

Action layout: [acceleration, logit_neutral, logit_extended, logit_defending]
  - acceleration: continuous in [-1, 1], scaled to max_acceleration by the env
  - logits 1-3: stance scores — argmax selects the active stance (0/1/2)

This module is the single source of truth for action space bounds and stance
extraction. All environments, trainers, and export paths should use these
helpers instead of inline int() casts.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Action space bounds
# ---------------------------------------------------------------------------

ACTION_SPACE_LOW = np.array([-1.0, -5.0, -5.0, -5.0], dtype=np.float32)
ACTION_SPACE_HIGH = np.array([1.0, 5.0, 5.0, 5.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# NumPy helpers (single env / CPU paths)
# ---------------------------------------------------------------------------


def extract_stance(action: np.ndarray) -> int:
    """Extract stance index from a single 4D action via argmax over logits.

    Args:
        action: shape (4,) — [accel, logit_neutral, logit_extended, logit_defending]

    Returns:
        Stance index: 0 (neutral), 1 (extended), or 2 (defending).
    """
    return int(np.argmax(action[1:4]))


def extract_stance_batch(actions: np.ndarray) -> np.ndarray:
    """Extract stance indices from a batch of 4D actions.

    Args:
        actions: shape (N, 4)

    Returns:
        int32 array of shape (N,) with values in {0, 1, 2}.
    """
    return np.argmax(actions[:, 1:4], axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# JAX helper (vmap / GPU paths)
# ---------------------------------------------------------------------------


def extract_stance_jax(action):
    """Extract stance index from a single JAX action array.

    Suitable for use inside ``jax.vmap``-ed step functions.

    Args:
        action: JAX array of shape (4,)

    Returns:
        JAX int32 scalar.
    """
    import jax.numpy as jnp

    return jnp.argmax(action[1:4]).astype(jnp.int32)
