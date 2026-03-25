"""Tests for the action_codec module — stance extraction from 4D logit actions."""

import numpy as np
import pytest


def test_extract_stance_neutral():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, 0.9, 0.1, 0.2], dtype=np.float32)
    assert extract_stance(action) == 0  # neutral has highest logit


def test_extract_stance_extended():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, -0.1, 0.8, 0.3], dtype=np.float32)
    assert extract_stance(action) == 1  # extended has highest logit


def test_extract_stance_defending():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, -0.5, -0.2, 0.9], dtype=np.float32)
    assert extract_stance(action) == 2  # defending has highest logit


def test_extract_stance_negative_logits():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, -3.0, -1.0, -2.0], dtype=np.float32)
    assert extract_stance(action) == 1  # -1.0 is highest among negatives


def test_extract_stance_equal_logits_returns_first():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 0.5, 0.5, 0.5], dtype=np.float32)
    # argmax returns first occurrence on ties
    assert extract_stance(action) == 0


def test_extract_stance_ignores_acceleration():
    from src.atom.training.action_codec import extract_stance

    action = np.array([999.0, -0.1, 0.8, 0.3], dtype=np.float32)
    assert extract_stance(action) == 1  # acceleration value doesn't affect stance


def test_extract_stance_batch():
    from src.atom.training.action_codec import extract_stance_batch

    actions = np.array(
        [
            [0.5, 0.9, 0.1, 0.2],  # neutral
            [0.5, -0.1, 0.8, 0.3],  # extended
            [0.5, -0.5, -0.2, 0.9],  # defending
            [0.0, -3.0, -1.0, -2.0],  # extended (-1.0 highest)
        ],
        dtype=np.float32,
    )
    result = extract_stance_batch(actions)
    np.testing.assert_array_equal(result, [0, 1, 2, 1])
    assert result.dtype == np.int32


def test_extract_stance_batch_single_row():
    from src.atom.training.action_codec import extract_stance_batch

    actions = np.array([[0.0, 0.1, 0.9, 0.5]], dtype=np.float32)
    result = extract_stance_batch(actions)
    np.testing.assert_array_equal(result, [1])


def test_extract_stance_batch_large():
    from src.atom.training.action_codec import extract_stance_batch

    n = 256
    actions = np.random.randn(n, 4).astype(np.float32)
    result = extract_stance_batch(actions)
    assert result.shape == (n,)
    assert result.dtype == np.int32
    assert np.all((result >= 0) & (result <= 2))


def test_action_space_constants_shape():
    from src.atom.training.action_codec import ACTION_SPACE_HIGH, ACTION_SPACE_LOW

    assert ACTION_SPACE_LOW.shape == (4,)
    assert ACTION_SPACE_HIGH.shape == (4,)
    # acceleration bounds
    assert ACTION_SPACE_LOW[0] == -1.0
    assert ACTION_SPACE_HIGH[0] == 1.0
    # logit bounds should be symmetric and wide enough for logits
    assert ACTION_SPACE_LOW[1] == ACTION_SPACE_LOW[2] == ACTION_SPACE_LOW[3]
    assert ACTION_SPACE_HIGH[1] == ACTION_SPACE_HIGH[2] == ACTION_SPACE_HIGH[3]
    assert ACTION_SPACE_HIGH[1] > 1.0  # must be wide enough for stance logits


def test_extract_stance_return_type():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 0.1, 0.9, 0.5], dtype=np.float32)
    result = extract_stance(action)
    assert isinstance(result, int)
    assert 0 <= result <= 2


def _get_jax():
    """Try to import JAX with CPU fallback. Returns (jnp, skip_reason)."""
    try:
        import os

        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax.numpy as jnp

        return jnp, None
    except (ImportError, RuntimeError) as e:
        return None, str(e)


def test_extract_stance_jax_available():
    """Test JAX stance extraction if JAX is available."""
    jnp, skip = _get_jax()
    if jnp is None:
        pytest.skip(f"JAX not available: {skip}")

    from src.atom.training.action_codec import extract_stance_jax

    action = jnp.array([0.5, -0.1, 0.8, 0.3])
    result = extract_stance_jax(action)
    assert int(result) == 1  # extended


def test_extract_stance_jax_all_stances():
    """Test JAX extraction returns all 3 stances correctly."""
    jnp, skip = _get_jax()
    if jnp is None:
        pytest.skip(f"JAX not available: {skip}")

    from src.atom.training.action_codec import extract_stance_jax

    cases = [
        (jnp.array([0.0, 0.9, 0.1, 0.2]), 0),  # neutral
        (jnp.array([0.0, 0.1, 0.9, 0.2]), 1),  # extended
        (jnp.array([0.0, 0.1, 0.2, 0.9]), 2),  # defending
    ]
    for action, expected in cases:
        assert int(extract_stance_jax(action)) == expected
