"""Tests for the action_codec module — stance extraction from 2D binned actions."""

import numpy as np
import pytest


def test_extract_stance_neutral():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, 0.0], dtype=np.float32)  # center bin → neutral
    assert extract_stance(action) == 0


def test_extract_stance_extended():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, 0.7], dtype=np.float32)  # selector >= 1/3 → extended
    assert extract_stance(action) == 1


def test_extract_stance_defending():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.5, -0.7], dtype=np.float32)  # selector < -1/3 → defending
    assert extract_stance(action) == 2


def test_extract_stance_center_default():
    """Zero selector → neutral (safe default for zero-mean policy init)."""
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 0.0], dtype=np.float32)
    assert extract_stance(action) == 0  # neutral


def test_extract_stance_boundary_left():
    """Selector at -1/3 boundary: float64 precision lands in neutral."""
    from src.atom.training.action_codec import extract_stance

    # float64 -1/3 is exactly on the boundary (>= -1/3 → neutral)
    action_f64 = np.array([0.0, -1 / 3], dtype=np.float64)
    assert extract_stance(action_f64) == 0  # neutral

    # float32 -1/3 rounds to -0.33333334 which is < -1/3 → defending
    # This is expected float32 precision behavior
    action_f32 = np.array([0.0, -1 / 3], dtype=np.float32)
    assert extract_stance(action_f32) == 2  # defending (float32 rounds past boundary)


def test_extract_stance_boundary_right():
    """Selector at +1/3 boundary: lands in extended."""
    from src.atom.training.action_codec import extract_stance

    # float64 +1/3 is exactly on the boundary (>= 1/3 → extended)
    action_f64 = np.array([0.0, 1 / 3], dtype=np.float64)
    assert extract_stance(action_f64) == 1  # extended

    # float32 +1/3 rounds to 0.33333334 which is >= 1/3 → extended
    action_f32 = np.array([0.0, 1 / 3], dtype=np.float32)
    assert extract_stance(action_f32) == 1  # extended


def test_extract_stance_just_below_left_boundary():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, -1 / 3 - 1e-7], dtype=np.float32)
    assert extract_stance(action) == 2  # defending


def test_extract_stance_just_above_left_boundary():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, -1 / 3 + 1e-7], dtype=np.float32)
    assert extract_stance(action) == 0  # neutral


def test_extract_stance_just_below_right_boundary():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 1 / 3 - 1e-7], dtype=np.float32)
    assert extract_stance(action) == 0  # neutral


def test_extract_stance_just_above_right_boundary():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 1 / 3 + 1e-7], dtype=np.float32)
    assert extract_stance(action) == 1  # extended


def test_extract_stance_ignores_acceleration():
    from src.atom.training.action_codec import extract_stance

    action = np.array([999.0, 0.7], dtype=np.float32)
    assert extract_stance(action) == 1  # acceleration doesn't affect stance


def test_extract_stance_clamps_selector():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 5.0], dtype=np.float32)
    assert extract_stance(action) == 1  # clamped to 1.0 → extended

    action = np.array([0.0, -5.0], dtype=np.float32)
    assert extract_stance(action) == 2  # clamped to -1.0 → defending


def test_extract_stance_batch():
    from src.atom.training.action_codec import extract_stance_batch

    actions = np.array(
        [
            [0.5, 0.0],    # neutral (center)
            [0.5, 0.7],    # extended
            [0.5, -0.7],   # defending
            [0.0, -0.1],   # neutral (just inside center)
        ],
        dtype=np.float32,
    )
    result = extract_stance_batch(actions)
    np.testing.assert_array_equal(result, [0, 1, 2, 0])
    assert result.dtype == np.int32


def test_extract_stance_batch_single_row():
    from src.atom.training.action_codec import extract_stance_batch

    actions = np.array([[0.0, 0.5]], dtype=np.float32)
    result = extract_stance_batch(actions)
    np.testing.assert_array_equal(result, [1])  # extended


def test_extract_stance_batch_large():
    from src.atom.training.action_codec import extract_stance_batch

    n = 256
    actions = np.random.randn(n, 2).astype(np.float32)
    result = extract_stance_batch(actions)
    assert result.shape == (n,)
    assert result.dtype == np.int32
    assert np.all((result >= 0) & (result <= 2))


def test_extract_stance_batch_boundaries():
    """Batch path matches single-action path at bin boundaries."""
    from src.atom.training.action_codec import extract_stance, extract_stance_batch

    # Use float64 to avoid float32 rounding at exact boundaries
    boundary_values = [-1.0, -1 / 3 - 1e-7, -1 / 3, -1 / 3 + 1e-7,
                       0.0, 1 / 3 - 1e-7, 1 / 3, 1 / 3 + 1e-7, 1.0]
    actions = np.array([[0.0, v] for v in boundary_values], dtype=np.float64)
    batch_result = extract_stance_batch(actions)

    for i, v in enumerate(boundary_values):
        single_result = extract_stance(np.array([0.0, v], dtype=np.float64))
        assert batch_result[i] == single_result, f"Mismatch at selector={v}"


def test_action_space_constants_shape():
    from src.atom.training.action_codec import ACTION_SPACE_HIGH, ACTION_SPACE_LOW

    assert ACTION_SPACE_LOW.shape == (2,)
    assert ACTION_SPACE_HIGH.shape == (2,)
    # acceleration bounds
    assert ACTION_SPACE_LOW[0] == -1.0
    assert ACTION_SPACE_HIGH[0] == 1.0
    # stance selector bounds
    assert ACTION_SPACE_LOW[1] == -1.0
    assert ACTION_SPACE_HIGH[1] == 1.0


def test_extract_stance_return_type():
    from src.atom.training.action_codec import extract_stance

    action = np.array([0.0, 0.5], dtype=np.float32)
    result = extract_stance(action)
    assert isinstance(result, int)
    assert 0 <= result <= 2


def test_scale_and_validate_action():
    from src.atom.training.action_codec import scale_and_validate_action

    action = np.array([0.5, -0.7], dtype=np.float32)
    accel, stance = scale_and_validate_action(action, 4.5)
    assert abs(accel - 2.25) < 1e-5
    assert stance == 2  # defending


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

    action = jnp.array([0.5, 0.7])
    result = extract_stance_jax(action)
    assert int(result) == 1  # extended


def test_extract_stance_jax_all_stances():
    """Test JAX extraction returns all 3 stances correctly."""
    jnp, skip = _get_jax()
    if jnp is None:
        pytest.skip(f"JAX not available: {skip}")

    from src.atom.training.action_codec import extract_stance_jax

    cases = [
        (jnp.array([0.0, 0.0]), 0),     # neutral (center)
        (jnp.array([0.0, 0.7]), 1),     # extended
        (jnp.array([0.0, -0.7]), 2),    # defending
    ]
    for action, expected in cases:
        assert int(extract_stance_jax(action)) == expected


def test_extract_stance_jax_boundaries():
    """Test JAX extraction matches NumPy at bin boundaries."""
    jnp, skip = _get_jax()
    if jnp is None:
        pytest.skip(f"JAX not available: {skip}")

    from src.atom.training.action_codec import extract_stance, extract_stance_jax

    # Use values clearly inside each bin to avoid float32 rounding at exact boundaries
    boundary_values = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
    for v in boundary_values:
        jax_result = int(extract_stance_jax(jnp.array([0.0, v])))
        np_result = extract_stance(np.array([0.0, v], dtype=np.float64))
        assert jax_result == np_result, f"JAX/NumPy mismatch at selector={v}"
