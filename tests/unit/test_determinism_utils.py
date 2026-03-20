"""Tests for deterministic seeding helpers."""

import random

import numpy as np

from src.training.utils.determinism import build_seeded_env, set_global_seeds


def test_set_global_seeds_reproducible_python_and_numpy():
    set_global_seeds(123)
    first_python = random.random()
    first_numpy = float(np.random.rand())

    set_global_seeds(123)
    second_python = random.random()
    second_numpy = float(np.random.rand())

    assert first_python == second_python
    assert first_numpy == second_numpy


def test_set_global_seeds_rejects_negative_seed():
    try:
        set_global_seeds(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative seed")


def test_build_seeded_env_populates_expected_variables():
    env = build_seeded_env(77)
    assert env["PYTHONHASHSEED"] == "77"
    assert env["ATOM_GLOBAL_SEED"] == "77"
    assert env["ATOM_TRAINING_SEED"] == "77"
