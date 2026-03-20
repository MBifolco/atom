"""
Pytest configuration for Atom Combat tests.

Goals:
- Keep imports stable regardless of invocation path.
- Provide deterministic seed fixtures for local reproducibility.
- Apply tier markers automatically while we migrate legacy tests into folders.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest


# Add project root to Python path.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


DEFAULT_TEST_SEED = int(os.environ.get("ATOM_TEST_SEED", "1337"))


def _seed_python_and_numpy(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _add_marker_once(item: pytest.Item, marker_name: str) -> None:
    if item.get_closest_marker(marker_name) is None:
        item.add_marker(getattr(pytest.mark, marker_name))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Classify tests into tier markers.

    Folder-based markers are preferred. Legacy root tests default to integration.
    """
    del config  # Not used, but kept for hook signature compatibility.

    for item in items:
        file_path = Path(str(item.fspath)).resolve()
        parts = {p.lower() for p in file_path.parts}
        file_name = file_path.name.lower()

        if "unit" in parts:
            _add_marker_once(item, "unit")
            continue

        if "integration" in parts:
            _add_marker_once(item, "integration")
            continue

        if "training" in parts:
            _add_marker_once(item, "training")
            _add_marker_once(item, "slow")
            continue

        if "e2e" in parts:
            _add_marker_once(item, "e2e")
            _add_marker_once(item, "slow")
            continue

        # Legacy flat tests directory -> treat as integration by default.
        _add_marker_once(item, "integration")

        slow_name_tokens = ("comprehensive", "coverage", "massive", "push", "real_integration")
        if any(token in file_name for token in slow_name_tokens):
            _add_marker_once(item, "slow")


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Project root path helper for tests."""
    return project_root


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Shared deterministic seed for local test runs."""
    return DEFAULT_TEST_SEED


@pytest.fixture(autouse=True)
def deterministic_python_numpy_seed(test_seed: int) -> None:
    """
    Apply deterministic Python/NumPy seed before each test.

    Torch/JAX seeding stays opt-in to avoid heavy imports for simple tests.
    """
    _seed_python_and_numpy(test_seed)


@pytest.fixture
def seed_torch(test_seed: int) -> int:
    """Optional fixture to seed torch in tests that need it."""
    try:
        import torch

        torch.manual_seed(test_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(test_seed)
    except Exception:
        pass
    return test_seed
