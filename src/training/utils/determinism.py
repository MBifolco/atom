"""
Deterministic seeding helpers for training and tests.

These utilities intentionally keep dependencies light so they can be used
in local test harnesses and scripts before heavyweight modules are imported.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedReport:
    """Report describing which runtime backends were successfully seeded."""

    seed: int
    python_random_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_deterministic_algorithms: bool


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> SeedReport:
    """
    Seed Python/NumPy/Torch (when available) for reproducible runs.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: Enable deterministic torch algorithm settings.

    Returns:
        SeedReport with backend seeding status.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Hash randomization can affect dict/set iteration order in some workflows.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    torch_deterministic_algorithms = False

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True

        if deterministic_torch:
            # warn_only avoids hard failures for ops lacking deterministic kernels.
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch_deterministic_algorithms = True
    except Exception:
        # Torch is optional in some local environments.
        pass

    return SeedReport(
        seed=seed,
        python_random_seeded=True,
        numpy_seeded=True,
        torch_seeded=torch_seeded,
        torch_deterministic_algorithms=torch_deterministic_algorithms,
    )


def build_seeded_env(seed: int) -> dict[str, str]:
    """
    Build a subprocess environment with deterministic seed markers.

    Args:
        seed: Non-negative integer seed.

    Returns:
        Environment dictionary suitable for subprocess execution.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    env = dict(os.environ)
    seed_value = str(seed)
    env["PYTHONHASHSEED"] = seed_value
    env["ATOM_GLOBAL_SEED"] = seed_value
    env["ATOM_TRAINING_SEED"] = seed_value
    return env
