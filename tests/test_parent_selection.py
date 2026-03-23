"""Tests for rank-weighted parent selection in population evolution."""

from __future__ import annotations

import random
from collections import Counter
from unittest.mock import MagicMock

import pytest


def _make_survivor_mocks(n: int) -> list:
    """Create n mock survivors with distinct names."""
    survivors = []
    for i in range(n):
        mock = MagicMock()
        mock.name = f"fighter_{i}"
        survivors.append(mock)
    return survivors


class TestRankWeightedParentSelection:
    """Tests for the rank-weighted parent selection logic."""

    def test_top_ranked_gets_more_children(self):
        """Over many samples, rank 1 should be selected more than rank N."""
        from src.atom.training.trainers.population.population_evolution import (
            PopulationEvolver,
        )

        survivors = _make_survivor_mocks(4)

        # Sample many times to check distribution
        random.seed(42)
        counts = Counter()
        for _ in range(1000):
            parent = PopulationEvolver.select_parent(survivors)
            counts[parent.name] += 1

        # Rank 0 (top) should be picked more than rank 3 (bottom)
        assert counts["fighter_0"] > counts["fighter_3"]

    def test_all_survivors_can_be_selected(self):
        """Every survivor should have a nonzero chance of being selected."""
        from src.atom.training.trainers.population.population_evolution import (
            PopulationEvolver,
        )

        survivors = _make_survivor_mocks(4)

        random.seed(42)
        selected_names = set()
        for _ in range(200):
            parent = PopulationEvolver.select_parent(survivors)
            selected_names.add(parent.name)

        assert selected_names == {f"fighter_{i}" for i in range(4)}

    def test_single_survivor_always_selected(self):
        """With one survivor, it's always selected."""
        from src.atom.training.trainers.population.population_evolution import (
            PopulationEvolver,
        )

        survivors = _make_survivor_mocks(1)
        parent = PopulationEvolver.select_parent(survivors)
        assert parent.name == "fighter_0"

    def test_two_survivors_top_favored(self):
        """With two survivors, rank 1 should get ~2/3 and rank 2 ~1/3."""
        from src.atom.training.trainers.population.population_evolution import (
            PopulationEvolver,
        )

        survivors = _make_survivor_mocks(2)

        random.seed(42)
        counts = Counter()
        for _ in range(3000):
            parent = PopulationEvolver.select_parent(survivors)
            counts[parent.name] += 1

        # Weights are [2, 1] so rank 0 gets ~66%, rank 1 ~33%
        ratio = counts["fighter_0"] / max(1, counts["fighter_1"])
        assert 1.5 < ratio < 2.5  # should be ~2.0

    def test_weights_are_rank_based(self):
        """Weights should be N, N-1, ..., 1 for N survivors."""
        from src.atom.training.trainers.population.population_evolution import (
            PopulationEvolver,
        )

        survivors = _make_survivor_mocks(5)

        # Verify the weight computation directly
        weights = PopulationEvolver.rank_weights(len(survivors))
        assert weights == [5, 4, 3, 2, 1]
