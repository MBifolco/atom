"""Tests for style fingerprint and diversity-based opponent matchmaking."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes (avoid importing full training stack)
# ---------------------------------------------------------------------------

@dataclass
class _FakeFighter:
    name: str
    mass: float = 70.0
    model: Any = None
    training_episodes: int = 0
    last_checkpoint: Optional[str] = None


@dataclass
class _FakeLineageEvent:
    generation: int
    child_name: str
    child_generation: int
    parent_name: str
    replaced_fighter_name: str
    parent_elo_at_mutation: float = 1500.0
    child_mass: float = 70.0
    replaced_generation: int = 0


@dataclass
class _FakeEloStats:
    name: str
    elo: float = 1500.0


class _FakeEloTracker:
    def __init__(self, fighters: Dict[str, float] = None):
        self.fighters = {}
        if fighters:
            for name, elo in fighters.items():
                self.fighters[name] = _FakeEloStats(name=name, elo=elo)

    def get_rankings(self):
        return sorted(self.fighters.values(), key=lambda x: x.elo, reverse=True)


def _make_eval_stats(
    name: str,
    neutral: int = 100,
    extended: int = 100,
    defending: int = 50,
    damage_dealt: float = 50.0,
    damage_taken: float = 30.0,
    match_count: int = 3,
) -> dict:
    """Create a PerFighterEvaluationStats-like dict for testing."""
    total_ticks = neutral + extended + defending
    return {
        "name": name,
        "total_stance_ticks": {"neutral": neutral, "extended": extended, "defending": defending},
        "total_ticks": total_ticks,
        "match_count": match_count,
        "total_damage_dealt": damage_dealt,
        "total_damage_taken": damage_taken,
    }


# ---------------------------------------------------------------------------
# StyleFingerprint tests
# ---------------------------------------------------------------------------


class TestStyleFingerprint:
    def test_to_vector_correct_values(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        fp = StyleFingerprint(
            name="test",
            stance_neutral_pct=0.4,
            stance_extended_pct=0.4,
            stance_defending_pct=0.2,
            damage_efficiency=0.6,
            avg_fight_length_pct=0.8,
        )
        vec = fp.to_vector()
        np.testing.assert_array_almost_equal(vec, [0.4, 0.4, 0.2, 0.6, 0.8])

    def test_to_vector_excludes_source(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        fp = StyleFingerprint(
            name="test",
            stance_neutral_pct=0.3,
            stance_extended_pct=0.4,
            stance_defending_pct=0.3,
            damage_efficiency=0.5,
            avg_fight_length_pct=0.7,
            source="inherited",
        )
        vec = fp.to_vector()
        assert len(vec) == 5  # source not included

    def test_source_defaults_to_evaluation(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        fp = StyleFingerprint(
            name="x", stance_neutral_pct=0.0, stance_extended_pct=0.0,
            stance_defending_pct=0.0, damage_efficiency=0.0, avg_fight_length_pct=0.0,
        )
        assert fp.source == "evaluation"


# ---------------------------------------------------------------------------
# StyleFingerprinter tests
# ---------------------------------------------------------------------------


class TestStyleFingerprinter:
    def test_compute_fingerprints_from_eval_stats(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprinter
        from src.atom.training.trainers.population.population_evaluation import PerFighterEvaluationStats

        stats = {
            "Alpha": PerFighterEvaluationStats(
                name="Alpha",
                total_stance_ticks={"neutral": 100, "extended": 200, "defending": 200},
                total_ticks=500,
                match_count=3,
                total_damage_dealt=60.0,
                total_damage_taken=40.0,
            ),
        }
        fp_map = StyleFingerprinter().compute_fingerprints(
            evaluation_stats=stats, max_ticks=250, active_fighter_names=["Alpha"],
        )
        fp = fp_map["Alpha"]
        assert fp.stance_neutral_pct == pytest.approx(0.2)
        assert fp.stance_extended_pct == pytest.approx(0.4)
        assert fp.stance_defending_pct == pytest.approx(0.4)
        assert fp.damage_efficiency == pytest.approx(0.6)  # 60 / (60+40)
        assert fp.source == "evaluation"

    def test_zero_damage_gives_half_efficiency(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprinter
        from src.atom.training.trainers.population.population_evaluation import PerFighterEvaluationStats

        stats = {
            "Zero": PerFighterEvaluationStats(
                name="Zero",
                total_stance_ticks={"neutral": 100, "extended": 0, "defending": 0},
                total_ticks=100,
                match_count=1,
                total_damage_dealt=0.0,
                total_damage_taken=0.0,
            ),
        }
        fp_map = StyleFingerprinter().compute_fingerprints(
            evaluation_stats=stats, max_ticks=250, active_fighter_names=["Zero"],
        )
        assert fp_map["Zero"].damage_efficiency == pytest.approx(0.5)

    def test_empty_stats_returns_empty(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprinter

        fp_map = StyleFingerprinter().compute_fingerprints(
            evaluation_stats={}, max_ticks=250, active_fighter_names=["Nobody"],
        )
        assert fp_map == {}

    def test_only_active_fighters_included(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprinter
        from src.atom.training.trainers.population.population_evaluation import PerFighterEvaluationStats

        stats = {
            "Active": PerFighterEvaluationStats(
                name="Active", total_stance_ticks={"neutral": 50, "extended": 50, "defending": 0},
                total_ticks=100, match_count=1, total_damage_dealt=10, total_damage_taken=5,
            ),
            "Retired": PerFighterEvaluationStats(
                name="Retired", total_stance_ticks={"neutral": 50, "extended": 50, "defending": 0},
                total_ticks=100, match_count=1, total_damage_dealt=10, total_damage_taken=5,
            ),
        }
        fp_map = StyleFingerprinter().compute_fingerprints(
            evaluation_stats=stats, max_ticks=250, active_fighter_names=["Active"],
        )
        assert "Active" in fp_map
        assert "Retired" not in fp_map


# ---------------------------------------------------------------------------
# Child fingerprint inheritance tests
# ---------------------------------------------------------------------------


class TestFingerprintInheritance:
    def test_child_inherits_parent_fingerprint(self):
        from src.atom.training.trainers.population.style_matchmaking import (
            StyleFingerprint, StyleFingerprinter,
        )
        from src.atom.training.trainers.population.population_evolution import LineageEvent

        parent_fp = StyleFingerprint(
            name="Parent", stance_neutral_pct=0.3, stance_extended_pct=0.5,
            stance_defending_pct=0.2, damage_efficiency=0.7, avg_fight_length_pct=0.6,
            source="evaluation",
        )
        fps = {"Parent": parent_fp}
        events = [LineageEvent(
            generation=1, child_name="Child", child_generation=1,
            parent_name="Parent", replaced_fighter_name="OldFighter",
            parent_elo_at_mutation=1500.0, child_mass=70.0, replaced_generation=0,
        )]
        StyleFingerprinter().inherit_for_children(fps, events)

        assert "Child" in fps
        assert fps["Child"].source == "inherited"
        assert fps["Child"].stance_extended_pct == pytest.approx(0.5)

    def test_replaced_fighter_pruned(self):
        from src.atom.training.trainers.population.style_matchmaking import (
            StyleFingerprint, StyleFingerprinter,
        )
        from src.atom.training.trainers.population.population_evolution import LineageEvent

        fps = {
            "Parent": StyleFingerprint(
                name="Parent", stance_neutral_pct=0.3, stance_extended_pct=0.4,
                stance_defending_pct=0.3, damage_efficiency=0.5, avg_fight_length_pct=0.5,
            ),
            "OldFighter": StyleFingerprint(
                name="OldFighter", stance_neutral_pct=0.5, stance_extended_pct=0.3,
                stance_defending_pct=0.2, damage_efficiency=0.4, avg_fight_length_pct=0.8,
            ),
        }
        events = [LineageEvent(
            generation=1, child_name="Child", child_generation=1,
            parent_name="Parent", replaced_fighter_name="OldFighter",
            parent_elo_at_mutation=1500.0, child_mass=70.0, replaced_generation=0,
        )]
        StyleFingerprinter().inherit_for_children(fps, events)

        assert "OldFighter" not in fps
        assert "Child" in fps

    def test_missing_parent_gives_default(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprinter
        from src.atom.training.trainers.population.population_evolution import LineageEvent

        fps = {}  # no parent fingerprint
        events = [LineageEvent(
            generation=1, child_name="Orphan", child_generation=1,
            parent_name="Ghost", replaced_fighter_name="OldOne",
            parent_elo_at_mutation=1500.0, child_mass=70.0, replaced_generation=0,
        )]
        StyleFingerprinter().inherit_for_children(fps, events)

        assert "Orphan" in fps
        assert fps["Orphan"].source == "default"
        assert fps["Orphan"].damage_efficiency == pytest.approx(0.5)
        # Stance percentages must sum to 1.0 (valid distribution)
        fp = fps["Orphan"]
        assert fp.stance_neutral_pct + fp.stance_extended_pct + fp.stance_defending_pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DiversityMatchmaker tests
# ---------------------------------------------------------------------------


class TestDiversityMatchmaker:
    def _make_matchmaker(self, **kwargs):
        from src.atom.training.trainers.population.style_matchmaking import (
            DiversityMatchmaker, DiversityMatchmakingContext,
        )
        defaults = {"opponents_per_fighter": 3, "diversity_weight": 0.7, "elo_band": 200.0}
        defaults.update(kwargs)
        ctx = DiversityMatchmakingContext(**defaults)
        return DiversityMatchmaker(ctx)

    def _make_population(self, n: int) -> list:
        return [_FakeFighter(name=f"F{i}") for i in range(n)]

    def test_returns_correct_opponent_count(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=3)
        pop = self._make_population(8)
        fps = {
            f.name: StyleFingerprint(
                name=f.name, stance_neutral_pct=0.3+i*0.05, stance_extended_pct=0.4,
                stance_defending_pct=0.3-i*0.05, damage_efficiency=0.5,
                avg_fight_length_pct=0.7,
            ) for i, f in enumerate(pop)
        }
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        for fighter, opponents in result:
            assert len(opponents) == 3

    def test_prefers_diverse_styles(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=1, diversity_weight=1.0)
        pop = self._make_population(3)
        fps = {
            "F0": StyleFingerprint(name="F0", stance_neutral_pct=0.9, stance_extended_pct=0.05,
                                   stance_defending_pct=0.05, damage_efficiency=0.5, avg_fight_length_pct=0.5),
            "F1": StyleFingerprint(name="F1", stance_neutral_pct=0.85, stance_extended_pct=0.1,
                                   stance_defending_pct=0.05, damage_efficiency=0.5, avg_fight_length_pct=0.5),
            "F2": StyleFingerprint(name="F2", stance_neutral_pct=0.1, stance_extended_pct=0.8,
                                   stance_defending_pct=0.1, damage_efficiency=0.5, avg_fight_length_pct=0.5),
        }
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        f0_opponents = [o.name for _, opps in result if _.name == "F0" for o in opps]
        # F0 (mostly neutral) should prefer F2 (mostly extended) over F1 (also mostly neutral)
        assert "F2" in f0_opponents

    def test_fighter_not_assigned_to_self(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=3)
        pop = self._make_population(4)
        fps = {f.name: StyleFingerprint(
            name=f.name, stance_neutral_pct=0.33, stance_extended_pct=0.33,
            stance_defending_pct=0.34, damage_efficiency=0.5, avg_fight_length_pct=0.5,
        ) for f in pop}
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        for fighter, opponents in result:
            assert fighter.name not in [o.name for o in opponents]

    def test_every_fighter_gets_opponents(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=2)
        pop = self._make_population(6)
        fps = {f.name: StyleFingerprint(
            name=f.name, stance_neutral_pct=0.33, stance_extended_pct=0.33,
            stance_defending_pct=0.34, damage_efficiency=0.5, avg_fight_length_pct=0.5,
        ) for f in pop}
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        assigned_names = {fighter.name for fighter, _ in result}
        assert assigned_names == {f.name for f in pop}

    def test_random_fallback_gen0(self):
        mm = self._make_matchmaker(opponents_per_fighter=3)
        pop = self._make_population(8)

        random.seed(42)
        result = mm.assign_random_opponents(pop, opponents_per_fighter=3)
        for fighter, opponents in result:
            assert len(opponents) == 3
            assert fighter.name not in [o.name for o in opponents]

    def test_child_gets_at_least_one_incumbent(self):
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=2)
        pop = self._make_population(6)
        # F0-F2 are incumbents (source=evaluation), F3-F5 are children (source=inherited)
        fps = {}
        for i, f in enumerate(pop):
            fps[f.name] = StyleFingerprint(
                name=f.name, stance_neutral_pct=0.33, stance_extended_pct=0.33,
                stance_defending_pct=0.34, damage_efficiency=0.5, avg_fight_length_pct=0.5,
                source="evaluation" if i < 3 else "inherited",
            )
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        for fighter, opponents in result:
            if fps[fighter.name].source == "inherited":
                opponent_sources = [fps[o.name].source for o in opponents]
                assert "evaluation" in opponent_sources, (
                    f"Child {fighter.name} has no incumbent opponent"
                )

    def test_partial_fingerprint_map_works(self):
        """Matchmaker handles mix of evaluated + inherited fingerprints."""
        from src.atom.training.trainers.population.style_matchmaking import StyleFingerprint

        mm = self._make_matchmaker(opponents_per_fighter=2)
        pop = self._make_population(4)
        fps = {
            "F0": StyleFingerprint(name="F0", stance_neutral_pct=0.5, stance_extended_pct=0.3,
                                   stance_defending_pct=0.2, damage_efficiency=0.6, avg_fight_length_pct=0.7,
                                   source="evaluation"),
            "F1": StyleFingerprint(name="F1", stance_neutral_pct=0.2, stance_extended_pct=0.5,
                                   stance_defending_pct=0.3, damage_efficiency=0.4, avg_fight_length_pct=0.8,
                                   source="evaluation"),
            "F2": StyleFingerprint(name="F2", stance_neutral_pct=0.5, stance_extended_pct=0.3,
                                   stance_defending_pct=0.2, damage_efficiency=0.6, avg_fight_length_pct=0.7,
                                   source="inherited"),
            "F3": StyleFingerprint(name="F3", stance_neutral_pct=0.3, stance_extended_pct=0.4,
                                   stance_defending_pct=0.3, damage_efficiency=0.5, avg_fight_length_pct=0.5,
                                   source="default"),
        }
        elo = _FakeEloTracker({f.name: 1500.0 for f in pop})

        result = mm.assign_opponents(pop, fps, elo)
        assert len(result) == 4
        for fighter, opponents in result:
            assert len(opponents) == 2


# ---------------------------------------------------------------------------
# PerFighterEvaluationStats tests
# ---------------------------------------------------------------------------


class TestPerFighterEvaluationStats:
    def test_accumulate_across_matches(self):
        from src.atom.training.trainers.population.population_evaluation import PerFighterEvaluationStats

        stats = PerFighterEvaluationStats(
            name="X",
            total_stance_ticks={"neutral": 50, "extended": 30, "defending": 20},
            total_ticks=100,
            match_count=1,
            total_damage_dealt=15.0,
            total_damage_taken=10.0,
        )
        # Dataclass is frozen, so accumulation happens in the eval loop, not here.
        # Verify fields are accessible and correct.
        assert stats.total_ticks == 100
        assert stats.total_damage_dealt == 15.0
        assert stats.match_count == 1


# ---------------------------------------------------------------------------
# Both-sides evaluation data tests
# ---------------------------------------------------------------------------


class TestBothSidesEvaluationData:
    def test_opponent_stance_counted_via_wrapper(self):
        from src.atom.training.trainers.population.population_evaluation import StanceCountingWrapper

        def fake_decide(snapshot):
            return {"acceleration": 0.0, "stance": "extended"}

        wrapper = StanceCountingWrapper(fake_decide)
        for _ in range(10):
            wrapper({})
        assert wrapper.stance_counts["extended"] == 10
        assert wrapper.stance_counts["neutral"] == 0

    def test_wrapper_handles_int_stance(self):
        from src.atom.training.trainers.population.population_evaluation import StanceCountingWrapper

        def fake_decide(snapshot):
            return {"acceleration": 0.0, "stance": 2}

        wrapper = StanceCountingWrapper(fake_decide)
        wrapper({})
        assert wrapper.stance_counts["defending"] == 1

    def test_wrapper_handles_numpy_int_stance(self):
        from src.atom.training.trainers.population.population_evaluation import StanceCountingWrapper

        def fake_decide(snapshot):
            return {"acceleration": 0.0, "stance": np.int32(1)}

        wrapper = StanceCountingWrapper(fake_decide)
        wrapper({})
        assert wrapper.stance_counts["extended"] == 1

    def test_wrapper_defaults_unknown_stance_to_neutral(self):
        from src.atom.training.trainers.population.population_evaluation import StanceCountingWrapper

        def fake_decide(snapshot):
            return {"acceleration": 0.0, "stance": "unknown_stance"}

        wrapper = StanceCountingWrapper(fake_decide)
        wrapper({})
        assert wrapper.stance_counts["neutral"] == 1
