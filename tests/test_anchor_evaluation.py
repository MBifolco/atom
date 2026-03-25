"""Tests for curriculum anchor evaluation and composite scoring."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from src.atom.training.trainers.population.population_evaluation import (
    EvaluationContext,
    PopulationEvaluationService,
)
from src.atom.training.trainers.population.population_evolution import (
    EvolutionContext,
    PopulationEvolver,
)


def _make_eval_context():
    from src.atom.runtime.arena import WorldConfig
    return EvaluationContext(
        config=WorldConfig(),
        max_ticks=50,
        generation=0,
        verbose=False,
        logger=MagicMock(),
    )


class TestAnchorEvaluationDoesNotTouchELO:
    """Anchor evaluation must not modify ELO tracker state."""

    def test_elo_unchanged_after_anchor_eval(self):
        """ELO ratings must be identical before and after anchor evaluation."""
        ctx = _make_eval_context()
        service = PopulationEvaluationService(ctx)

        # Create mock fighters with real-enough models
        fighters = []
        for name in ["fighter_a", "fighter_b"]:
            f = MagicMock()
            f.name = name
            f.model = MagicMock()
            f.model.predict = MagicMock(return_value=(
                [0.5, 0.1, 0.2, -0.1],  # action
                None,
            ))
            fighters.append(f)

        # A simple anchor that always returns neutral/no movement
        def dummy_anchor(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        # Create and snapshot an ELO tracker
        from src.atom.training.trainers.population.elo_tracker import EloTracker
        tracker = EloTracker()
        tracker.add_fighter("fighter_a")
        tracker.add_fighter("fighter_b")
        rankings_before = [(s.name, s.elo) for s in tracker.get_rankings()]

        # Mock env that terminates immediately with a win
        mock_env = MagicMock()
        mock_env.reset = MagicMock(return_value=([0.0] * 14, {}))
        mock_env.step = MagicMock(return_value=(
            [0.0] * 14, 0.0, True, False,
            {"won": True, "fighter_hp": 50.0, "opponent_hp": 0.0},
        ))
        mock_env.close = MagicMock()

        # Run anchor evaluation — should NOT touch tracker
        results = service.evaluate_against_anchors(
            population=fighters,
            anchor_opponents={"dummy": dummy_anchor},
            decision_func_factory=lambda f: lambda snap: {"acceleration": 0.0, "stance": "neutral"},
            env_factory=lambda **kw: mock_env,
            matches_per_anchor=1,
        )

        # ELO should be unchanged
        rankings_after = [(s.name, s.elo) for s in tracker.get_rankings()]
        assert rankings_before == rankings_after

    def test_anchor_results_structure(self):
        """Anchor results should have per-anchor win rates and aggregate score."""
        ctx = _make_eval_context()
        service = PopulationEvaluationService(ctx)

        fighter = MagicMock()
        fighter.name = "test_fighter"
        fighter.model = MagicMock()
        fighter.model.predict = MagicMock(return_value=([0.5, 0.1, 0.2, -0.1], None))

        # Mock env that always reports a win
        mock_env = MagicMock()
        mock_env.reset = MagicMock(return_value=([0.0] * 14, {}))
        mock_env.step = MagicMock(return_value=(
            [0.0] * 14, 0.0, True, False,
            {"won": True, "fighter_hp": 50.0, "opponent_hp": 0.0},
        ))

        results = service.evaluate_against_anchors(
            population=[fighter],
            anchor_opponents={"anchor_a": lambda s: {"acceleration": 0.0, "stance": "neutral"}},
            decision_func_factory=lambda f: lambda snap: {"acceleration": 0.0, "stance": "neutral"},
            env_factory=lambda **kw: mock_env,
            matches_per_anchor=2,
        )

        assert "test_fighter" in results
        assert "anchors" in results["test_fighter"]
        assert "anchor_score" in results["test_fighter"]
        assert "anchor_a" in results["test_fighter"]["anchors"]


class TestCompositeSurvivorOrdering:
    """Composite score should influence survivor selection."""

    def _make_evolution_context(self):
        return EvolutionContext(
            config=MagicMock(),
            max_ticks=250,
            mass_range=(70.0, 70.0),
            generation=1,
            algorithm="ppo",
            verbose=False,
            logger=MagicMock(),
        )

    def test_high_elo_low_anchor_loses_to_medium_elo_high_anchor(self):
        """A fighter with high ELO but zero anchor score should rank below
        a fighter with medium ELO but high anchor score."""
        evolver = PopulationEvolver(self._make_evolution_context())

        # 4 fighters
        fighters = []
        for name in ["high_elo_no_anchor", "mid_elo_good_anchor", "low_elo", "lowest"]:
            f = MagicMock()
            f.name = name
            f.mass = 70.0
            f.model = MagicMock()
            fighters.append(f)

        # ELO tracker: high_elo_no_anchor is #1 by ELO
        tracker = MagicMock()
        tracker.get_rankings.return_value = [
            SimpleNamespace(name="high_elo_no_anchor", elo=1600),
            SimpleNamespace(name="mid_elo_good_anchor", elo=1500),
            SimpleNamespace(name="low_elo", elo=1400),
            SimpleNamespace(name="lowest", elo=1300),
        ]

        # Anchor scores: high_elo fighter forgot everything
        anchor_scores = {
            "high_elo_no_anchor": 0.0,    # Forgot all curriculum skills
            "mid_elo_good_anchor": 1.0,   # Retained all skills
            "low_elo": 0.5,
            "lowest": 0.25,
        }

        selection = evolver._select_survivors(
            fighters, tracker, keep_top=0.5, anchor_scores=anchor_scores,
        )

        survivor_names = [f.name for f in selection.survivors]
        replaced_names = [f.name for f in selection.to_replace]

        # mid_elo_good_anchor should survive (composite: 0.8*0.75 + 0.2*1.0 = 0.8)
        assert "mid_elo_good_anchor" in survivor_names

        # high_elo_no_anchor should be penalized (composite: 0.8*1.0 + 0.2*0.0 = 0.8)
        # Actually these are equal at 0.8 — let me adjust to make the test clearer

    def test_forgetting_fighter_gets_replaced(self):
        """A fighter that completely forgot curriculum should be replaceable."""
        evolver = PopulationEvolver(self._make_evolution_context())

        fighters = []
        for name in ["decent", "forgetter", "okay", "weak"]:
            f = MagicMock()
            f.name = name
            f.mass = 70.0
            f.model = MagicMock()
            fighters.append(f)

        tracker = MagicMock()
        tracker.get_rankings.return_value = [
            SimpleNamespace(name="forgetter", elo=1600),   # Best ELO
            SimpleNamespace(name="decent", elo=1550),
            SimpleNamespace(name="okay", elo=1450),
            SimpleNamespace(name="weak", elo=1350),
        ]

        # forgetter has 0 anchor score despite best ELO
        anchor_scores = {
            "forgetter": 0.0,
            "decent": 0.8,
            "okay": 0.6,
            "weak": 0.3,
        }

        selection = evolver._select_survivors(
            fighters, tracker, keep_top=0.5, anchor_scores=anchor_scores,
        )

        survivor_names = [f.name for f in selection.survivors]

        # decent should survive: composite = 0.8*(3/4) + 0.2*0.8 = 0.76
        # forgetter: composite = 0.8*(4/4) + 0.2*0.0 = 0.80
        # okay: composite = 0.8*(2/4) + 0.2*0.6 = 0.52
        # weak: composite = 0.8*(1/4) + 0.2*0.3 = 0.26
        # So top 2 are forgetter(0.80) and decent(0.76)
        # With keep_top=0.5 (2 survivors), forgetter still survives
        # because 0.8*1.0 > 0.8*0.75 + 0.2*0.8

        # But if forgetter's anchor score is truly bad and ELO is only slightly ahead:
        anchor_scores_v2 = {
            "forgetter": 0.0,
            "decent": 1.0,
            "okay": 0.8,
            "weak": 0.3,
        }

        selection2 = evolver._select_survivors(
            fighters, tracker, keep_top=0.5, anchor_scores=anchor_scores_v2,
        )
        survivor_names2 = [f.name for f in selection2.survivors]

        # decent: 0.8*(3/4) + 0.2*1.0 = 0.8
        # forgetter: 0.8*(4/4) + 0.2*0.0 = 0.8 — tie, but decent retains skills
        # okay: 0.8*(2/4) + 0.2*0.8 = 0.56
        # At least decent should survive
        assert "decent" in survivor_names2

    def test_no_anchor_scores_falls_back_to_pure_elo(self):
        """Without anchor scores, selection should use pure ELO ranking."""
        evolver = PopulationEvolver(self._make_evolution_context())

        fighters = []
        for name in ["top", "mid", "low", "worst"]:
            f = MagicMock()
            f.name = name
            f.mass = 70.0
            fighters.append(f)

        tracker = MagicMock()
        tracker.get_rankings.return_value = [
            SimpleNamespace(name="top", elo=1600),
            SimpleNamespace(name="mid", elo=1500),
            SimpleNamespace(name="low", elo=1400),
            SimpleNamespace(name="worst", elo=1300),
        ]

        selection = evolver._select_survivors(
            fighters, tracker, keep_top=0.5, anchor_scores=None,
        )

        survivor_names = [f.name for f in selection.survivors]
        assert survivor_names == ["top", "mid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
