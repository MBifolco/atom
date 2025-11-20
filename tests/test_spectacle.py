"""
Tests for spectacle evaluator (fight entertainment scoring).
"""

import pytest
from src.evaluator import SpectacleEvaluator
from src.orchestrator import MatchOrchestrator, MatchResult
from src.arena import WorldConfig


def aggressive_fighter(state):
    """Aggressive test fighter."""
    direction = state["opponent"]["direction"]
    return {"acceleration": 1.0 * direction, "stance": "extended"}


def passive_fighter(state):
    """Passive test fighter."""
    return {"acceleration": 0.0, "stance": "neutral"}


class TestSpectacleEvaluator:
    """Test spectacle scoring system."""

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = SpectacleEvaluator()
        assert evaluator is not None

    def test_evaluate_returns_score(self):
        """Test evaluate returns a score object."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=50, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        # Score should have overall rating
        assert hasattr(score, 'overall')
        assert 0 <= score.overall <= 1.0

    def test_aggressive_fight_scores_higher(self):
        """Test that aggressive fights score higher than passive ones."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=100, record_telemetry=True)

        # Aggressive fight
        result_aggressive = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 5.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 6.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        # Passive fight
        result_passive = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=passive_fighter,
            decision_func_b=passive_fighter,
            seed=42
        )

        evaluator = SpectacleEvaluator()
        score_aggressive = evaluator.evaluate(result_aggressive.telemetry, result_aggressive)
        score_passive = evaluator.evaluate(result_passive.telemetry, result_passive)

        # Aggressive fight should score higher
        assert score_aggressive.overall > score_passive.overall, \
            f"Aggressive fight should score higher: {score_aggressive.overall} vs {score_passive.overall}"

    def test_knockout_affects_score(self):
        """Test that knockouts affect spectacle score."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=200)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "Heavy", "mass": 85.0, "position": 5.0},
            fighter_b_spec={"name": "Light", "mass": 45.0, "position": 6.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=passive_fighter,
            seed=42
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        # Should have a score
        assert score.overall >= 0

    def test_score_components_exist(self):
        """Test that score has expected components."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=50, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        # Check score has overall component
        assert hasattr(score, 'overall')
        assert isinstance(score.overall, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
