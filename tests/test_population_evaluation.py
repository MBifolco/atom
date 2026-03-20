"""Focused tests for population evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np

from src.arena import WorldConfig
from src.training.trainers.population.population_evaluation import (
    EvaluationContext,
    PopulationEvaluationService,
)


@dataclass
class _DummyFighter:
    name: str
    mass: float
    model: Mock


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("population_evaluation_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _context(verbose: bool = False) -> EvaluationContext:
    return EvaluationContext(
        config=WorldConfig(),
        max_ticks=50,
        generation=2,
        verbose=verbose,
        logger=_test_logger(),
    )


def test_run_returns_zero_when_no_pairs():
    service = PopulationEvaluationService(_context(verbose=False))
    elo_tracker = Mock()

    matches = service.run(
        population=[],
        elo_tracker=elo_tracker,
        decision_func_factory=lambda fighter: (lambda snapshot: {"acceleration": 0.0, "stance": "neutral"}),
        env_factory=Mock(),
        num_matches_per_pair=1,
    )

    assert matches == 0


def test_run_single_pair_updates_elo_and_returns_match_count():
    service = PopulationEvaluationService(_context(verbose=False))

    model_a = Mock()
    model_a.predict.return_value = (np.array([0.0, 0.0]), None)
    model_b = Mock()
    model_b.predict.return_value = (np.array([0.0, 0.0]), None)
    fighter_a = _DummyFighter(name="a", mass=70.0, model=model_a)
    fighter_b = _DummyFighter(name="b", mass=71.0, model=model_b)

    elo_tracker = Mock()
    elo_tracker.update_ratings.return_value = (1010.0, 990.0)

    mock_env = Mock()
    mock_env.reset.return_value = (np.zeros(9), {})
    mock_env.step.return_value = (
        np.zeros(9),
        1.0,
        True,
        False,
        {"won": True, "episode_damage_dealt": 30.0, "episode_damage_taken": 10.0},
    )
    env_factory = Mock(return_value=mock_env)

    matches = service.run(
        population=[fighter_a, fighter_b],
        elo_tracker=elo_tracker,
        decision_func_factory=lambda fighter: (lambda snapshot: {"acceleration": 0.0, "stance": "neutral"}),
        env_factory=env_factory,
        num_matches_per_pair=1,
    )

    assert matches == 1
    assert elo_tracker.update_ratings.call_count == 1
    mock_env.close.assert_called()
