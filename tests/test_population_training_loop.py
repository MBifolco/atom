"""Focused tests for population training loop helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

from src.atom.training.trainers.population.population_training_loop import (
    PopulationTrainingLoopContext,
    PopulationTrainingLoopHelper,
)


@dataclass
class _DummyFighter:
    name: str


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("population_training_loop_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _context(verbose: bool = False) -> PopulationTrainingLoopContext:
    return PopulationTrainingLoopContext(
        population_size=4,
        generations=5,
        episodes_per_generation=500,
        evolution_frequency=2,
        keep_top=0.5,
        mutation_rate=0.1,
        replay_recording_frequency=3,
        replay_matches_per_pair=2,
        verbose=verbose,
        logger=_test_logger(),
    )


def test_build_fighter_opponent_pairs_uses_match_pairs():
    helper = PopulationTrainingLoopHelper(_context())
    a = _DummyFighter("a")
    b = _DummyFighter("b")
    c = _DummyFighter("c")
    d = _DummyFighter("d")
    population = [a, b, c, d]
    pairs = [(a, b), (c, d)]

    fighter_pairs = helper.build_fighter_opponent_pairs(population, pairs)

    lookup = {fighter.name: opponents for fighter, opponents in fighter_pairs}
    assert lookup["a"] == [b]
    assert lookup["b"] == [a]
    assert lookup["c"] == [d]
    assert lookup["d"] == [c]


def test_should_evolve_obeys_frequency_and_not_last_generation():
    helper = PopulationTrainingLoopHelper(_context())

    assert helper.should_evolve(0) is False
    assert helper.should_evolve(1) is True
    assert helper.should_evolve(3) is True
    assert helper.should_evolve(4) is False


def test_print_and_log_final_report_updates_logger_and_replay_index():
    helper = PopulationTrainingLoopHelper(_context(verbose=False))
    elo_tracker = Mock()
    elo_tracker.get_diversity_metrics.return_value = {"elo_range": 120.0, "elo_std": 25.0}
    replay_recorder = Mock()

    helper.print_and_log_final_report(
        generation=3,
        total_matches=12,
        elo_tracker=elo_tracker,
        replay_recorder=replay_recorder,
    )

    elo_tracker.get_diversity_metrics.assert_called_once()
    replay_recorder.save_replay_index.assert_called_once()
