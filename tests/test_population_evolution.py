"""Focused tests for population evolution components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import Mock, call, patch

from src.arena import WorldConfig
from src.training.trainers.population.population_evolution import (
    EvolutionContext,
    PopulationEvolver,
)


@dataclass
class _DummyFighter:
    name: str
    mass: float
    model: Any
    last_checkpoint: str | None = None
    generation: int = 0
    lineage: str = "founder"
    training_episodes: int = 0


class _DummyEloTracker:
    def __init__(self, ranking_names: List[str]):
        self._rankings = [SimpleNamespace(name=name) for name in ranking_names]
        self.remove_fighter = Mock()
        self.add_fighter = Mock()

    def get_rankings(self):
        return self._rankings


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("population_evolution_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _context() -> EvolutionContext:
    return EvolutionContext(
        config=WorldConfig(),
        max_ticks=50,
        mass_range=(65.0, 85.0),
        generation=2,
        algorithm="ppo",
        verbose=False,
        logger=_test_logger(),
    )


def test_select_survivors_respects_elo_rank_order():
    evolver = PopulationEvolver(_context())
    population = [
        _DummyFighter("a", 70.0, Mock()),
        _DummyFighter("b", 70.0, Mock()),
        _DummyFighter("c", 70.0, Mock()),
        _DummyFighter("d", 70.0, Mock()),
    ]
    elo_tracker = _DummyEloTracker(["c", "a", "d", "b"])

    selection = evolver._select_survivors(population, elo_tracker, keep_top=0.5)  # pylint: disable=protected-access

    assert [fighter.name for fighter in selection.survivors] == ["c", "a"]
    assert [fighter.name for fighter in selection.to_replace] == ["d", "b"]


def test_evolve_replaces_bottom_fighters_and_updates_elo_tracker():
    evolver = PopulationEvolver(_context())
    population = [
        _DummyFighter("a", 70.0, Mock()),
        _DummyFighter("b", 71.0, Mock()),
        _DummyFighter("c", 72.0, Mock()),
        _DummyFighter("d", 73.0, Mock()),
    ]
    elo_tracker = _DummyEloTracker(["a", "b", "c", "d"])

    with patch.object(evolver, "_clone_and_mutate_model", side_effect=[Mock(), Mock()]):
        evolver.evolve(
            population=population,
            elo_tracker=elo_tracker,
            keep_top=0.5,
            mutation_rate=0.1,
            create_fighter_name=lambda idx, gen: f"child_{idx}_g{gen}",
            fighter_factory=lambda name, model, generation, lineage, mass: _DummyFighter(
                name=name,
                model=model,
                generation=generation,
                lineage=lineage,
                mass=mass,
            ),
        )

    # Top two should survive, bottom two replaced.
    assert population[0].name == "a"
    assert population[1].name == "b"
    assert population[2].name.startswith("child_")
    assert population[3].name.startswith("child_")

    elo_tracker.remove_fighter.assert_has_calls([call("c"), call("d")], any_order=True)
    assert elo_tracker.remove_fighter.call_count == 2
    assert elo_tracker.add_fighter.call_count == 2


def test_load_parent_model_uses_checkpoint_when_available():
    evolver = PopulationEvolver(_context())
    parent = _DummyFighter(
        name="parent",
        mass=70.0,
        model=Mock(save=Mock()),
        last_checkpoint="/tmp/parent_checkpoint.zip",
    )
    env = Mock()

    with patch("src.training.trainers.population.population_evolution.PPO.load", return_value="loaded") as mock_load:
        loaded_model = evolver._load_parent_model(parent=parent, env=env)  # pylint: disable=protected-access

    assert loaded_model == "loaded"
    mock_load.assert_called_once_with("/tmp/parent_checkpoint.zip", env=env)
    parent.model.save.assert_not_called()


def test_load_parent_model_without_checkpoint_saves_temp_copy():
    evolver = PopulationEvolver(_context())

    def _save_model(path: Path):
        path.write_text("model")

    parent = _DummyFighter(
        name="parent",
        mass=70.0,
        model=Mock(save=Mock(side_effect=_save_model)),
        last_checkpoint=None,
    )
    env = Mock()

    with patch("src.training.trainers.population.population_evolution.PPO.load", return_value="loaded") as mock_load:
        loaded_model = evolver._load_parent_model(parent=parent, env=env)  # pylint: disable=protected-access

    assert loaded_model == "loaded"
    parent.model.save.assert_called_once()
    assert mock_load.call_count == 1
