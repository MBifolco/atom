"""Focused tests for population persistence/export helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.training.trainers.population.population_persistence import (
    PopulationPersistenceContext,
    PopulationPersistenceService,
)


@dataclass
class _DummyFighter:
    name: str
    model: Mock
    generation: int = 0
    lineage: str = "founder"
    mass: float = 70.0
    training_episodes: int = 0
    last_checkpoint: str | None = None


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("population_persistence_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _context(tmp_path: Path) -> PopulationPersistenceContext:
    return PopulationPersistenceContext(
        models_dir=tmp_path / "models",
        project_root=tmp_path / "repo",
        algorithm="ppo",
        population_size=8,
        generation=3,
        verbose=False,
        logger=_test_logger(),
    )


def test_save_generation_models_sets_last_checkpoint(tmp_path):
    context = _context(tmp_path)
    context.models_dir.mkdir(parents=True, exist_ok=True)
    service = PopulationPersistenceService(context)
    generation_dir = service.generation_dir()

    def _save(path: Path):
        path.write_text("model")

    fighter = _DummyFighter(name="fighter_a", model=Mock(save=Mock(side_effect=_save)))

    service.save_generation_models([fighter], generation_dir)

    expected_path = generation_dir / "fighter_a.zip"
    assert expected_path.exists()
    assert fighter.last_checkpoint == str(expected_path)


def test_write_rankings_file_contains_entries(tmp_path):
    context = _context(tmp_path)
    context.models_dir.mkdir(parents=True, exist_ok=True)
    service = PopulationPersistenceService(context)
    generation_dir = service.generation_dir()

    rankings = [
        SimpleNamespace(name="a", elo=1010.0, wins=3, losses=1, draws=0),
        SimpleNamespace(name="b", elo=990.0, wins=1, losses=3, draws=0),
    ]

    rankings_file = service.write_rankings_file(rankings, generation_dir)
    text = rankings_file.read_text()
    assert "Generation 3 Rankings" in text
    assert "a: ELO=1010" in text
    assert "b: ELO=990" in text


def test_compute_win_rate_handles_zero_and_nonzero_matches():
    zero_stats = SimpleNamespace(wins=0, losses=0, draws=0)
    mixed_stats = SimpleNamespace(wins=3, losses=1, draws=2)

    assert PopulationPersistenceService.compute_win_rate(zero_stats) == 0.0
    assert PopulationPersistenceService.compute_win_rate(mixed_stats) == 4.0 / 6.0


def test_export_model_to_onnx_uses_default_obs_shape_when_missing_observation_space(tmp_path):
    context = _context(tmp_path)
    service = PopulationPersistenceService(context)

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(9, 2)

        def forward(self, x):
            return self.net(x)

    model = Mock()
    model.policy = _Policy()
    output_path = tmp_path / "model.onnx"

    with patch("torch.onnx.export") as mock_export:
        service.export_model_to_onnx(model, output_path)

    assert mock_export.call_count == 1
