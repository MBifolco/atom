"""Focused tests for population parallel orchestration components."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from src.arena import WorldConfig
from src.atom.training.trainers.population.parallel_orchestrator import (
    ModelArtifactStore,
    ParallelTrainingContext,
    ParallelTrainingOrchestrator,
    TrainingWorker,
)


@dataclass
class _DummyFighter:
    name: str
    mass: float
    model: Any
    training_episodes: int = 0


class _ImmediateExecutor:
    """Executor test double that resolves tasks synchronously."""

    def __init__(self, max_workers=None, mp_context=None):
        self.max_workers = max_workers
        self.mp_context = mp_context

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # pragma: no cover - defensive branch
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("population_orchestrator_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)
    return logger


def _context(tmp_path: Path) -> ParallelTrainingContext:
    return ParallelTrainingContext(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        config=WorldConfig(),
        max_ticks=50,
        algorithm="ppo",
        n_envs_per_fighter=2,
        n_parallel_fighters=2,
        use_vmap=False,
        n_vmap_envs=8,
        generation=3,
        verbose=False,
        logger=_test_logger(),
    )


def _model(save_side_effect):
    model = Mock()
    model.save = Mock(side_effect=save_side_effect)
    return model


def test_orchestrator_returns_empty_without_executor_use(tmp_path):
    ctx = _context(tmp_path)
    orchestrator = ParallelTrainingOrchestrator(ctx)

    called = {"value": False}

    def _should_not_run(*args, **kwargs):
        called["value"] = True
        return _ImmediateExecutor()

    result = orchestrator.run(
        fighter_opponent_pairs=[],
        episodes_per_fighter=10,
        worker=TrainingWorker(lambda *args: {}),
        executor_factory=_should_not_run,
    )

    assert result == []
    assert called["value"] is False


def test_build_training_tasks_reuses_opponent_artifact_path(tmp_path):
    ctx = _context(tmp_path)
    ctx.models_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = ParallelTrainingOrchestrator(ctx)
    artifacts = ModelArtifactStore(ctx)

    def _save(path: Path):
        path.write_text("model")

    fighter_a = _DummyFighter(name="a", mass=70.0, model=_model(_save))
    fighter_b = _DummyFighter(name="b", mass=71.0, model=_model(_save))

    pairs = [
        (fighter_a, [fighter_b]),
        (fighter_b, [fighter_a]),
    ]

    tasks = orchestrator._build_training_tasks(  # pylint: disable=protected-access
        fighter_opponent_pairs=pairs,
        episodes_per_fighter=12,
        artifacts=artifacts,
    )

    assert len(tasks) == 2
    first_opponent_path = tasks[0][3][0][2]
    second_opponent_path = tasks[1][3][0][2]
    assert first_opponent_path.endswith("temp_b_3.zip")
    assert second_opponent_path.endswith("temp_a_3.zip")
    assert len(artifacts.temp_model_paths) == 2


def test_orchestrator_run_executes_worker_and_returns_results(tmp_path):
    ctx = _context(tmp_path)
    ctx.models_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = ParallelTrainingOrchestrator(ctx)

    def _save(path: Path):
        path.write_text("model")

    fighter_a = _DummyFighter(name="a", mass=70.0, model=_model(_save))
    fighter_b = _DummyFighter(name="b", mass=71.0, model=_model(_save))
    pairs = [(fighter_a, [fighter_b])]

    worker = TrainingWorker(
        lambda *args: {"fighter": args[0], "episodes": 5, "mean_reward": 12.5}
    )

    with patch.object(ModelArtifactStore, "reload_updated_models", autospec=True) as mock_reload:
        results = orchestrator.run(
            fighter_opponent_pairs=pairs,
            episodes_per_fighter=5,
            worker=worker,
            executor_factory=_ImmediateExecutor,
        )

    assert len(results) == 1
    assert results[0]["fighter"] == "a"
    mock_reload.assert_called_once()


def test_model_artifact_store_reload_updates_model_and_episode_count(tmp_path):
    ctx = _context(tmp_path)
    ctx.models_dir.mkdir(parents=True, exist_ok=True)
    store = ModelArtifactStore(ctx)

    fighter = _DummyFighter(name="a", mass=70.0, model=Mock(), training_episodes=10)
    model_path = ctx.models_dir / "temp_a_3.zip"
    model_path.write_text("model")
    store.temp_model_paths["a"] = model_path

    fake_env = Mock()

    with patch("src.atom.training.trainers.population.parallel_orchestrator.DummyVecEnv", return_value=fake_env):
        with patch("src.atom.training.trainers.population.parallel_orchestrator.PPO.load", return_value="loaded-model"):
            store.reload_updated_models([(fighter, [])], episodes_per_fighter=7)

    assert fighter.model == "loaded-model"
    assert fighter.training_episodes == 17
    fake_env.close.assert_called_once()
