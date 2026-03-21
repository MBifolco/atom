"""Guardrail tests for the `src.atom.*` namespace migration."""

from importlib import import_module

from pathlib import Path


def test_legacy_runtime_and_training_modules_alias_new_modules():
    module_pairs = [
        (
            "src.training.replay_recorder",
            "src.atom.training.replay_recorder",
        ),
        (
            "src.training.trainers.population.population_trainer",
            "src.atom.training.trainers.population.population_trainer",
        ),
        (
            "src.training.trainers.curriculum_trainer",
            "src.atom.training.trainers.curriculum_trainer",
        ),
        (
            "src.training.utils.colab_preflight",
            "src.atom.training.utils.colab_preflight",
        ),
    ]

    for legacy_name, new_name in module_pairs:
        legacy_module = import_module(legacy_name)
        new_module = import_module(new_name)
        assert legacy_module is new_module, f"{legacy_name} should alias {new_name}"


def test_patching_legacy_module_updates_new_replay_recorder(monkeypatch):
    legacy_module = import_module("src.training.replay_recorder")
    new_module = import_module("src.atom.training.replay_recorder")
    sentinel = object()

    monkeypatch.setattr(legacy_module, "save_replay", sentinel)

    assert new_module.save_replay is sentinel


def test_patching_legacy_module_updates_new_population_trainer(monkeypatch):
    legacy_module = import_module("src.training.trainers.population.population_trainer")
    new_module = import_module("src.atom.training.trainers.population.population_trainer")
    sentinel = object()

    monkeypatch.setattr(legacy_module, "AtomCombatEnv", sentinel)

    assert new_module.AtomCombatEnv is sentinel


def test_patching_legacy_module_updates_new_curriculum_trainer(monkeypatch):
    legacy_module = import_module("src.training.trainers.curriculum_trainer")
    new_module = import_module("src.atom.training.trainers.curriculum_trainer")
    sentinel = object()

    monkeypatch.setattr(legacy_module, "PPO", sentinel)

    assert new_module.PPO is sentinel


def test_selected_legacy_utility_wrappers_are_retired():
    repo_root = Path(__file__).resolve().parents[2]
    retired_paths = [
        repo_root / "src" / "training" / "utils" / "baseline_harness.py",
        repo_root / "src" / "training" / "utils" / "determinism.py",
        repo_root / "src" / "training" / "utils" / "nan_detector.py",
        repo_root / "src" / "training" / "utils" / "stable_ppo.py",
        repo_root / "src" / "training" / "utils" / "stable_ppo_config.py",
        repo_root / "src" / "training" / "trainers" / "population" / "elo_tracker.py",
        repo_root / "src" / "training" / "trainers" / "population" / "fighter_loader.py",
        repo_root / "src" / "training" / "trainers" / "population" / "parallel_orchestrator.py",
        repo_root / "src" / "training" / "trainers" / "population" / "population_evaluation.py",
        repo_root / "src" / "training" / "trainers" / "population" / "population_evolution.py",
        repo_root / "src" / "training" / "trainers" / "population" / "population_persistence.py",
        repo_root / "src" / "training" / "trainers" / "population" / "population_protocols.py",
        repo_root / "src" / "training" / "trainers" / "population" / "population_training_loop.py",
        repo_root / "src" / "evaluator" / "spectacle_evaluator.py",
        repo_root / "src" / "renderer" / "ascii_renderer.py",
        repo_root / "src" / "renderer" / "html_renderer.py",
        repo_root / "src" / "telemetry" / "replay_store.py",
        repo_root / "src" / "training" / "trainers" / "population" / "debug_gym_env.py",
        repo_root / "src" / "training" / "trainers" / "population" / "test_fighter_loading.py",
        repo_root / "src" / "training" / "trainers" / "population" / "test_single_match.py",
        repo_root / "src" / "training" / "trainers" / "population" / "train_multicore.py",
        repo_root / "src" / "training" / "trainers" / "population" / "train_population.py",
        repo_root / "src" / "training" / "trainers" / "population" / "train_population_multi.py",
        repo_root / "src" / "training" / "pipelines" / "progressive_trainer.py",
        repo_root / "src" / "registry" / "fighter_registry.py",
        repo_root / "src" / "coaching" / "coaching_wrapper.py",
    ]

    for path in retired_paths:
        assert not path.exists(), f"Retired wrapper still present: {path}"


def test_package_level_coaching_exports_match_new_namespace():
    legacy_module = import_module("src.coaching")
    new_module = import_module("src.atom.coaching")

    assert legacy_module.CoachingWrapper is new_module.CoachingWrapper
