"""Guardrail tests for the `src.atom.*` namespace migration."""

from importlib import import_module


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
            "src.training.utils.baseline_harness",
            "src.atom.training.utils.baseline_harness",
        ),
        (
            "src.training.utils.colab_preflight",
            "src.atom.training.utils.colab_preflight",
        ),
        (
            "src.training.utils.determinism",
            "src.atom.training.utils.determinism",
        ),
        (
            "src.coaching",
            "src.atom.coaching",
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
