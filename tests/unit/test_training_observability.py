"""Focused tests for structured training observability helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
from src.atom.training.trainers.population.population_evolution import LineageEvent
from src.atom.training.trainers.population.population_trainer import PopulationFighter, PopulationTrainer
from src.atom.training.utils.observability import append_jsonl, build_run_manifest, write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_observability_helpers_write_manifest_and_jsonl(tmp_path):
    manifest = build_run_manifest(
        repo_root=tmp_path,
        output_dir=tmp_path / "outputs",
        training_config={"seed": 1337, "phase": "complete_pipeline"},
        seed_report={"seed": 1337},
    )

    manifest_path = write_json(tmp_path / "outputs" / "analysis" / "run_manifest.json", manifest)
    append_jsonl(tmp_path / "outputs" / "analysis" / "events.jsonl", {"event": "ok"})

    saved_manifest = json.loads(manifest_path.read_text())
    assert saved_manifest["training"]["phase"] == "complete_pipeline"
    assert saved_manifest["training"]["seed"] == 1337
    assert saved_manifest["runtime"]["python_version"]

    event_records = _read_jsonl(tmp_path / "outputs" / "analysis" / "events.jsonl")
    assert event_records == [{"event": "ok"}]


def test_curriculum_trainer_writes_level_summary_and_failure_event():
    with TemporaryDirectory() as tmpdir:
        trainer = CurriculumTrainer(output_dir=tmpdir, verbose=False)
        trainer.model = SimpleNamespace(num_timesteps=4321)
        trainer.progress.episodes_at_level = 120
        trainer.progress.wins_at_level = 96
        trainer.progress.total_episodes = 120
        trainer.progress.total_wins = 96
        trainer.progress.recent_episodes = [True] * 40 + [False] * 10
        trainer.progress.recent_rewards = [10.0, 20.0, 30.0]
        trainer.progress.recent_reward_breakdowns = [
            {"damage": 5.0, "inaction": -2.0},
            {"damage": 7.0, "inaction": -1.0},
        ]
        trainer._begin_level_observation_window()
        trainer._write_level_summary(end_reason="graduated")
        trainer._record_failure_event(
            event_type="sanity_gate_triggered",
            error_type="RuntimeError",
            message="example failure",
            recovery_action="abort_run",
            recovery_succeeded=False,
        )

        level_records = _read_jsonl(Path(tmpdir) / "analysis" / "level_summaries.jsonl")
        assert len(level_records) == 1
        assert level_records[0]["level_name"] == "Fundamentals"
        assert level_records[0]["end_reason"] == "graduated"
        assert level_records[0]["timesteps_consumed"] == 0
        assert level_records[0]["reward_component_means"]["damage"] == 6.0

        failure_records = _read_jsonl(Path(tmpdir) / "analysis" / "failure_events.jsonl")
        assert len(failure_records) == 1
        assert failure_records[0]["event_type"] == "sanity_gate_triggered"
        assert failure_records[0]["recovery_action"] == "abort_run"


def test_curriculum_trainer_writes_holdout_eval_record(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        trainer = CurriculumTrainer(output_dir=tmpdir, verbose=False)
        trainer.model = SimpleNamespace(num_timesteps=9876)
        trainer.progress.current_level = 1
        trainer.progress.total_episodes = 345
        monkeypatch.setattr(
            trainer,
            "_get_holdout_suite",
            lambda: [
                {
                    "category": "movement",
                    "label": "holdout_a",
                    "opponent_path": "fighters/test_dummies/atomic/approach_slow.py",
                },
                {
                    "category": "expert",
                    "label": "holdout_b",
                    "opponent_path": "fighters/examples/boxer.py",
                },
            ],
        )
        monkeypatch.setattr(
            trainer,
            "_run_holdout_match",
            lambda opponent_path, env_id=0: {
                "won": opponent_path.endswith("approach_slow.py"),
                "damage_dealt": 12.0 + env_id,
                "damage_taken": 3.0,
                "fight_length": 42,
                "reward": 7.5,
            },
        )

        trainer._evaluate_holdout_suite("level_2_basic_skills_graduated")

        holdout_records = _read_jsonl(Path(tmpdir) / "analysis" / "holdout_eval.jsonl")
        assert len(holdout_records) == 1
        assert holdout_records[0]["checkpoint_label"] == "level_2_basic_skills_graduated"
        assert holdout_records[0]["level_name"] == "Basic Skills"
        assert holdout_records[0]["overall_matches"] == 2
        assert holdout_records[0]["overall_wins"] == 1
        assert len(holdout_records[0]["suite_results"]) == 2
        assert holdout_records[0]["suite_results"][0]["opponent"] == "holdout_a"


def test_population_trainer_writes_generation_summary_and_export_failure():
    with TemporaryDirectory() as tmpdir:
        trainer = PopulationTrainer(output_dir=tmpdir, verbose=False, use_vmap=False, population_size=2)
        trainer.population = [
            PopulationFighter(name="Alpha", model=Mock(), generation=0, lineage="founder"),
            PopulationFighter(name="Bravo_G1", model=Mock(), generation=1, lineage="Alpha→Bravo_G1"),
        ]
        trainer.elo_tracker.add_fighter("Alpha")
        trainer.elo_tracker.add_fighter("Bravo_G1")
        trainer.elo_tracker.fighters["Alpha"].elo = 1525.0
        trainer.elo_tracker.fighters["Bravo_G1"].elo = 1490.0
        champion_before = trainer.elo_tracker.fighters["Alpha"]
        champion_after = trainer.elo_tracker.fighters["Alpha"]

        trainer._append_generation_summary(
            generation=1,
            pre_population_names=["Alpha", "Bravo"],
            post_population_names=["Alpha", "Bravo_G1"],
            champion_before=champion_before,
            champion_after=champion_after,
            results=[
                {"fighter": "Alpha", "episodes": 250, "mean_reward": 10.0, "opponent_names": ["Bravo"]},
                {"fighter": "Bravo", "episodes": 250, "mean_reward": 5.0, "opponent_names": ["Alpha"]},
            ],
            episodes_per_fighter=250,
            training_seconds=12.5,
            evaluation_seconds=3.0,
            saving_seconds=1.5,
        )
        trainer._record_export_failure(
            fighter_name="Alpha",
            export_target="fighters/AIs/Alpha",
            exception=RuntimeError("onnx failed"),
            training_artifacts_saved=True,
            win_rate=0.75,
            elo=1525.0,
        )

        generation_records = _read_jsonl(Path(tmpdir) / "analysis" / "generation_summary.jsonl")
        assert len(generation_records) == 1
        assert generation_records[0]["generation"] == 1
        assert generation_records[0]["survivor_count_carried_forward"] == 1
        assert generation_records[0]["child_count_introduced"] == 1
        assert generation_records[0]["champion_before_training"] == "Alpha"

        export_failures = _read_jsonl(Path(tmpdir) / "analysis" / "export_failures.jsonl")
        assert len(export_failures) == 1
        assert export_failures[0]["fighter_name"] == "Alpha"
        assert export_failures[0]["exception_type"] == "RuntimeError"

        trainer._append_lineage_events(
            generation=1,
            lineage_events=[
                LineageEvent(
                    generation=1,
                    child_name="Bravo_G1",
                    child_generation=1,
                    parent_name="Alpha",
                    replaced_fighter_name="Bravo",
                    parent_elo_at_mutation=1525.0,
                    child_mass=72.5,
                    replaced_generation=0,
                )
            ],
            active_population_names=["Alpha", "Bravo_G1"],
        )
        trainer._append_current_leaderboard_snapshot(
            generation=1,
            active_population_names=["Alpha", "Bravo_G1"],
            new_child_names={"Bravo_G1"},
        )

        lineage_records = _read_jsonl(Path(tmpdir) / "analysis" / "lineage_events.jsonl")
        assert len(lineage_records) == 1
        assert lineage_records[0]["child_name"] == "Bravo_G1"
        assert lineage_records[0]["parent_name"] == "Alpha"

        leaderboard_records = _read_jsonl(Path(tmpdir) / "analysis" / "current_leaderboard.jsonl")
        assert len(leaderboard_records) == 2
        assert any(
            record["fighter_name"] == "Bravo_G1" and record["status_in_generation"] == "new_child"
            for record in leaderboard_records
        )
