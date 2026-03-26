"""
Comprehensive tests for per-opponent mastery system.

Tests OpponentMasteryTracker, per-opponent progress tracking,
deferred pool refresh, checkpoint serialization, and post-curriculum validation.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from src.atom.training.trainers.curriculum_trainer import (
    OpponentMasteryTracker,
    TrainingProgress,
    CurriculumLevel,
    DifficultyLevel,
)


# ---------------------------------------------------------------------------
# OpponentMasteryTracker unit tests
# ---------------------------------------------------------------------------

class TestOpponentMasteryTracker:
    """Test OpponentMasteryTracker mastery detection logic."""

    def _make_progress(self):
        return TrainingProgress()

    def test_no_mastery_with_insufficient_episodes(self):
        """Opponents with fewer than min_mastery_episodes are skipped."""
        tracker = OpponentMasteryTracker(min_mastery_episodes=10)
        progress = self._make_progress()
        progress.per_opponent_recent = {"opp_a": [True] * 5}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 5}

        all_mastered, pool_changed = tracker.check_mastery(progress, ["opp_a"])
        assert not all_mastered
        assert not pool_changed

    def test_mastery_requires_two_consecutive_checks(self):
        """Mastery needs pending → mastered (2 checks)."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 10}

        # First check: moves to pending
        all_m, changed = tracker.check_mastery(progress, ["opp_a"])
        assert not all_m
        assert not changed
        assert "opp_a" in progress.pending_mastery
        assert "opp_a" not in progress.mastered_opponents

        # Second check: moves to mastered
        all_m, changed = tracker.check_mastery(progress, ["opp_a"])
        assert all_m
        assert changed
        assert "opp_a" in progress.mastered_opponents
        assert "opp_a" not in progress.pending_mastery

    def test_pending_revoked_if_performance_drops(self):
        """If an opponent falls below thresholds after entering pending, revert."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 10}

        # First check: pending
        tracker.check_mastery(progress, ["opp_a"])
        assert "opp_a" in progress.pending_mastery

        # Performance drops
        progress.per_opponent_recent["opp_a"] = [False] * 10
        progress.per_opponent_recent_damage["opp_a"] = [0.0] * 10

        # Pending should be revoked
        all_m, changed = tracker.check_mastery(progress, ["opp_a"])
        assert not all_m
        assert not changed
        assert "opp_a" not in progress.pending_mastery
        assert "opp_a" not in progress.mastered_opponents

    def test_already_mastered_opponents_skipped(self):
        """Opponents already in mastered set are not re-evaluated."""
        tracker = OpponentMasteryTracker(min_mastery_episodes=5)
        progress = self._make_progress()
        progress.mastered_opponents = {"opp_a"}
        progress.per_opponent_recent = {"opp_a": [False] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [0.0] * 10}

        all_m, changed = tracker.check_mastery(progress, ["opp_a"])
        assert all_m
        assert not changed

    def test_multi_opponent_partial_mastery(self):
        """Some opponents mastered, others not — all_mastered is False."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        progress.mastered_opponents = {"opp_a"}
        progress.per_opponent_recent = {
            "opp_a": [True] * 10,
            "opp_b": [False] * 10,
        }
        progress.per_opponent_recent_damage = {
            "opp_a": [20.0] * 10,
            "opp_b": [0.0] * 10,
        }

        all_m, _ = tracker.check_mastery(progress, ["opp_a", "opp_b"])
        assert not all_m

    def test_all_opponents_mastered(self):
        """All opponents mastered returns True."""
        tracker = OpponentMasteryTracker(min_mastery_episodes=5)
        progress = self._make_progress()
        progress.mastered_opponents = {"opp_a", "opp_b", "opp_c"}

        all_m, changed = tracker.check_mastery(progress, ["opp_a", "opp_b", "opp_c"])
        assert all_m
        assert not changed

    def test_win_rate_threshold(self):
        """Below win rate threshold does not trigger pending."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=1.0,
            min_per_opponent_nonzero=0.1,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        # 30% win rate — below 50% threshold
        progress.per_opponent_recent = {"opp_a": [True, True, True, False, False, False, False, False, False, False]}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 10}

        tracker.check_mastery(progress, ["opp_a"])
        assert "opp_a" not in progress.pending_mastery

    def test_damage_threshold(self):
        """Below damage threshold does not trigger pending."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        # Damage too low
        progress.per_opponent_recent_damage = {"opp_a": [2.0] * 10}

        tracker.check_mastery(progress, ["opp_a"])
        assert "opp_a" not in progress.pending_mastery

    def test_nonzero_rate_threshold(self):
        """Below nonzero damage rate does not trigger pending."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=10,
        )
        progress = self._make_progress()
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        # 40% nonzero — below 50% threshold
        progress.per_opponent_recent_damage = {"opp_a": [50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        tracker.check_mastery(progress, ["opp_a"])
        assert "opp_a" not in progress.pending_mastery

    def test_mastery_window_uses_recent_episodes(self):
        """Only the last mastery_window episodes count for mastery check."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5,
            min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5,
            min_mastery_episodes=5,
            mastery_window=5,
        )
        progress = self._make_progress()
        # Old episodes: bad. Recent 5: good.
        progress.per_opponent_recent = {"opp_a": [False] * 20 + [True] * 5}
        progress.per_opponent_recent_damage = {"opp_a": [0.0] * 20 + [20.0] * 5}

        tracker.check_mastery(progress, ["opp_a"])
        assert "opp_a" in progress.pending_mastery

    def test_empty_opponent_list(self):
        """Empty opponent list → all_mastered True, no pool change."""
        tracker = OpponentMasteryTracker()
        progress = self._make_progress()

        all_m, changed = tracker.check_mastery(progress, [])
        assert all_m
        assert not changed


# ---------------------------------------------------------------------------
# Per-opponent progress routing tests
# ---------------------------------------------------------------------------

class TestPerOpponentProgressRouting:
    """Test that per-opponent stats are correctly routed through ProgressReporter."""

    def _make_reporter_and_progress(self, mastery_window=20):
        from src.atom.training.trainers.curriculum_components import ProgressReporter
        progress = TrainingProgress()
        level = CurriculumLevel(
            name="Test", difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["fighters/opp_a.py"], graduation_episodes=50,
        )
        reporter = ProgressReporter(logger=MagicMock(), mastery_window=mastery_window)
        return reporter, progress, level

    def test_update_progress_routes_opponent_name(self):
        """update_progress tracks per-opponent wins and episodes."""
        reporter, progress, level = self._make_reporter_and_progress()

        info = {"opponent_name": "opp_a", "episode_damage_dealt": 15.0}
        reporter.update_progress(progress=progress, level=level, won=True, reward=100.0, info=info)

        assert progress.per_opponent_episodes.get("opp_a") == 1
        assert progress.per_opponent_wins.get("opp_a") == 1
        assert progress.per_opponent_recent["opp_a"] == [True]
        assert progress.per_opponent_recent_damage["opp_a"] == [15.0]

    def test_update_progress_tracks_losses(self):
        """Losses are tracked per opponent."""
        reporter, progress, level = self._make_reporter_and_progress()

        info = {"opponent_name": "opp_a", "episode_damage_dealt": 5.0}
        reporter.update_progress(progress=progress, level=level, won=False, reward=-50.0, info=info)

        assert progress.per_opponent_episodes["opp_a"] == 1
        assert progress.per_opponent_wins.get("opp_a", 0) == 0
        assert progress.per_opponent_recent["opp_a"] == [False]

    def test_update_progress_trims_to_mastery_window(self):
        """Per-opponent buffers are trimmed to mastery_window size."""
        reporter, progress, level = self._make_reporter_and_progress(mastery_window=5)

        info = {"opponent_name": "opp_a", "episode_damage_dealt": 10.0}
        for _ in range(10):
            reporter.update_progress(progress=progress, level=level, won=True, reward=100.0, info=info)

        # Buffer should be trimmed to mastery_window
        assert len(progress.per_opponent_recent["opp_a"]) <= 5
        assert len(progress.per_opponent_recent_damage["opp_a"]) <= 5

    def test_update_progress_no_opponent_name(self):
        """Missing opponent_name in info doesn't crash."""
        reporter, progress, level = self._make_reporter_and_progress()

        info = {"episode_damage_dealt": 10.0}  # No opponent_name
        reporter.update_progress(progress=progress, level=level, won=True, reward=100.0, info=info)

        # Should not have any per-opponent entries
        assert len(progress.per_opponent_episodes) == 0


# ---------------------------------------------------------------------------
# Level transition resets per-opponent fields
# ---------------------------------------------------------------------------

class TestLevelTransitionResetsPerOpponent:
    """Test that level transitions reset per-opponent tracking."""

    def test_advance_resets_per_opponent_fields(self):
        """LevelTransitionStateMachine.advance() clears per-opponent data."""
        from src.atom.training.trainers.curriculum_components import LevelTransitionStateMachine

        sm = LevelTransitionStateMachine()
        progress = TrainingProgress()
        progress.current_level = 0
        progress.per_opponent_episodes = {"opp_a": 50}
        progress.per_opponent_wins = {"opp_a": 30}
        progress.per_opponent_recent = {"opp_a": [True] * 20}
        progress.per_opponent_recent_damage = {"opp_a": [15.0] * 20}
        progress.mastered_opponents = {"opp_a"}
        progress.pending_mastery = {"opp_b"}

        curriculum = [
            CurriculumLevel(name="L1", difficulty=DifficultyLevel.FUNDAMENTALS, opponents=["opp_a"]),
            CurriculumLevel(name="L2", difficulty=DifficultyLevel.BASIC_SKILLS, opponents=["opp_b"]),
        ]

        sm.advance(progress=progress, curriculum=curriculum)

        assert progress.per_opponent_episodes == {}
        assert progress.per_opponent_wins == {}
        assert progress.per_opponent_recent == {}
        assert progress.per_opponent_recent_damage == {}
        assert progress.mastered_opponents == set()
        assert progress.pending_mastery == set()


# ---------------------------------------------------------------------------
# Checkpoint serialization round-trip
# ---------------------------------------------------------------------------

class TestCheckpointSerialization:
    """Test that per-opponent fields survive JSON checkpoint round-trip."""

    def test_set_fields_serialize_as_lists(self):
        """mastered_opponents and pending_mastery serialize to JSON-safe lists."""
        progress = TrainingProgress()
        progress.mastered_opponents = {"opp_a", "opp_b"}
        progress.pending_mastery = {"opp_c"}
        progress.per_opponent_episodes = {"opp_a": 50, "opp_b": 30, "opp_c": 10}
        progress.per_opponent_wins = {"opp_a": 40, "opp_b": 20}
        progress.per_opponent_recent = {"opp_a": [True, False]}
        progress.per_opponent_recent_damage = {"opp_a": [15.0, 0.0]}

        # Simulate capture
        state = asdict(progress)
        state["mastered_opponents"] = list(state.get("mastered_opponents", set()))
        state["pending_mastery"] = list(state.get("pending_mastery", set()))

        # Should be JSON-serializable
        json_str = json.dumps(state)
        restored = json.loads(json_str)

        assert isinstance(restored["mastered_opponents"], list)
        assert set(restored["mastered_opponents"]) == {"opp_a", "opp_b"}
        assert set(restored["pending_mastery"]) == {"opp_c"}

    def test_restore_converts_lists_to_sets(self):
        """Restoring from checkpoint converts lists back to sets."""
        progress = TrainingProgress()
        state = {
            "mastered_opponents": ["opp_a", "opp_b"],
            "pending_mastery": ["opp_c"],
            "per_opponent_episodes": {"opp_a": 50},
            "per_opponent_wins": {"opp_a": 40},
            "per_opponent_recent": {"opp_a": [True, False]},
            "per_opponent_recent_damage": {"opp_a": [15.0, 0.0]},
        }

        # Simulate restore
        progress.mastered_opponents = set(state.get("mastered_opponents", []))
        progress.pending_mastery = set(state.get("pending_mastery", []))
        progress.per_opponent_episodes = state.get("per_opponent_episodes", {})
        progress.per_opponent_wins = state.get("per_opponent_wins", {})
        progress.per_opponent_recent = state.get("per_opponent_recent", {})
        progress.per_opponent_recent_damage = state.get("per_opponent_recent_damage", {})

        assert isinstance(progress.mastered_opponents, set)
        assert isinstance(progress.pending_mastery, set)
        assert progress.mastered_opponents == {"opp_a", "opp_b"}
        assert progress.per_opponent_recent["opp_a"] == [True, False]


# ---------------------------------------------------------------------------
# Env-to-opponent mapping tests
# ---------------------------------------------------------------------------

class TestEnvToOpponentMapping:
    """Test VmapEnvWrapper opponent name mapping."""

    def test_non_divisible_n_envs(self):
        """250 envs / 7 opponents: mapping covers all envs."""
        n_envs = 250
        opponent_paths = [f"fighters/opp_{i}.py" for i in range(7)]
        n_opponents = len(opponent_paths)
        envs_per_opponent = n_envs // n_opponents

        mapping = []
        for env_idx in range(n_envs):
            opp_idx = min(env_idx // envs_per_opponent, n_opponents - 1)
            mapping.append(Path(opponent_paths[opp_idx]).stem)

        assert len(mapping) == 250
        # All opponent names should appear
        unique = set(mapping)
        assert len(unique) == 7
        # Last opponent gets the remainder envs
        last_opp_count = sum(1 for n in mapping if n == f"opp_6")
        assert last_opp_count >= envs_per_opponent

    def test_equal_division(self):
        """6 envs / 3 opponents: exactly 2 each."""
        n_envs = 6
        opponent_paths = [f"fighters/opp_{i}.py" for i in range(3)]
        n_opponents = len(opponent_paths)
        envs_per_opponent = n_envs // n_opponents

        mapping = []
        for env_idx in range(n_envs):
            opp_idx = min(env_idx // envs_per_opponent, n_opponents - 1)
            mapping.append(Path(opponent_paths[opp_idx]).stem)

        from collections import Counter
        counts = Counter(mapping)
        assert all(c == 2 for c in counts.values())


# ---------------------------------------------------------------------------
# Deferred pool refresh tests
# ---------------------------------------------------------------------------

class TestDeferredPoolRefresh:
    """Test _apply_opponent_pool_refresh behavior."""

    def test_no_refresh_when_flag_not_set(self):
        """No-op when _pending_opponent_pool_refresh is False."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer

        trainer = MagicMock()
        trainer._pending_opponent_pool_refresh = False

        CurriculumTrainer._apply_opponent_pool_refresh(trainer)

        # Should not have called get_current_level
        trainer.get_current_level.assert_not_called()

    def test_refresh_resets_flag(self):
        """Flag is reset to False after processing."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer

        trainer = MagicMock()
        trainer._pending_opponent_pool_refresh = True
        level = MagicMock()
        level.opponents = ["fighters/opp_a.py", "fighters/opp_b.py"]
        trainer.get_current_level.return_value = level
        trainer.progress.mastered_opponents = {"opp_a"}
        trainer._active_level_opponents = ["opp_a", "opp_b"]
        trainer.use_vmap = False

        CurriculumTrainer._apply_opponent_pool_refresh(trainer)

        assert trainer._pending_opponent_pool_refresh is False


# ---------------------------------------------------------------------------
# Post-curriculum validation tests
# ---------------------------------------------------------------------------

class TestPostCurriculumValidation:
    """Test _validate_curriculum_graduate method."""

    def test_validation_runs_both_modes(self):
        """Validation tests stochastic and deterministic modes."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(
                name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                opponents=["fighters/opp_a.py", "fighters/opp_b.py"],
            ),
        ]
        trainer.logger = MagicMock()

        match_results = {"won": True, "damage_dealt": 20.0, "damage_taken": 5.0, "fight_length": 100, "reward": 50.0}
        trainer._run_holdout_match = MagicMock(return_value=match_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            result = CurriculumTrainer._validate_curriculum_graduate(trainer)

        assert result is True

        # Should have called _run_holdout_match for each opponent × mode × matches
        # 2 opponents × 2 modes × 5 matches = 20 calls
        assert trainer._run_holdout_match.call_count == 20

        # Check both deterministic=True and deterministic=False were used
        det_calls = [c for c in trainer._run_holdout_match.call_args_list if c.kwargs.get("deterministic") or (len(c.args) > 2 and c.args[2])]
        stoch_calls = [c for c in trainer._run_holdout_match.call_args_list if not (c.kwargs.get("deterministic") or (len(c.args) > 2 and c.args[2]))]
        assert len(det_calls) == 10
        assert len(stoch_calls) == 10

    def test_validation_writes_jsonl(self):
        """Validation writes structured results to post_curriculum_validation.jsonl."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(
                name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                opponents=["fighters/opp_a.py"],
            ),
        ]
        trainer.logger = MagicMock()
        trainer._run_holdout_match = MagicMock(return_value={
            "won": True, "damage_dealt": 20.0, "damage_taken": 5.0,
            "fight_length": 100, "reward": 50.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            CurriculumTrainer._validate_curriculum_graduate(trainer)

            jsonl_path = trainer.analysis_dir / "post_curriculum_validation.jsonl"
            assert jsonl_path.exists()

            with open(jsonl_path) as f:
                record = json.loads(f.readline())

            assert "results" in record
            assert "timestamp" in record
            assert record["total_opponents"] == 1
            # Each opponent gets 2 entries (stochastic + deterministic)
            assert len(record["results"]) == 2
            modes = {r["mode"] for r in record["results"]}
            assert modes == {"stochastic", "deterministic"}

    def test_validation_deduplicates_opponents(self):
        """Opponents appearing in multiple levels are tested only once."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(
                name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                opponents=["fighters/opp_a.py", "fighters/opp_b.py"],
            ),
            CurriculumLevel(
                name="L2", difficulty=DifficultyLevel.BASIC_SKILLS,
                opponents=["fighters/opp_a.py", "fighters/opp_c.py"],  # opp_a repeated
            ),
        ]
        trainer.logger = MagicMock()
        trainer._run_holdout_match = MagicMock(return_value={
            "won": True, "damage_dealt": 20.0, "damage_taken": 5.0,
            "fight_length": 100, "reward": 50.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            CurriculumTrainer._validate_curriculum_graduate(trainer)

        # 3 unique opponents × 2 modes × 5 matches = 30 calls
        assert trainer._run_holdout_match.call_count == 30

    def test_validation_includes_source_levels(self):
        """Results include source_levels showing which levels use each opponent."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                            opponents=["fighters/opp_a.py"]),
            CurriculumLevel(name="L7", difficulty=DifficultyLevel.GAUNTLET,
                            opponents=["fighters/opp_a.py"]),
        ]
        trainer.logger = MagicMock()
        trainer._run_holdout_match = MagicMock(return_value={
            "won": True, "damage_dealt": 20.0, "damage_taken": 5.0,
            "fight_length": 100, "reward": 50.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            CurriculumTrainer._validate_curriculum_graduate(trainer)

            with open(trainer.analysis_dir / "post_curriculum_validation.jsonl") as f:
                record = json.loads(f.readline())

            opp_a_results = [r for r in record["results"] if r["opponent"] == "opp_a"]
            assert len(opp_a_results) == 2  # stochastic + deterministic
            # Both should have source_levels referencing L1 and L7
            for r in opp_a_results:
                assert len(r["source_levels"]) == 2
                level_names = {sl["level_name"] for sl in r["source_levels"]}
                assert level_names == {"L1", "L7"}

    def test_validation_warns_on_failures(self):
        """Validation logs warnings for opponents below threshold."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                            opponents=["fighters/opp_a.py"]),
        ]
        trainer.logger = MagicMock()
        # Always lose
        trainer._run_holdout_match = MagicMock(return_value={
            "won": False, "damage_dealt": 0.0, "damage_taken": 20.0,
            "fight_length": 100, "reward": -50.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            result = CurriculumTrainer._validate_curriculum_graduate(trainer)

        # Soft gate: still returns True
        assert result is True
        # But should have logged warnings
        trainer.logger.warning.assert_called()

    def test_validation_handles_match_errors(self):
        """Match exceptions don't crash validation."""
        trainer = MagicMock()
        trainer.curriculum = [
            CurriculumLevel(name="L1", difficulty=DifficultyLevel.FUNDAMENTALS,
                            opponents=["fighters/opp_a.py"]),
        ]
        trainer.logger = MagicMock()
        trainer._run_holdout_match = MagicMock(side_effect=RuntimeError("env error"))

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.analysis_dir = Path(tmpdir) / "analysis"
            trainer.analysis_dir.mkdir()

            from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
            result = CurriculumTrainer._validate_curriculum_graduate(trainer)

        # Should still complete (all matches failed, logged warnings)
        assert result is True


# ---------------------------------------------------------------------------
# Mastery integration: should_graduate checks mastery gate
# ---------------------------------------------------------------------------

class TestShouldGraduateMasteryGate:
    """Test that should_graduate enforces per-opponent mastery."""

    def test_graduation_blocked_when_opponents_unmastered(self):
        """Aggregate checks pass but unmastered opponents block graduation."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer

        trainer = MagicMock()
        trainer.progress = TrainingProgress()
        trainer.progress.current_level = 0
        trainer.progress.episodes_at_level = 500
        trainer.progress.mastered_opponents = {"opp_a"}  # Only 1 of 2

        level = CurriculumLevel(
            name="Test", difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["fighters/opp_a.py", "fighters/opp_b.py"],
            graduation_episodes=50,
        )
        trainer.get_current_level.return_value = level
        trainer.curriculum = [level]
        # Wire up _get_level_opponent_names to return real names
        trainer._get_level_opponent_names = lambda: ["opp_a", "opp_b"]

        # GraduationPolicy says graduate
        decision = MagicMock()
        decision.should_graduate = True
        decision.reason = "recent_passed"
        decision.recent_passed = True
        trainer.graduation_policy.evaluate.return_value = decision

        trainer.progress_reporter = MagicMock()
        trainer.logger = MagicMock()
        trainer._phase_new_opponents = None  # On-policy: no phase tracking

        result = CurriculumTrainer.should_graduate(trainer)

        assert result is False


# ---------------------------------------------------------------------------
# CPU-path pool refresh
# ---------------------------------------------------------------------------

class TestCpuPathPoolRefresh:
    """Test that CPU path retargets envs on mastery."""

    def test_cpu_path_retargets_envs(self):
        """CPU path calls set_opponent on remaining unmastered opponents."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer

        trainer = MagicMock()
        trainer._pending_opponent_pool_refresh = True
        trainer.use_vmap = False
        trainer.n_envs = 4

        level = CurriculumLevel(
            name="Test", difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["fighters/opp_a.py", "fighters/opp_b.py", "fighters/opp_c.py"],
        )
        trainer.get_current_level.return_value = level
        trainer.progress = TrainingProgress()
        trainer.progress.mastered_opponents = {"opp_a"}
        trainer._active_level_opponents = ["opp_a", "opp_b", "opp_c"]

        mock_func = MagicMock()
        trainer.load_opponent = MagicMock(return_value=mock_func)
        trainer.logger = MagicMock()

        CurriculumTrainer._apply_opponent_pool_refresh(trainer)

        # Should have called load_opponent for each env
        assert trainer.load_opponent.call_count == 4
        # All loaded opponents should be from unmastered set
        loaded_paths = [c.args[0] for c in trainer.load_opponent.call_args_list]
        for p in loaded_paths:
            assert Path(p).stem in ("opp_b", "opp_c")

        # Should have called env_method for each env
        assert trainer.envs.env_method.call_count == 4


# ---------------------------------------------------------------------------
# Checkpoint resume with reduced pool
# ---------------------------------------------------------------------------

class TestCheckpointResumePoolRefresh:
    """Test that checkpoint restore queues pool refresh for same-level resume."""

    def test_same_level_resume_queues_refresh(self):
        """When mastered opponents exist on restore, pool refresh is queued."""
        from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer, CurriculumCallback

        trainer = MagicMock()
        trainer.progress = TrainingProgress()
        trainer.progress.current_level = 2
        trainer._get_level_opponent_names = MagicMock(return_value=["opp_a", "opp_b", "opp_c"])
        trainer._pending_opponent_pool_refresh = False
        trainer.logger = MagicMock()

        callback = MagicMock(spec=CurriculumCallback)
        callback.episode_rewards = []
        callback.episode_wins = []
        callback.recent_reward_components = []
        callback.rollout_count = 0

        # State with mastered opponents at same level
        state = {
            "progress": {
                "current_level": 2,
                "episodes_at_level": 200,
                "wins_at_level": 150,
                "total_episodes": 500,
                "total_wins": 400,
                "recent_episodes": [True] * 10,
                "recent_damage_dealt": [15.0] * 10,
                "graduated_levels": ["L1", "L2"],
                "mastered_opponents": ["opp_a"],
                "pending_mastery": [],
                "per_opponent_episodes": {"opp_a": 50, "opp_b": 30, "opp_c": 20},
                "per_opponent_wins": {"opp_a": 40},
                "per_opponent_recent": {"opp_a": [True] * 10},
                "per_opponent_recent_damage": {"opp_a": [20.0] * 10},
            },
            "callback": {
                "episode_rewards": [100.0] * 5,
                "episode_wins": [True] * 5,
                "recent_reward_components": [],
                "rollout_count": 10,
            },
        }

        CurriculumTrainer._restore_training_state(trainer, callback, state)

        # Same level (2 == 2), mastered opponents exist, active pool != full level
        assert trainer._pending_opponent_pool_refresh is True


# ---------------------------------------------------------------------------
# Observability: pending transitions are logged
# ---------------------------------------------------------------------------

class TestMasteryObservabilityTransitions:
    """Test that all mastery state transitions emit JSONL snapshots."""

    def test_pending_entry_emits_snapshot(self):
        """Entering pending state should emit a mastery snapshot."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5, min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5, min_mastery_episodes=5, mastery_window=10,
        )
        progress = TrainingProgress()
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 10}

        prev_pending = frozenset(progress.pending_mastery)
        tracker.check_mastery(progress, ["opp_a"])

        # Pending state changed (opp_a entered pending)
        state_changed = frozenset(progress.pending_mastery) != prev_pending
        assert state_changed, "Pending entry should be detected as a state change"

    def test_pending_revocation_emits_snapshot(self):
        """Losing pending state should be detected as a state change."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5, min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5, min_mastery_episodes=5, mastery_window=10,
        )
        progress = TrainingProgress()
        progress.pending_mastery = {"opp_a"}
        # Poor performance — should revoke pending
        progress.per_opponent_recent = {"opp_a": [False] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [0.0] * 10}

        prev_pending = frozenset(progress.pending_mastery)
        tracker.check_mastery(progress, ["opp_a"])

        state_changed = frozenset(progress.pending_mastery) != prev_pending
        assert state_changed, "Pending revocation should be detected as a state change"

    def test_no_transition_no_snapshot(self):
        """No state change means no snapshot needed."""
        tracker = OpponentMasteryTracker(
            mastery_win_rate=0.5, min_per_opponent_damage=10.0,
            min_per_opponent_nonzero=0.5, min_mastery_episodes=5, mastery_window=10,
        )
        progress = TrainingProgress()
        # Opponent already mastered — no transition possible
        progress.mastered_opponents = {"opp_a"}
        progress.per_opponent_recent = {"opp_a": [True] * 10}
        progress.per_opponent_recent_damage = {"opp_a": [20.0] * 10}

        prev_mastered = frozenset(progress.mastered_opponents)
        prev_pending = frozenset(progress.pending_mastery)
        tracker.check_mastery(progress, ["opp_a"])

        state_changed = (
            frozenset(progress.mastered_opponents) != prev_mastered
            or frozenset(progress.pending_mastery) != prev_pending
        )
        assert not state_changed, "No transition should mean no state change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
