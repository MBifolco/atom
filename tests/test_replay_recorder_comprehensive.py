"""
Comprehensive tests for ReplayRecorder to increase coverage.
Tests initialization, data conversion, and replay index management.
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.atom.training.replay_recorder import ReplayRecorder, ReplayMetadata
from src.arena import WorldConfig


class TestReplayMetadata:
    """Tests for ReplayMetadata dataclass."""

    def test_create_metadata_with_required_fields(self):
        """Test creating metadata with required fields."""
        metadata = ReplayMetadata(
            stage="curriculum_level_1",
            stage_type="curriculum",
            spectacle_score=0.75,
            spectacle_rank="top",
            fighter_a="AI_Fighter",
            fighter_b="Boxer",
            winner="AI_Fighter"
        )
        assert metadata.stage == "curriculum_level_1"
        assert metadata.stage_type == "curriculum"
        assert metadata.spectacle_score == 0.75
        assert metadata.spectacle_rank == "top"
        assert metadata.notes == ""

    def test_create_metadata_with_notes(self):
        """Test creating metadata with notes."""
        metadata = ReplayMetadata(
            stage="population_gen_5",
            stage_type="population",
            spectacle_score=0.5,
            spectacle_rank="middle",
            fighter_a="Alpha",
            fighter_b="Beta",
            winner="Alpha",
            notes="Great match with lots of action"
        )
        assert metadata.notes == "Great match with lots of action"

    def test_metadata_default_notes(self):
        """Test that notes defaults to empty string."""
        metadata = ReplayMetadata(
            stage="test",
            stage_type="test",
            spectacle_score=0.0,
            spectacle_rank="bottom",
            fighter_a="A",
            fighter_b="B",
            winner="A"
        )
        assert metadata.notes == ""


class TestReplayRecorderInit:
    """Tests for ReplayRecorder initialization."""

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.replays_dir.exists()
            assert (Path(tmpdir) / "replays").exists()

    def test_init_with_custom_config(self):
        """Test initialization with custom WorldConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(arena_width=15.0)
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                config=config,
                verbose=False
            )
            assert recorder.config.arena_width == 15.0

    def test_init_with_default_config(self):
        """Test initialization creates default WorldConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.config is not None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                max_ticks=500,
                samples_per_stage=5,
                min_matches_for_sampling=10,
                verbose=True
            )
            assert recorder.max_ticks == 500
            assert recorder.samples_per_stage == 5
            assert recorder.min_matches_for_sampling == 10
            assert recorder.verbose is True

    def test_init_creates_orchestrator(self):
        """Test that orchestrator is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.orchestrator is not None

    def test_init_creates_spectacle_evaluator(self):
        """Test that spectacle evaluator is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.spectacle_evaluator is not None

    def test_init_empty_replay_index(self):
        """Test that replay index starts empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.replay_index == []


class TestReplayRecorderSnapshotConversion:
    """Tests for snapshot to observation conversion."""

    def test_snapshot_to_obs_basic(self):
        """Test basic snapshot to observation conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            snapshot = {
                "you": {
                    "hp": 80,
                    "max_hp": 100,
                    "stamina": 50,
                    "max_stamina": 100,
                    "position": 3.0,
                    "velocity": 0.5
                },
                "opponent": {
                    "hp": 90,
                    "max_hp": 100,
                    "stamina": 75,
                    "max_stamina": 100,
                    "distance": 4.0,
                    "direction": 1.0,
                    "velocity": -0.2
                },
                "arena": {
                    "width": 12.5
                }
            }

            obs = recorder._snapshot_to_obs(snapshot)

            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert len(obs) == 13  # Enhanced observation space

            # Check specific values
            assert obs[0] == 3.0  # position
            assert obs[1] == 0.5  # velocity
            assert obs[2] == 0.8  # hp_norm (80/100)
            assert obs[3] == 0.5  # stamina_norm (50/100)
            assert obs[4] == 4.0  # distance
            assert obs[5] == pytest.approx(-0.2)  # rel_velocity
            assert obs[6] == 0.9  # opp_hp_norm (90/100)
            assert obs[7] == 0.75  # opp_stamina_norm (75/100)

    def test_snapshot_to_obs_full_hp_stamina(self):
        """Test conversion with full HP and stamina."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            snapshot = {
                "you": {
                    "hp": 100,
                    "max_hp": 100,
                    "stamina": 100,
                    "max_stamina": 100,
                    "position": 5.0,
                    "velocity": 0.0
                },
                "opponent": {
                    "hp": 100,
                    "max_hp": 100,
                    "stamina": 100,
                    "max_stamina": 100,
                    "distance": 0.0,
                    "direction": 1.0,
                    "velocity": 0.0
                },
                "arena": {
                    "width": 12.5
                }
            }

            obs = recorder._snapshot_to_obs(snapshot)

            assert obs[2] == 1.0  # hp_norm
            assert obs[3] == 1.0  # stamina_norm
            assert obs[6] == 1.0  # opp_hp_norm
            assert obs[7] == 1.0  # opp_stamina_norm

    def test_snapshot_to_obs_low_hp_stamina(self):
        """Test conversion with low HP and stamina."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            snapshot = {
                "you": {
                    "hp": 10,
                    "max_hp": 100,
                    "stamina": 5,
                    "max_stamina": 100,
                    "position": 2.0,
                    "velocity": -1.0
                },
                "opponent": {
                    "hp": 15,
                    "max_hp": 100,
                    "stamina": 10,
                    "max_stamina": 100,
                    "distance": 8.0,
                    "direction": 1.0,
                    "velocity": 2.0
                },
                "arena": {
                    "width": 12.5
                }
            }

            obs = recorder._snapshot_to_obs(snapshot)

            assert obs[2] == 0.1  # hp_norm (10/100)
            assert obs[3] == 0.05  # stamina_norm (5/100)
            assert obs[6] == 0.15  # opp_hp_norm (15/100)
            assert obs[7] == 0.1  # opp_stamina_norm (10/100)


class TestReplayRecorderActionConversion:
    """Tests for action to dictionary conversion."""

    def test_action_to_dict_neutral_stance(self):
        """Test action conversion with neutral stance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            action = np.array([0.5, 0])  # acceleration 0.5, stance 0 (neutral)
            result = recorder._action_to_dict(action)

            assert result["acceleration"] == 0.5
            assert result["stance"] == "neutral"

    def test_action_to_dict_extended_stance(self):
        """Test action conversion with extended stance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            action = np.array([1.0, 1])  # acceleration 1.0, stance 1 (extended)
            result = recorder._action_to_dict(action)

            assert result["acceleration"] == 1.0
            assert result["stance"] == "extended"

    def test_action_to_dict_defending_stance(self):
        """Test action conversion with defending stance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            action = np.array([-0.5, 2])  # acceleration -0.5, stance 2 (defending)
            result = recorder._action_to_dict(action)

            assert result["acceleration"] == -0.5
            assert result["stance"] == "defending"

    def test_action_to_dict_clips_acceleration(self):
        """Test that acceleration is clipped to [-1, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Test over max
            action = np.array([2.0, 1])
            result = recorder._action_to_dict(action)
            assert result["acceleration"] == 1.0

            # Test under min
            action = np.array([-2.0, 1])
            result = recorder._action_to_dict(action)
            assert result["acceleration"] == -1.0

    def test_action_to_dict_clips_stance_index(self):
        """Test that stance index is clipped to valid range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Test over max
            action = np.array([0.0, 10])
            result = recorder._action_to_dict(action)
            assert result["stance"] == "defending"  # Index 2 (max)

            # Test under min
            action = np.array([0.0, -5])
            result = recorder._action_to_dict(action)
            assert result["stance"] == "neutral"  # Index 0 (min)


class TestReplayRecorderIndex:
    """Tests for replay index management."""

    def test_save_replay_index_empty(self):
        """Test saving empty replay index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            recorder.save_replay_index()

            index_path = Path(tmpdir) / "replay_index.json"
            assert index_path.exists()

            with open(index_path) as f:
                data = json.load(f)

            assert data["total_replays"] == 0
            assert data["replays"] == []

    def test_save_replay_index_with_entries(self):
        """Test saving replay index with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Add some replay metadata
            recorder.replay_index.append(ReplayMetadata(
                stage="curriculum_level_1",
                stage_type="curriculum",
                spectacle_score=0.75,
                spectacle_rank="top",
                fighter_a="AI_Fighter",
                fighter_b="Boxer",
                winner="AI_Fighter",
                notes="Great match"
            ))
            recorder.replay_index.append(ReplayMetadata(
                stage="curriculum_level_1",
                stage_type="curriculum",
                spectacle_score=0.25,
                spectacle_rank="bottom",
                fighter_a="AI_Fighter",
                fighter_b="Tank",
                winner="Tank",
                notes="One-sided"
            ))

            recorder.save_replay_index()

            index_path = Path(tmpdir) / "replay_index.json"
            with open(index_path) as f:
                data = json.load(f)

            assert data["total_replays"] == 2
            assert len(data["replays"]) == 2
            assert data["replays"][0]["spectacle_rank"] == "top"
            assert data["replays"][1]["spectacle_rank"] == "bottom"

    def test_save_replay_index_preserves_all_fields(self):
        """Test that all metadata fields are preserved in index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            recorder.replay_index.append(ReplayMetadata(
                stage="population_gen_10",
                stage_type="population",
                spectacle_score=0.666,
                spectacle_rank="middle",
                fighter_a="Champion",
                fighter_b="Challenger",
                winner="Champion",
                notes="Close fight"
            ))

            recorder.save_replay_index()

            index_path = Path(tmpdir) / "replay_index.json"
            with open(index_path) as f:
                data = json.load(f)

            replay = data["replays"][0]
            assert replay["stage"] == "population_gen_10"
            assert replay["stage_type"] == "population"
            assert replay["spectacle_score"] == 0.666
            assert replay["spectacle_rank"] == "middle"
            assert replay["fighter_a"] == "Champion"
            assert replay["fighter_b"] == "Challenger"
            assert replay["winner"] == "Champion"
            assert replay["notes"] == "Close fight"


class TestReplayRecorderLogging:
    """Tests for logging behavior."""

    def test_verbose_mode_enabled(self):
        """Test that verbose mode sets up logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=True
            )
            assert recorder.verbose is True
            assert recorder.logger is not None

    def test_verbose_mode_disabled(self):
        """Test that verbose can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            assert recorder.verbose is False


class TestReplayRecorderEdgeCases:
    """Tests for edge cases."""

    def test_config_with_custom_arena_width(self):
        """Test snapshot conversion uses config arena width."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(arena_width=20.0)
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                config=config,
                verbose=False
            )

            snapshot = {
                "you": {
                    "hp": 100, "max_hp": 100,
                    "stamina": 100, "max_stamina": 100,
                    "position": 10.0, "velocity": 0.0
                },
                "opponent": {
                    "hp": 100, "max_hp": 100,
                    "stamina": 100, "max_stamina": 100,
                    "distance": 5.0, "direction": 1.0, "velocity": 0.0
                },
                "arena": {
                    "width": 20.0
                }
            }

            obs = recorder._snapshot_to_obs(snapshot)
            # Arena width should be in the observation
            assert obs[8] == 20.0

    def test_multiple_save_replay_index_overwrites(self):
        """Test that saving index multiple times overwrites."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # First save with one entry
            recorder.replay_index.append(ReplayMetadata(
                stage="stage1", stage_type="curriculum",
                spectacle_score=0.5, spectacle_rank="middle",
                fighter_a="A", fighter_b="B", winner="A"
            ))
            recorder.save_replay_index()

            # Add more and save again
            recorder.replay_index.append(ReplayMetadata(
                stage="stage2", stage_type="curriculum",
                spectacle_score=0.7, spectacle_rank="top",
                fighter_a="C", fighter_b="D", winner="C"
            ))
            recorder.save_replay_index()

            index_path = Path(tmpdir) / "replay_index.json"
            with open(index_path) as f:
                data = json.load(f)

            assert data["total_replays"] == 2

    def test_action_with_float_stance(self):
        """Test action conversion with float stance value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Float stance should be converted to int
            action = np.array([0.0, 1.7])
            result = recorder._action_to_dict(action)
            assert result["stance"] == "extended"  # int(1.7) = 1 (extended)


class TestSaveSampledReplays:
    """Tests for _save_sampled_replays method."""

    def test_skips_when_too_few_matches(self):
        """Test that sampling is skipped with too few matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                min_matches_for_sampling=10,
                verbose=False
            )

            # Only 3 matches, need 10
            matches = [
                (Mock(), Mock(overall=0.5), "A", "B"),
                (Mock(), Mock(overall=0.6), "A", "C"),
                (Mock(), Mock(overall=0.7), "A", "D"),
            ]

            recorder._save_sampled_replays(
                matches_with_scores=matches,
                stage_id="test_stage",
                stage_type="curriculum"
            )

            # No replays should be saved
            assert len(recorder.replay_index) == 0

    def test_saves_three_samples(self):
        """Test that bottom, middle, and top samples are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.atom.training.replay_recorder.save_replay') as mock_save:
                recorder = ReplayRecorder(
                    output_dir=tmpdir,
                    min_matches_for_sampling=5,
                    verbose=False
                )

                # Create mock matches with different spectacle scores
                matches = []
                for i in range(10):
                    mock_result = Mock()
                    mock_result.telemetry = {"ticks": []}
                    mock_result.winner = "AI"

                    mock_score = Mock()
                    mock_score.overall = i * 0.1  # 0.0 to 0.9
                    mock_score.to_dict = Mock(return_value={"overall": i * 0.1})

                    matches.append((mock_result, mock_score, "AI", f"Opp{i}"))

                recorder._save_sampled_replays(
                    matches_with_scores=matches,
                    stage_id="test_stage",
                    stage_type="curriculum"
                )

                # Should save 3 replays (bottom, middle, top)
                assert mock_save.call_count == 3
                assert len(recorder.replay_index) == 3

                # Check ranks
                ranks = [r.spectacle_rank for r in recorder.replay_index]
                assert "bottom" in ranks
                assert "middle" in ranks
                assert "top" in ranks

    def test_sorts_by_spectacle_score(self):
        """Test that matches are sorted by spectacle score before sampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.atom.training.replay_recorder.save_replay'):
                recorder = ReplayRecorder(
                    output_dir=tmpdir,
                    min_matches_for_sampling=5,
                    verbose=False
                )

                # Create matches in random order
                matches = []
                scores = [0.7, 0.2, 0.9, 0.1, 0.5, 0.8]
                for i, score in enumerate(scores):
                    mock_result = Mock()
                    mock_result.telemetry = {"ticks": []}
                    mock_result.winner = "AI"

                    mock_score = Mock()
                    mock_score.overall = score
                    mock_score.to_dict = Mock(return_value={"overall": score})

                    matches.append((mock_result, mock_score, "AI", f"Opp{i}"))

                recorder._save_sampled_replays(
                    matches_with_scores=matches,
                    stage_id="test_stage",
                    stage_type="curriculum"
                )

                # Bottom should have low score, top should have high score
                bottom_replay = next(r for r in recorder.replay_index if r.spectacle_rank == "bottom")
                top_replay = next(r for r in recorder.replay_index if r.spectacle_rank == "top")

                assert bottom_replay.spectacle_score < 0.3  # Low score
                assert top_replay.spectacle_score > 0.7  # High score

    def test_creates_correct_metadata(self):
        """Test that metadata is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.atom.training.replay_recorder.save_replay') as mock_save:
                recorder = ReplayRecorder(
                    output_dir=tmpdir,
                    min_matches_for_sampling=5,
                    verbose=False
                )

                matches = []
                for i in range(6):
                    mock_result = Mock()
                    mock_result.telemetry = {"ticks": []}
                    mock_result.winner = "Fighter_A"

                    mock_score = Mock()
                    mock_score.overall = i * 0.15
                    mock_score.to_dict = Mock(return_value={"overall": i * 0.15})

                    matches.append((mock_result, mock_score, "Fighter_A", "Fighter_B"))

                recorder._save_sampled_replays(
                    matches_with_scores=matches,
                    stage_id="population_gen_5",
                    stage_type="population"
                )

                # Check that save_replay was called with correct metadata
                assert mock_save.call_count == 3
                for call in mock_save.call_args_list:
                    metadata = call[1]['metadata']
                    assert metadata['stage'] == "population_gen_5"
                    assert metadata['stage_type'] == "population"
                    assert 'spectacle_score' in metadata
                    assert 'spectacle_rank' in metadata

    def test_verbose_logging(self):
        """Test that verbose mode logs information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.atom.training.replay_recorder.save_replay'):
                recorder = ReplayRecorder(
                    output_dir=tmpdir,
                    min_matches_for_sampling=5,
                    verbose=True
                )

                matches = []
                for i in range(6):
                    mock_result = Mock()
                    mock_result.telemetry = {"ticks": []}
                    mock_result.winner = "AI"

                    mock_score = Mock()
                    mock_score.overall = i * 0.15
                    mock_score.to_dict = Mock(return_value={"overall": i * 0.15})

                    matches.append((mock_result, mock_score, "AI", f"Opp{i}"))

                # Should not raise even with verbose logging
                recorder._save_sampled_replays(
                    matches_with_scores=matches,
                    stage_id="test_stage",
                    stage_type="curriculum"
                )

                assert len(recorder.replay_index) == 3


class TestRecordCurriculumStage:
    """Tests for record_curriculum_stage method."""

    def test_creates_correct_stage_id(self):
        """Test that stage ID is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)
            recorder._run_curriculum_eval_matches = Mock(return_value=[])
            recorder._save_sampled_replays = Mock()

            recorder.record_curriculum_stage(
                stage_name="Basic Movement",
                level_num=2,
                model=Mock(),
                opponent_paths=[]
            )

            # Verify _run_curriculum_eval_matches was called
            recorder._run_curriculum_eval_matches.assert_called_once()

    def test_handles_empty_matches(self):
        """Test handling when no matches are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)
            recorder._run_curriculum_eval_matches = Mock(return_value=[])
            recorder._save_sampled_replays = Mock()

            recorder.record_curriculum_stage(
                stage_name="Test",
                level_num=1,
                model=Mock(),
                opponent_paths=[]
            )

            # _save_sampled_replays should not be called with empty matches
            recorder._save_sampled_replays.assert_not_called()

    def test_calls_save_sampled_with_matches(self):
        """Test that _save_sampled_replays is called when matches exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            mock_matches = [
                (Mock(), Mock(overall=0.5), "AI", "Opp")
            ]
            recorder._run_curriculum_eval_matches = Mock(return_value=mock_matches)
            recorder._save_sampled_replays = Mock()

            recorder.record_curriculum_stage(
                stage_name="Test Stage",
                level_num=3,
                model=Mock(),
                opponent_paths=["opp1.py"]
            )

            recorder._save_sampled_replays.assert_called_once()
            call_args = recorder._save_sampled_replays.call_args
            assert call_args[1]['stage_id'] == "curriculum_level_3_test_stage"
            assert call_args[1]['stage_type'] == "curriculum"

    def test_verbose_logging_enabled(self):
        """Test verbose logging during curriculum recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=True)
            recorder._run_curriculum_eval_matches = Mock(return_value=[])

            # Should not raise
            recorder.record_curriculum_stage(
                stage_name="Fundamentals",
                level_num=1,
                model=Mock(),
                opponent_paths=[]
            )


class TestRecordPopulationGeneration:
    """Tests for record_population_generation method."""

    def test_creates_correct_stage_id(self):
        """Test that stage ID is created correctly for population."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)
            recorder._run_population_eval_matches = Mock(return_value=[])
            recorder._save_sampled_replays = Mock()

            recorder.record_population_generation(
                generation=10,
                fighters=[]
            )

            recorder._run_population_eval_matches.assert_called_once()

    def test_handles_empty_matches(self):
        """Test handling when no matches are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)
            recorder._run_population_eval_matches = Mock(return_value=[])
            recorder._save_sampled_replays = Mock()

            recorder.record_population_generation(
                generation=5,
                fighters=[]
            )

            recorder._save_sampled_replays.assert_not_called()

    def test_calls_save_sampled_with_matches(self):
        """Test that _save_sampled_replays is called when matches exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            mock_matches = [
                (Mock(), Mock(overall=0.5), "Fighter1", "Fighter2")
            ]
            recorder._run_population_eval_matches = Mock(return_value=mock_matches)
            recorder._save_sampled_replays = Mock()

            recorder.record_population_generation(
                generation=7,
                fighters=[Mock(), Mock()]
            )

            recorder._save_sampled_replays.assert_called_once()
            call_args = recorder._save_sampled_replays.call_args
            assert call_args[1]['stage_id'] == "population_gen_7"
            assert call_args[1]['stage_type'] == "population"

    def test_verbose_logging_enabled(self):
        """Test verbose logging during population recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=True)
            recorder._run_population_eval_matches = Mock(return_value=[])

            # Should not raise
            recorder.record_population_generation(
                generation=1,
                fighters=[]
            )


class TestRunCurriculumEvalMatches:
    """Tests for _run_curriculum_eval_matches method."""

    def test_returns_empty_for_no_opponents(self):
        """Test that no opponents returns no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            result = recorder._run_curriculum_eval_matches(
                model=Mock(),
                opponent_paths=[],
                num_matches_per_opponent=3,
                stage_name="Test"
            )

            assert result == []

    def test_handles_invalid_opponent_path(self):
        """Test handling of invalid opponent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            # Invalid path should be skipped
            result = recorder._run_curriculum_eval_matches(
                model=Mock(),
                opponent_paths=["/nonexistent/path/opponent.py"],
                num_matches_per_opponent=3,
                stage_name="Test"
            )

            assert result == []

    def test_loads_and_runs_opponent(self):
        """Test loading opponent and running matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy opponent file
            opp_file = Path(tmpdir) / "test_opp.py"
            opp_file.write_text('''
def decide(state):
    return {"stance": "neutral", "acceleration": 0}
''')

            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            # Mock the orchestrator's run_match
            mock_result = Mock()
            mock_result.telemetry = {"ticks": []}
            mock_result.winner = "AI"
            recorder.orchestrator.run_match = Mock(return_value=mock_result)

            # Mock spectacle evaluator
            mock_spectacle = Mock()
            mock_spectacle.overall = 0.5
            recorder.spectacle_evaluator.evaluate = Mock(return_value=mock_spectacle)

            # Mock model predict
            mock_model = Mock()
            mock_model.predict = Mock(return_value=(np.array([0.0, 1.0]), None))

            result = recorder._run_curriculum_eval_matches(
                model=mock_model,
                opponent_paths=[str(opp_file)],
                num_matches_per_opponent=2,
                stage_name="Test"
            )

            assert len(result) == 2
            assert recorder.orchestrator.run_match.call_count == 2


class TestRunPopulationEvalMatches:
    """Tests for _run_population_eval_matches method."""

    def test_returns_empty_for_single_fighter(self):
        """Test that single fighter returns no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            # Single fighter - no pairs possible
            fighter = Mock()
            fighter.name = "Fighter1"
            fighter.mass = 70.0

            result = recorder._run_population_eval_matches(
                fighters=[fighter],
                num_matches_per_pair=2,
                generation=1
            )

            assert result == []

    def test_returns_empty_for_no_fighters(self):
        """Test that no fighters returns no matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            result = recorder._run_population_eval_matches(
                fighters=[],
                num_matches_per_pair=2,
                generation=1
            )

            assert result == []

    def test_runs_matches_for_fighter_pairs(self):
        """Test running matches for multiple fighter pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            # Mock the orchestrator's run_match
            mock_result = Mock()
            mock_result.telemetry = {"ticks": []}
            mock_result.winner = "Fighter1"
            recorder.orchestrator.run_match = Mock(return_value=mock_result)

            # Mock spectacle evaluator
            mock_spectacle = Mock()
            mock_spectacle.overall = 0.5
            recorder.spectacle_evaluator.evaluate = Mock(return_value=mock_spectacle)

            # Create mock fighters
            fighters = []
            for i in range(3):
                fighter = Mock()
                fighter.name = f"Fighter{i}"
                fighter.mass = 70.0
                fighter.model = Mock()
                fighter.model.predict = Mock(return_value=(np.array([0.0, 1.0]), None))
                fighters.append(fighter)

            result = recorder._run_population_eval_matches(
                fighters=fighters,
                num_matches_per_pair=1,
                generation=5
            )

            # 3 fighters = 3 pairs (0-1, 0-2, 1-2), 1 match each = 3 matches
            assert len(result) == 3
            assert recorder.orchestrator.run_match.call_count == 3

    def test_correct_number_of_matches_per_pair(self):
        """Test that correct number of matches run per pair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            mock_result = Mock()
            mock_result.telemetry = {"ticks": []}
            mock_result.winner = "Fighter1"
            recorder.orchestrator.run_match = Mock(return_value=mock_result)

            mock_spectacle = Mock()
            mock_spectacle.overall = 0.5
            recorder.spectacle_evaluator.evaluate = Mock(return_value=mock_spectacle)

            fighters = []
            for i in range(2):
                fighter = Mock()
                fighter.name = f"Fighter{i}"
                fighter.mass = 70.0
                fighter.model = Mock()
                fighter.model.predict = Mock(return_value=(np.array([0.0, 1.0]), None))
                fighters.append(fighter)

            result = recorder._run_population_eval_matches(
                fighters=fighters,
                num_matches_per_pair=3,
                generation=1
            )

            # 2 fighters = 1 pair, 3 matches each = 3 matches
            assert len(result) == 3
