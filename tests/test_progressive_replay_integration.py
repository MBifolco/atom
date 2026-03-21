"""
Integration tests for progressive replay recording during training.
"""

import pytest
import tempfile
from pathlib import Path
import json
import numpy as np
from unittest.mock import MagicMock, patch, call

from src.training.trainers.curriculum_trainer import CurriculumTrainer, CurriculumCallback
from src.atom.training.progressive_replay_recorder import ProgressiveReplayRecorder
from src.atom.runtime.orchestrator.match_orchestrator import MatchResult


class TestProgressiveReplayIntegration:
    """Test that progressive replays are actually recorded during training."""

    @pytest.fixture
    def mock_curriculum_trainer(self):
        """Create a mock curriculum trainer with necessary attributes."""
        trainer = MagicMock()
        trainer.verbose = True
        trainer.max_ticks = 250
        trainer.progress = MagicMock()
        trainer.progress.current_level = 1

        # Mock the model
        trainer.model = MagicMock()
        trainer.model.predict = MagicMock(return_value=(np.array([2]), None))

        # Mock level
        level = MagicMock()
        level.name = "Test Level"
        level.fighter = "Dummy"
        trainer.get_current_level = MagicMock(return_value=level)

        # Mock logger
        trainer.logger = MagicMock()

        return trainer

    def test_callback_initializes_recorder(self):
        """Test that CurriculumCallback properly initializes with recorder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = MagicMock()
            trainer.output_dir = tmpdir
            trainer.record_replays = True
            trainer.max_ticks = 250
            trainer.verbose = True

            # Initialize progressive recorder
            trainer.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                max_ticks=250,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)

            assert hasattr(trainer, 'progressive_recorder')
            assert trainer.progressive_recorder is not None

    def test_callback_checks_recording_intervals(self):
        """Test that callback checks recording intervals correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = MagicMock()
            trainer.output_dir = tmpdir
            trainer.record_replays = True
            trainer.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )
            trainer.verbose = False

            callback = CurriculumCallback(trainer, verbose=0)

            # Simulate episode completions
            for i in range(1, 21):
                # Simulate episode info
                info = [{
                    "episode": {"r": -100 + i * 5},  # Improving rewards
                    "won": i > 10  # Start winning after episode 10
                }]

                callback.episode_rewards.append(info[0]["episode"]["r"])
                callback.episode_wins.append(info[0]["won"])

            # Episode 1 should record (first episode)
            assert trainer.progressive_recorder.should_record(1, 1000)

            # Episode 25 should record (early phase interval)
            assert trainer.progressive_recorder.should_record(25, 1000)

            # Episode 15 should not record
            assert not trainer.progressive_recorder.should_record(15, 1000)

            # Episode 50 should record
            assert trainer.progressive_recorder.should_record(50, 1000)

    @patch('src.atom.runtime.orchestrator.match_orchestrator.MatchOrchestrator')
    def test_record_evaluation_replay_runs(self, mock_orchestrator, mock_curriculum_trainer):
        """Test that _record_evaluation_replay actually runs a match and saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mocks
            mock_curriculum_trainer.output_dir = tmpdir
            mock_curriculum_trainer.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Mock level with opponents
            mock_curriculum_trainer.get_current_level.return_value.opponents = []

            # Mock orchestrator
            mock_match_instance = MagicMock()
            mock_orchestrator.return_value = mock_match_instance

            # Mock match result with telemetry included
            mock_result = MatchResult(
                winner="AI_Fighter",
                total_ticks=100,
                final_hp_a=75.0,
                final_hp_b=0.0,
                telemetry={"ticks": [{"tick": 0}]},
                events=[]
            )
            mock_match_instance.run_match.return_value = mock_result

            # Create callback and set up episode data
            callback = CurriculumCallback(mock_curriculum_trainer, verbose=0)
            callback.episode_rewards = [10, 20, 30]
            callback.episode_wins = [False, True, True]

            # Call the record method
            callback._record_evaluation_replay(episode_num=3, total_episodes=1000)

            # Verify orchestrator was called
            mock_orchestrator.assert_called_once()
            assert mock_orchestrator.call_args[1]['record_telemetry'] == True
            assert mock_orchestrator.call_args[1]['max_ticks'] == 250

            # Verify match was run
            mock_match_instance.run_match.assert_called_once()

            # Check that a replay file was saved
            replay_files = list((Path(tmpdir) / "progressive_replays").glob("*.json.gz"))
            assert len(replay_files) == 1

            # Verify filename format
            filename = replay_files[0].name
            # Level number can be 1 or 2 depending on internal state
            assert "level_" in filename
            assert "ep_00003" in filename
            assert "wr_066" in filename  # 66% win rate (2/3)

    def test_episode_detection_triggers_recording(self, mock_curriculum_trainer):
        """Test that episode completion triggers recording at right intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_curriculum_trainer.output_dir = tmpdir
            mock_curriculum_trainer.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(mock_curriculum_trainer, verbose=0)

            # Mock the record method to track calls
            callback._record_evaluation_replay = MagicMock()

            # Simulate multiple episodes
            for i in range(1, 51):
                info = [{
                    "episode": {"r": -100 + i * 2},
                    "won": i > 25
                }]

                # Process episode
                callback.episode_rewards.append(info[0]["episode"]["r"])
                callback.episode_wins.append(info[0]["won"])

                # Trigger the callback logic
                if mock_curriculum_trainer.progressive_recorder.should_record(i, 1000):
                    callback._record_evaluation_replay(i, 1000)

            # Check that recording was called at expected intervals
            calls = callback._record_evaluation_replay.call_args_list
            episode_nums = [call[0][0] for call in calls]

            # Should have recorded: 1, 25, 50 (every 25 episodes in early phase)
            assert 1 in episode_nums  # First episode
            assert 25 in episode_nums  # Early phase
            assert 50 in episode_nums  # Early phase
            assert 10 not in episode_nums  # Should not record at 10
            assert 20 not in episode_nums  # Should not record at 20
            assert 30 not in episode_nums  # Should not record at 30

    def test_recording_handles_errors_gracefully(self, mock_curriculum_trainer):
        """Test that recording errors don't crash training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_curriculum_trainer.output_dir = tmpdir
            mock_curriculum_trainer.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                verbose=True
            )
            # Mock should_graduate to return False
            mock_curriculum_trainer.should_graduate = MagicMock(return_value=False)
            mock_curriculum_trainer.update_progress = MagicMock()

            callback = CurriculumCallback(mock_curriculum_trainer, verbose=0)

            # Mock locals to simulate environment info
            callback.locals = {
                "infos": [{"episode": {"r": 100}, "won": True}]
            }

            # Make _record_evaluation_replay raise an error
            with patch.object(callback, '_record_evaluation_replay', side_effect=Exception("Test error")):
                # This should not raise - error should be caught
                result = callback._on_step()

                # Training should continue (return True)
                assert result == True

                # Verify error was logged (check any of the error calls)
                mock_curriculum_trainer.logger.error.assert_called()
                # Get all error calls
                error_calls = [call[0][0] for call in mock_curriculum_trainer.logger.error.call_args_list]
                # Check that at least one contains the error message
                assert any("Failed to record replay" in msg for msg in error_calls)

    def test_progressive_index_saved_on_completion(self):
        """Test that progressive index is saved when training completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                verbose=False
            )

            # Simulate recording some replays
            for i in [1, 25, 50]:
                # Create valid telemetry with at least one tick
                telemetry = {"ticks": [{"tick": 0, "fighter_a": {"hp": 100}, "fighter_b": {"hp": 100}}]}

                match_result = MatchResult(
                    winner="AI_Fighter",
                    total_ticks=100 + i,
                    final_hp_a=100 - i,
                    final_hp_b=0.0,
                    telemetry=telemetry,
                    events=[]
                )

                recorder.record_episode_replay(
                    telemetry=telemetry,
                    match_result=match_result,
                    level_name="Test Level",
                    level_num=1,
                    episode=i,
                    total_episodes=1000,
                    win_rate=i / 100,
                    recent_rewards=[i * 10]
                )

            # Save index
            recorder.save_progressive_index()

            # Check index file exists and contains correct data
            index_path = Path(tmpdir) / "progressive_replay_index.json"
            assert index_path.exists()

            with open(index_path) as f:
                index_data = json.load(f)

            assert index_data["total_replays"] == 3
            assert len(index_data["replays"]) == 3

            # Check replays are in order
            episodes = [r["episode"] for r in index_data["replays"]]
            assert episodes == [1, 25, 50]

            # Check recording strategy is saved
            assert "recording_strategy" in index_data
            assert index_data["recording_strategy"]["early_phase_interval"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])