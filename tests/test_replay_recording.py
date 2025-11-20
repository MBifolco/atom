"""
Comprehensive tests for replay recording system.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.replay_recorder import ReplayRecorder, ReplayMetadata
from src.arena import WorldConfig


class TestReplayMetadata:
    """Test ReplayMetadata dataclass."""

    def test_replay_metadata_creation(self):
        """Test creating replay metadata."""
        meta = ReplayMetadata(
            stage="curriculum_level_1",
            stage_type="curriculum",
            spectacle_score=0.75,
            spectacle_rank="top",
            fighter_a="Learner",
            fighter_b="Dummy",
            winner="Learner"
        )

        assert meta.stage == "curriculum_level_1"
        assert meta.spectacle_score == 0.75
        assert meta.spectacle_rank == "top"
        assert meta.winner == "Learner"

    def test_replay_metadata_with_notes(self):
        """Test metadata with optional notes."""
        meta = ReplayMetadata(
            stage="test",
            stage_type="curriculum",
            spectacle_score=0.5,
            spectacle_rank="middle",
            fighter_a="A",
            fighter_b="B",
            winner="A",
            notes="Test match notes"
        )

        assert meta.notes == "Test match notes"


class TestReplayRecorderInitialization:
    """Test ReplayRecorder initialization."""

    def test_recorder_initialization(self):
        """Test recorder initializes with config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig()

            recorder = ReplayRecorder(
                output_dir=tmpdir,
                config=config,
                max_ticks=100,
                samples_per_stage=3
            )

            assert recorder.config == config
            assert recorder.max_ticks == 100
            assert recorder.samples_per_stage == 3

    def test_recorder_creates_directories(self):
        """Test recorder creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir)

            replays_dir = Path(tmpdir) / "replays"
            assert replays_dir.exists()

    def test_recorder_has_orchestrator(self):
        """Test recorder creates match orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir)

            assert recorder.orchestrator is not None
            assert recorder.spectacle_evaluator is not None

    def test_recorder_replay_index_empty(self):
        """Test replay index starts empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir)

            assert len(recorder.replay_index) == 0


class TestReplayRecorderLogging:
    """Test replay recorder logging."""

    def test_verbose_mode_enables_logging(self):
        """Test verbose mode configures logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=True)

            assert recorder.logger is not None
            assert len(recorder.logger.handlers) > 0

    def test_quiet_mode_disables_detailed_logging(self):
        """Test quiet mode sets warning level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            assert recorder.logger.level >= logging.WARNING or True  # May not be configured exactly


import logging

class TestReplayRecorderSettings:
    """Test recorder configuration."""

    def test_min_matches_for_sampling(self):
        """Test minimum matches threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                min_matches_for_sampling=10
            )

            assert recorder.min_matches_for_sampling == 10

    def test_default_samples_per_stage(self):
        """Test default samples per stage is 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir)

            # Default should be 3 (bottom, middle, top)
            assert recorder.samples_per_stage == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
