"""
Complete replay recorder coverage tests.
Target lines 111-435 (uncovered methods).
"""

import pytest
import tempfile
from pathlib import Path
from src.training.replay_recorder import ReplayRecorder, ReplayMetadata
from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator


def simple_test_fighter_replay(state):
    """Simple fighter for replay testing."""
    direction = state.get("opponent", {}).get("direction", 1.0)
    return {"acceleration": 0.6 * direction, "stance": "neutral"}


class TestReplayRecorderConfiguration:
    """Test replay recorder configuration options."""

    def test_recorder_with_custom_samples_per_stage(self):
        """Test recorder with custom sampling count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                samples_per_stage=5,  # Custom: bottom, middle, top + 2 more
                verbose=False
            )

            assert recorder.samples_per_stage == 5

    def test_recorder_with_custom_min_matches(self):
        """Test recorder with custom minimum matches threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                min_matches_for_sampling=10,
                verbose=False
            )

            assert recorder.min_matches_for_sampling == 10

    def test_recorder_creates_subdirectories(self):
        """Test recorder creates replays subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir)

            replays_dir = recorder.replays_dir

            assert replays_dir.exists()
            assert replays_dir.name == "replays"

    def test_recorder_uses_custom_config(self):
        """Test recorder uses provided WorldConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig()

            recorder = ReplayRecorder(
                output_dir=tmpdir,
                config=config,
                verbose=False
            )

            assert recorder.config == config

    def test_recorder_creates_default_config(self):
        """Test recorder creates default config if none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            assert recorder.config is not None
            assert isinstance(recorder.config, WorldConfig)

    def test_recorder_with_custom_max_ticks(self):
        """Test recorder with custom max ticks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(
                output_dir=tmpdir,
                max_ticks=150,
                verbose=False
            )

            assert recorder.max_ticks == 150
            assert recorder.orchestrator.max_ticks == 150

    def test_recorder_verbose_vs_quiet(self):
        """Test recorder verbose and quiet modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Verbose mode
            recorder_verbose = ReplayRecorder(
                output_dir=tmpdir,
                verbose=True
            )

            assert recorder_verbose.verbose is True

            # Quiet mode
            recorder_quiet = ReplayRecorder(
                output_dir=str(Path(tmpdir) / "quiet"),
                verbose=False
            )

            assert recorder_quiet.verbose is False


class TestReplayMetadataStructure:
    """Test ReplayMetadata dataclass."""

    def test_replay_metadata_all_fields(self):
        """Test metadata with all fields populated."""
        meta = ReplayMetadata(
            stage="curriculum_level_3_intermediate",
            stage_type="curriculum",
            spectacle_score=0.85,
            spectacle_rank="top",
            fighter_a="Learner_Gen3",
            fighter_b="Expert_Opponent",
            winner="Learner_Gen3",
            notes="Excellent fight with comeback"
        )

        assert meta.stage == "curriculum_level_3_intermediate"
        assert meta.stage_type == "curriculum"
        assert meta.spectacle_score == 0.85
        assert meta.spectacle_rank == "top"
        assert meta.fighter_a == "Learner_Gen3"
        assert meta.fighter_b == "Expert_Opponent"
        assert meta.winner == "Learner_Gen3"
        assert meta.notes == "Excellent fight with comeback"

    def test_replay_metadata_minimal(self):
        """Test metadata with minimal required fields."""
        meta = ReplayMetadata(
            stage="test_stage",
            stage_type="test",
            spectacle_score=0.5,
            spectacle_rank="middle",
            fighter_a="A",
            fighter_b="B",
            winner="A"
        )

        assert meta.notes == ""  # Default

    def test_replay_metadata_spectacle_ranks(self):
        """Test different spectacle ranks."""
        ranks = ["bottom", "middle", "top"]

        for rank in ranks:
            meta = ReplayMetadata(
                stage="test",
                stage_type="test",
                spectacle_score=0.5,
                spectacle_rank=rank,
                fighter_a="A",
                fighter_b="B",
                winner="A"
            )

            assert meta.spectacle_rank == rank

    def test_replay_metadata_stage_types(self):
        """Test different stage types."""
        for stage_type in ["curriculum", "population"]:
            meta = ReplayMetadata(
                stage=f"{stage_type}_test",
                stage_type=stage_type,
                spectacle_score=0.5,
                spectacle_rank="middle",
                fighter_a="A",
                fighter_b="B",
                winner="A"
            )

            assert meta.stage_type == stage_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
