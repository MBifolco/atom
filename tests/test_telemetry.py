"""
Tests for telemetry/replay store system.
"""

import pytest
import tempfile
import json
import gzip
from pathlib import Path
from src.telemetry import ReplayStore
from src.orchestrator import MatchOrchestrator, MatchResult
from src.arena import WorldConfig


def simple_test_fighter(state):
    """Simple fighter for testing (renamed to not be detected as test)."""
    direction = state["opponent"]["direction"]
    return {"acceleration": 0.8 * direction, "stance": "extended"}


class TestReplayStore:
    """Test replay storage and retrieval."""

    def test_replay_store_initialization(self):
        """Test ReplayStore creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = Path(tmpdir) / "replays"
            store = ReplayStore(str(replay_dir))

            assert store.replay_dir == replay_dir
            assert replay_dir.exists()

    def test_save_replay_uncompressed(self):
        """Test saving replay as uncompressed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            # Create a simple match result
            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=20, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            # Save replay
            filepath = store.save(
                result.telemetry,
                result,
                compress=False,
                filename="test_replay.json"
            )

            assert filepath.exists()
            assert filepath.suffix == ".json"

            # Verify content
            with open(filepath) as f:
                data = json.load(f)

            assert "version" in data
            assert "result" in data
            assert "telemetry" in data
            assert data["result"]["winner"] == result.winner

    def test_save_replay_compressed(self):
        """Test saving replay as compressed JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=20, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            # Save compressed replay
            filepath = store.save(
                result.telemetry,
                result,
                compress=True,
                filename="test_replay.json.gz"
            )

            assert filepath.exists()
            assert filepath.suffix == ".gz"

            # Verify can decompress and read
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)

            assert data["result"]["winner"] == result.winner

    def test_save_with_metadata(self):
        """Test saving replay with additional metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            metadata = {
                "tournament": "Test Tournament",
                "round": 1,
                "notes": "Test match"
            }

            filepath = store.save(
                result.telemetry,
                result,
                metadata=metadata,
                compress=False
            )

            # Read and verify metadata
            with open(filepath) as f:
                data = json.load(f)

            assert data["metadata"]["tournament"] == "Test Tournament"
            assert data["metadata"]["round"] == 1

    def test_auto_generated_filename(self):
        """Test auto-generated filename includes fighter names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "Boxer", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "Slugger", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            # Save with auto-generated filename
            filepath = store.save(
                result.telemetry,
                result,
                compress=True
            )

            # Filename should contain fighter names
            assert "Boxer" in filepath.name or "Slugger" in filepath.name
            assert "replay_" in filepath.name
            assert filepath.suffix == ".gz"

    def test_load_replay_uncompressed(self):
        """Test loading uncompressed replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            # Save
            filepath = store.save(result.telemetry, result, compress=False, filename="test.json")

            # Load (convert Path to string)
            loaded = store.load(str(filepath))

            assert loaded is not None
            assert "result" in loaded
            assert loaded["result"]["winner"] == result.winner

    def test_load_replay_compressed(self):
        """Test loading compressed replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)

            config = WorldConfig()
            orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

            result = orchestrator.run_match(
                fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
                fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
                decision_func_a=simple_test_fighter,
                decision_func_b=simple_test_fighter,
                seed=42
            )

            # Save compressed
            filepath = store.save(result.telemetry, result, compress=True, filename="test.json.gz")

            # Load (convert Path to string)
            loaded = store.load(str(filepath))

            assert loaded is not None
            assert loaded["result"]["winner"] == result.winner


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
