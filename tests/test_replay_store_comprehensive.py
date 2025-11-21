"""
Comprehensive tests for replay_store to cover uncovered paths.
"""

import pytest
import tempfile
import json
import gzip
from pathlib import Path
from unittest.mock import Mock

from src.telemetry.replay_store import ReplayStore, save_replay, load_replay


class TestReplayStoreMethods:
    """Tests for ReplayStore class methods."""

    def test_init_creates_directory(self):
        """Test that ReplayStore creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = Path(tmpdir) / "new_replays"
            store = ReplayStore(replay_dir=str(replay_dir))
            assert replay_dir.exists()

    def test_list_replays_empty(self):
        """Test listing replays in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)
            replays = store.list_replays()
            assert replays == []

    def test_list_replays_with_files(self):
        """Test listing replays with files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)

            # Create some replay files
            (Path(tmpdir) / "replay_001.json").write_text('{}')
            (Path(tmpdir) / "replay_002.json.gz").write_bytes(gzip.compress(b'{}'))

            replays = store.list_replays()
            assert len(replays) == 2
            assert "replay_002.json.gz" in replays
            assert "replay_001.json" in replays

    def test_load_compressed_replay(self):
        """Test loading gzip compressed replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)

            # Create compressed replay
            replay_data = {"version": "1.0", "ticks": []}
            filepath = Path(tmpdir) / "replay_test.json.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(replay_data, f)

            loaded = store.load("replay_test.json.gz")
            assert loaded["version"] == "1.0"

    def test_load_uncompressed_replay(self):
        """Test loading uncompressed replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)

            # Create uncompressed replay
            replay_data = {"version": "1.0", "events": []}
            filepath = Path(tmpdir) / "replay_test.json"
            with open(filepath, 'w') as f:
                json.dump(replay_data, f)

            loaded = store.load("replay_test.json")
            assert loaded["version"] == "1.0"

    def test_load_nonexistent_raises(self):
        """Test that loading nonexistent replay raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)

            with pytest.raises(FileNotFoundError) as exc_info:
                store.load("nonexistent.json")
            assert "Replay not found" in str(exc_info.value)

    def test_get_replay_info(self):
        """Test getting replay info/metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(replay_dir=tmpdir)

            # Create replay with metadata
            replay_data = {
                "version": "1.0",
                "timestamp": "2024-01-01T12:00:00",
                "telemetry": {
                    "fighter_a_name": "Alpha",
                    "fighter_b_name": "Beta",
                    "ticks": []
                },
                "result": {
                    "winner": "fighter_a",
                    "total_ticks": 150
                },
                "metadata": {"stage": "finals"}
            }
            filepath = Path(tmpdir) / "replay_meta.json"
            with open(filepath, 'w') as f:
                json.dump(replay_data, f)

            info = store.get_replay_info("replay_meta.json")
            assert info["filename"] == "replay_meta.json"
            assert info["fighter_a"] == "Alpha"
            assert info["fighter_b"] == "Beta"
            assert info["winner"] == "fighter_a"
            assert info["total_ticks"] == 150
            assert info["metadata"]["stage"] == "finals"


class TestConvenienceFunctions:
    """Tests for save_replay and load_replay convenience functions."""

    def test_save_and_load_compressed(self):
        """Test save_replay and load_replay with compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_replay.json")

            telemetry = {"ticks": [], "events": []}
            result = Mock()
            result.winner = "fighter_a"
            result.total_ticks = 100
            result.final_hp_a = 80
            result.final_hp_b = 0
            result.events = []

            # Save with compression (default)
            save_replay(telemetry, result, filepath)

            # File should have .gz extension added
            actual_path = filepath + ".gz"
            assert Path(actual_path).exists()

            # Load it back
            loaded = load_replay(actual_path)
            assert loaded["result"]["winner"] == "fighter_a"
            assert loaded["result"]["total_ticks"] == 100

    def test_save_and_load_uncompressed(self):
        """Test save_replay and load_replay without compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_replay.json")

            telemetry = {"ticks": [], "events": []}
            result = Mock()
            result.winner = "fighter_b"
            result.total_ticks = 200
            result.final_hp_a = 0
            result.final_hp_b = 50
            result.events = []

            # Save without compression
            save_replay(telemetry, result, filepath, compress=False)

            # File should NOT have .gz extension
            assert Path(filepath).exists()
            assert not Path(filepath + ".gz").exists()

            # Load it back
            loaded = load_replay(filepath)
            assert loaded["result"]["winner"] == "fighter_b"

    def test_save_with_metadata(self):
        """Test save_replay with custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_replay.json")

            telemetry = {"ticks": []}
            result = Mock()
            result.winner = "draw"
            result.total_ticks = 300
            result.final_hp_a = 50
            result.final_hp_b = 50
            result.events = []

            metadata = {"tournament": "finals", "round": 3}

            save_replay(telemetry, result, filepath, compress=False, metadata=metadata)

            loaded = load_replay(filepath)
            assert loaded["metadata"]["tournament"] == "finals"
            assert loaded["metadata"]["round"] == 3

    def test_load_compressed_file(self):
        """Test load_replay with .gz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json.gz"

            data = {"version": "1.0", "test": True}
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f)

            loaded = load_replay(str(filepath))
            assert loaded["test"] == True

    def test_load_uncompressed_file(self):
        """Test load_replay with plain .json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            data = {"version": "1.0", "uncompressed": True}
            with open(filepath, 'w') as f:
                json.dump(data, f)

            loaded = load_replay(str(filepath))
            assert loaded["uncompressed"] == True

    def test_save_preserves_telemetry_structure(self):
        """Test that save_replay preserves telemetry structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test.json")

            telemetry = {
                "ticks": [{"tick": 0}, {"tick": 1}],
                "events": [{"type": "HIT", "damage": 10}],
                "config": {"arena_width": 12.0}
            }
            result = Mock()
            result.winner = "fighter_a"
            result.total_ticks = 2
            result.final_hp_a = 90
            result.final_hp_b = 80
            result.events = [{"type": "COLLISION"}]

            save_replay(telemetry, result, filepath, compress=False)

            loaded = load_replay(filepath)
            assert loaded["telemetry"]["ticks"] == telemetry["ticks"]
            assert loaded["telemetry"]["events"] == telemetry["events"]
            assert loaded["events"] == [{"type": "COLLISION"}]
