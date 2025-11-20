"""
Comprehensive PPO trainer tests to boost coverage.
"""

import pytest
import tempfile
import logging
from pathlib import Path
from src.training.trainers.ppo.trainer import VerboseLoggingCallback
from stable_baselines3.common.callbacks import BaseCallback


class TestVerboseLoggingCallbackComplete:
    """Complete coverage of VerboseLoggingCallback."""

    def test_callback_initialization_complete(self):
        """Test callback full initialization."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Opp1", "Opp2", "Opp3"],
                verbose=2
            )

            assert callback.log_path == log_path
            assert len(callback.opponent_names) == 3
            assert callback.opponent_names == ["Opp1", "Opp2", "Opp3"]
            assert callback.episode_count == 0
            assert callback.file_logger is None  # Not initialized until training starts
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_callback_inherits_from_base_callback(self):
        """Test callback is a BaseCallback subclass."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["Test"]
            )

            assert isinstance(callback, BaseCallback)
            assert hasattr(callback, '_on_training_start')
            assert hasattr(callback, '_on_step')

    def test_callback_has_episode_tracking(self):
        """Test callback tracks episode count."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["A", "B"]
            )

            assert hasattr(callback, 'episode_count')
            assert callback.episode_count == 0

    def test_callback_stores_opponent_names(self):
        """Test callback stores opponent names list."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            opponents = ["Fighter1", "Fighter2", "Fighter3", "Fighter4"]
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=opponents,
                verbose=1
            )

            assert callback.opponent_names == opponents
            assert len(callback.opponent_names) == 4

    def test_callback_different_verbosity_levels(self):
        """Test callback with different verbosity settings."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            # Verbose level 0
            callback0 = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["Test"],
                verbose=0
            )
            assert callback0.verbose == 0

            # Verbose level 1
            callback1 = VerboseLoggingCallback(
                log_path=f.name + ".1",
                opponent_names=["Test"],
                verbose=1
            )
            assert callback1.verbose == 1

            # Verbose level 2
            callback2 = VerboseLoggingCallback(
                log_path=f.name + ".2",
                opponent_names=["Test"],
                verbose=2
            )
            assert callback2.verbose == 2

    def test_callback_log_path_validation(self):
        """Test callback accepts valid log path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "training.log")

            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Test"]
            )

            assert callback.log_path == log_path

    def test_callback_with_single_opponent(self):
        """Test callback with single opponent."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["SingleOpponent"]
            )

            assert len(callback.opponent_names) == 1

    def test_callback_with_many_opponents(self):
        """Test callback with many opponents."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            opponents = [f"Opponent{i}" for i in range(10)]
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=opponents
            )

            assert len(callback.opponent_names) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
