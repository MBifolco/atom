"""
Tests for PPO trainer module.
"""

import pytest
import tempfile
import logging
from pathlib import Path
from src.training.trainers.ppo.trainer import VerboseLoggingCallback
from stable_baselines3.common.callbacks import BaseCallback


class TestVerboseLoggingCallback:
    """Test verbose logging callback for training."""

    def test_callback_initialization(self):
        """Test callback initializes with log path."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Opponent1", "Opponent2"],
                verbose=1
            )

            assert callback.log_path == log_path
            assert callback.opponent_names == ["Opponent1", "Opponent2"]
            assert callback.episode_count == 0
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_callback_is_base_callback(self):
        """Test callback extends BaseCallback."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["Test"],
                verbose=1
            )

            assert isinstance(callback, BaseCallback)

    def test_callback_has_required_methods(self):
        """Test callback has required methods."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            callback = VerboseLoggingCallback(
                log_path=f.name,
                opponent_names=["Test"]
            )

            assert hasattr(callback, '_on_training_start')
            assert hasattr(callback, '_on_step')
            assert callable(callback._on_training_start)
            assert callable(callback._on_step)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
