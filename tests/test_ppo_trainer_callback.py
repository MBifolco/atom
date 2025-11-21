"""
Comprehensive tests for PPO trainer callback and related utilities.
"""

import pytest
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.training.trainers.ppo.trainer import VerboseLoggingCallback


class TestVerboseLoggingCallback:
    """Tests for VerboseLoggingCallback class."""

    def test_callback_initialization(self):
        """Test callback initializes with required parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Boxer", "Rusher"],
                verbose=1
            )
            assert callback.log_path == log_path
            assert callback.opponent_names == ["Boxer", "Rusher"]
            assert callback.episode_count == 0

    def test_callback_initialization_empty_opponents(self):
        """Test callback with empty opponent list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=[],
                verbose=0
            )
            assert callback.opponent_names == []
            assert callback.verbose == 0

    def test_callback_on_training_start(self):
        """Test _on_training_start creates log file and logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Boxer"],
                verbose=1
            )

            # Mock training_env via patching the property
            mock_env = Mock()
            mock_env.num_envs = 4
            with patch.object(type(callback), 'training_env', new_callable=lambda: property(lambda self: mock_env)):
                callback._on_training_start()

            assert callback.file_logger is not None
            assert Path(log_path).exists()

            # Check log content
            with open(log_path) as f:
                content = f.read()
            assert "TRAINING SESSION STARTED" in content
            assert "Boxer" in content

    def test_callback_on_step_no_episodes(self):
        """Test _on_step with no episode completions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Boxer"],
                verbose=1
            )

            # Setup mock logger
            callback.file_logger = Mock()
            callback.locals = {"infos": [{}]}

            result = callback._on_step()

            assert result is True  # Should continue training
            assert callback.episode_count == 0

    def test_callback_on_step_with_episode(self):
        """Test _on_step when episode completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Boxer", "Rusher"],
                verbose=1
            )

            # Setup mock logger
            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {"episode": {"r": 50.0, "l": 100}}
                ]
            }

            result = callback._on_step()

            assert result is True
            assert callback.episode_count == 1
            callback.file_logger.debug.assert_called()

    def test_callback_on_step_with_hp_info(self):
        """Test _on_step logs HP information when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Boxer"],
                verbose=1
            )

            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {
                        "episode": {"r": 100.0, "l": 150},
                        "fighter_hp": 75.0,
                        "opponent_hp": 25.0
                    }
                ]
            }

            callback._on_step()

            # Check debug was called multiple times (once for episode, once for HP)
            assert callback.file_logger.debug.call_count >= 2

    def test_callback_on_step_with_damage_info(self):
        """Test _on_step logs damage information when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Tank"],
                verbose=1
            )

            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {
                        "episode": {"r": 80.0, "l": 200},
                        "episode_damage_dealt": 120.0,
                        "episode_damage_taken": 50.0
                    }
                ]
            }

            callback._on_step()

            assert callback.file_logger.debug.call_count >= 2

    def test_callback_on_step_with_win_info(self):
        """Test _on_step logs win/loss result when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Counter"],
                verbose=1
            )

            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {
                        "episode": {"r": 100.0, "l": 120},
                        "won": True
                    }
                ]
            }

            callback._on_step()

            assert callback.file_logger.debug.call_count >= 2

    def test_callback_on_step_multiple_envs(self):
        """Test _on_step with multiple environments completing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["A", "B", "C", "D"],
                verbose=1
            )

            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {"episode": {"r": 50.0, "l": 100}},
                    {},  # No episode completion
                    {"episode": {"r": -20.0, "l": 80}},
                    {"episode": {"r": 100.0, "l": 150}}
                ]
            }

            callback._on_step()

            # Should count 3 episode completions
            assert callback.episode_count == 3

    def test_callback_opponent_cycling(self):
        """Test opponent names cycle correctly with multiple envs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["OpponentA", "OpponentB"],
                verbose=1
            )

            callback.file_logger = Mock()

            # Env 0 -> OpponentA (0 % 2 = 0)
            # Env 1 -> OpponentB (1 % 2 = 1)
            # Env 2 -> OpponentA (2 % 2 = 0)
            callback.locals = {
                "infos": [
                    {"episode": {"r": 50.0, "l": 100}},
                    {"episode": {"r": 60.0, "l": 110}},
                    {"episode": {"r": 70.0, "l": 120}},
                ]
            }

            callback._on_step()

            # Check that opponent names were used correctly in logging
            calls = callback.file_logger.debug.call_args_list
            call_strings = [str(call) for call in calls]
            assert any("OpponentA" in s for s in call_strings)
            assert any("OpponentB" in s for s in call_strings)


class TestVerboseLoggingCallbackEdgeCases:
    """Edge case tests for VerboseLoggingCallback."""

    def test_callback_with_long_opponent_names(self):
        """Test with very long opponent names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            long_names = [f"VeryLongOpponentName_{i}" * 5 for i in range(10)]
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=long_names,
                verbose=1
            )
            assert len(callback.opponent_names) == 10

    def test_callback_log_path_in_nested_dir(self):
        """Test creating log in nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "logs", "training", "session1", "training.log")
            os.makedirs(os.path.dirname(nested_path), exist_ok=True)

            callback = VerboseLoggingCallback(
                log_path=nested_path,
                opponent_names=["Test"],
                verbose=1
            )

            mock_env = Mock()
            mock_env.num_envs = 1
            with patch.object(type(callback), 'training_env', new_callable=lambda: property(lambda self: mock_env)):
                callback._on_training_start()

            assert Path(nested_path).exists()

    def test_callback_continues_training_on_step(self):
        """Test that _on_step always returns True to continue training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Test"],
                verbose=1
            )
            callback.file_logger = Mock()
            callback.locals = {"infos": []}

            for _ in range(10):
                result = callback._on_step()
                assert result is True

    def test_callback_with_full_episode_info(self):
        """Test callback with all possible episode info fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Full"],
                verbose=1
            )

            callback.file_logger = Mock()
            callback.locals = {
                "infos": [
                    {
                        "episode": {"r": 150.0, "l": 250},
                        "fighter_hp": 100.0,
                        "opponent_hp": 0.0,
                        "episode_damage_dealt": 200.0,
                        "episode_damage_taken": 0.0,
                        "won": True
                    }
                ]
            }

            callback._on_step()

            # Should have logged: episode basic + HP + damage + win = 4 debug calls
            assert callback.file_logger.debug.call_count >= 4


class TestCallbackIntegration:
    """Integration tests for callback behavior."""

    def test_training_session_logging(self):
        """Test a simulated training session with logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "session.log")
            callback = VerboseLoggingCallback(
                log_path=log_path,
                opponent_names=["Alpha", "Beta"],
                verbose=1
            )

            # Start training
            mock_env = Mock()
            mock_env.num_envs = 2
            with patch.object(type(callback), 'training_env', new_callable=lambda: property(lambda self: mock_env)):
                callback._on_training_start()

            # Simulate multiple steps
            for step in range(5):
                callback.locals = {
                    "infos": [
                        {"episode": {"r": 50.0 + step * 10, "l": 100 + step * 20}},
                        {}
                    ]
                }
                callback._on_step()

            assert callback.episode_count == 5

            # Verify log file has content
            with open(log_path) as f:
                content = f.read()
            assert "Episode 1" in content
            assert "Episode 5" in content
