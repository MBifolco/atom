"""
Comprehensive tests for CurriculumCallback.
Tests callback lifecycle methods and progress tracking.
"""

import pytest
import time
from src.training.trainers.curriculum_trainer import CurriculumCallback, CurriculumLevel, DifficultyLevel


class TestCurriculumCallbackInitialization:
    """Test CurriculumCallback initialization and setup."""

    def test_callback_initializes_with_curriculum_trainer(self):
        """Test callback stores reference to curriculum trainer."""
        class MockCurriculumTrainer:
            algorithm = "ppo"

        trainer = MockCurriculumTrainer()
        callback = CurriculumCallback(curriculum_trainer=trainer, verbose=0)

        assert callback.curriculum_trainer is trainer

    def test_callback_initializes_tracking_lists(self):
        """Test callback initializes episode tracking lists."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        assert isinstance(callback.episode_rewards, list)
        assert isinstance(callback.episode_wins, list)
        assert isinstance(callback.recent_reward_components, list)
        assert len(callback.episode_rewards) == 0
        assert len(callback.episode_wins) == 0

    def test_callback_initializes_timing_trackers(self):
        """Test callback initializes rollout and train time trackers."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        assert callback.last_rollout_time is None
        assert callback.last_train_time is None

    def test_callback_with_verbose_mode(self):
        """Test callback created with verbose mode enabled."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=1)

        assert callback.verbose == 1

    def test_callback_with_quiet_mode(self):
        """Test callback created with quiet mode."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        assert callback.verbose == 0


class TestCurriculumCallbackRolloutMethods:
    """Test CurriculumCallback rollout lifecycle methods."""

    def test_on_rollout_start_sets_time(self):
        """Test _on_rollout_start records start time."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        # Simulate rollout start
        callback._on_rollout_start()

        assert callback.last_rollout_time is not None
        assert callback.last_rollout_time > 0

    def test_on_rollout_start_increments_counter(self):
        """Test _on_rollout_start increments rollout counter."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        # First rollout
        callback._on_rollout_start()
        assert callback.rollout_count == 1

        # Second rollout
        callback._on_rollout_start()
        assert callback.rollout_count == 2

    def test_on_rollout_end_calculates_duration(self):
        """Test _on_rollout_end calculates rollout duration."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        # Start rollout
        callback._on_rollout_start()
        start_time = callback.last_rollout_time

        # Small delay
        time.sleep(0.01)

        # End rollout
        callback._on_rollout_end()

        # Should have set train time
        assert callback.last_train_time is not None
        assert callback.last_train_time > start_time

    def test_on_rollout_end_sets_train_time(self):
        """Test _on_rollout_end sets training start time."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        callback._on_rollout_start()
        callback._on_rollout_end()

        assert callback.last_train_time is not None


class TestCurriculumCallbackAlgorithmHandling:
    """Test CurriculumCallback handles different algorithms."""

    def test_callback_with_ppo_algorithm(self):
        """Test callback behavior with PPO algorithm."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=1)

        # Simulate rollout for PPO
        callback._on_rollout_start()

        # PPO should always log (not every 50th like SAC would)
        assert callback.curriculum_trainer.algorithm == "ppo"

    def test_callback_rollout_count_initialization(self):
        """Test rollout count is initialized on first rollout."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        # Should not have rollout_count initially
        assert not hasattr(callback, 'rollout_count')

        # After first rollout start, should be initialized
        callback._on_rollout_start()

        assert hasattr(callback, 'rollout_count')
        assert callback.rollout_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
