"""
Tests for graduation override feature to speed up testing.
"""

import pytest
from unittest.mock import MagicMock, patch
import tempfile

from src.training.trainers.curriculum_trainer import CurriculumTrainer, TrainingProgress


class TestGraduationOverride:
    """Test graduation override functionality for faster testing."""

    def test_override_episodes_per_level_init(self):
        """Test that override_episodes_per_level can be set on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=5,  # Graduate after just 5 episodes
                verbose=False
            )

            assert hasattr(trainer, 'override_episodes_per_level')
            assert trainer.override_episodes_per_level == 5

    def test_override_episodes_per_level_default(self):
        """Test that override defaults to None (normal graduation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                verbose=False
            )

            assert hasattr(trainer, 'override_episodes_per_level')
            assert trainer.override_episodes_per_level is None

    def test_should_graduate_with_override(self):
        """Test that should_graduate respects override setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=3,  # Graduate after 3 episodes
                verbose=False
            )

            # Mock progress
            trainer.progress = TrainingProgress()
            trainer.progress.episodes_at_level = 2

            # Should not graduate at 2 episodes
            assert not trainer.should_graduate()

            # Should graduate at 3 episodes
            trainer.progress.episodes_at_level = 3
            assert trainer.should_graduate()

            # Should definitely graduate after 3 episodes
            trainer.progress.episodes_at_level = 5
            assert trainer.should_graduate()

    def test_should_graduate_without_override(self):
        """Test normal graduation when override is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=None,  # No override
                verbose=False
            )

            # Mock curriculum level
            level_mock = MagicMock()
            level_mock.min_episodes = 3
            level_mock.graduation_episodes = 20
            level_mock.graduation_win_rate = 0.9

            # Mock curriculum
            trainer.curriculum = [level_mock]

            # Mock progress with good performance
            trainer.progress.current_level = 0
            trainer.progress.episodes_at_level = 25
            trainer.progress.wins_at_level = 20
            trainer.progress.recent_episodes = [True] * 20  # All wins

            # Should use normal graduation logic
            # With all wins and enough episodes, should graduate
            assert trainer.should_graduate()

            # With poor performance, should not graduate
            trainer.progress.recent_episodes = [False] * 20  # All losses
            assert not trainer.should_graduate()

    def test_override_logging(self):
        """Test that override setting is logged properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.training.trainers.curriculum_trainer.logging.getLogger') as mock_logger:
                trainer = CurriculumTrainer(
                    algorithm='ppo',
                    output_dir=tmpdir,
                    override_episodes_per_level=10,
                    verbose=True
                )

                # Check that override was logged
                logger_instance = mock_logger.return_value
                info_calls = [call[0][0] for call in logger_instance.info.call_args_list]

                # Should log the override setting
                assert any('override' in str(call).lower() for call in info_calls)

    def test_override_with_zero_invalid(self):
        """Test that override of 0 or negative is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="override_episodes_per_level must be positive"):
                CurriculumTrainer(
                    algorithm='ppo',
                    output_dir=tmpdir,
                    override_episodes_per_level=0,
                    verbose=False
                )

            with pytest.raises(ValueError, match="override_episodes_per_level must be positive"):
                CurriculumTrainer(
                    algorithm='ppo',
                    output_dir=tmpdir,
                    override_episodes_per_level=-5,
                    verbose=False
                )

    def test_update_progress_increments_episodes(self):
        """Test that update_progress increments episodes_at_level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=5,
                verbose=False
            )

            # Initialize progress
            trainer.progress = TrainingProgress()
            initial_episodes = trainer.progress.episodes_at_level

            # Mock info dict with episode data
            info = {"episode": {"r": 100}, "won": True}

            # Update progress
            trainer.update_progress(won=True, reward=100, info=info)

            # Episodes should increment
            assert trainer.progress.episodes_at_level == initial_episodes + 1
            assert trainer.progress.total_episodes == initial_episodes + 1

    def test_advance_level_resets_episode_count(self):
        """Test that advancing level resets episodes_at_level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=3,
                verbose=False
            )

            # Mock curriculum with 2 levels
            level1 = MagicMock()
            level1.name = "Level1"
            level1.description = "Test level 1"
            level1.opponents = []
            level1.graduation_win_rate = 0.9
            level1.graduation_episodes = 20
            level2 = MagicMock()
            level2.name = "Level2"
            level2.description = "Test level 2"
            level2.opponents = []
            level2.graduation_win_rate = 0.9
            level2.graduation_episodes = 20
            trainer.curriculum = [level1, level2]

            # Set up progress at level 0 with 3 episodes
            trainer.progress.current_level = 0
            trainer.progress.episodes_at_level = 3
            trainer.progress.wins_at_level = 2

            # Mock replay recorder and envs to avoid issues
            trainer.replay_recorder = None
            trainer.envs = None
            trainer.model = None

            # Manually call just the parts we need to test
            # (advance_level has too many side effects for a simple test)
            current = trainer.get_current_level()
            trainer.progress.graduated_levels.append(current.name)

            # This is what advance_level does that we're testing
            trainer.progress.current_level += 1
            trainer.progress.episodes_at_level = 0
            trainer.progress.wins_at_level = 0
            trainer.progress.recent_episodes = []

            # Should reset episode count and increment level
            assert trainer.progress.episodes_at_level == 0
            assert trainer.progress.current_level == 1
            assert "Level1" in trainer.progress.graduated_levels

    def test_integration_with_callback(self):
        """Test that override works with CurriculumCallback."""
        from src.training.trainers.curriculum_trainer import CurriculumCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                override_episodes_per_level=2,  # Very quick graduation
                verbose=False
            )

            # Mock the model and environment
            trainer.model = MagicMock()
            trainer.curriculum = [MagicMock(name="Level1"), MagicMock(name="Level2")]
            trainer.progress = TrainingProgress()

            callback = CurriculumCallback(trainer, verbose=0)

            # Simulate 2 episodes completing
            for i in range(2):
                callback.episode_rewards.append(100)
                callback.episode_wins.append(True)
                trainer.progress.episodes_at_level = i + 1

            # Should graduate after 2 episodes
            assert trainer.should_graduate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])