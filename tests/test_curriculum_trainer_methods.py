"""
Tests for CurriculumTrainer class methods.
Focuses on testable helper methods and data structures.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.trainers.curriculum_trainer import (
    CurriculumTrainer,
    CurriculumLevel,
    DifficultyLevel,
    TrainingProgress,
    VmapEnvAdapter
)
from src.arena import WorldConfig


class TestVmapEnvAdapterVecEnvInterface:
    """Test VmapEnvAdapter implements VecEnv interface correctly."""

    def test_adapter_implements_reset(self):
        """Test adapter has reset method."""
        assert hasattr(VmapEnvAdapter, 'reset')
        assert callable(VmapEnvAdapter.reset)

    def test_adapter_implements_step_async(self):
        """Test adapter has step_async method for async stepping."""
        assert hasattr(VmapEnvAdapter, 'step_async')
        assert callable(VmapEnvAdapter.step_async)

    def test_adapter_implements_step_wait(self):
        """Test adapter has step_wait method to complete async step."""
        assert hasattr(VmapEnvAdapter, 'step_wait')
        assert callable(VmapEnvAdapter.step_wait)

    def test_adapter_implements_close(self):
        """Test adapter has close method for cleanup."""
        assert hasattr(VmapEnvAdapter, 'close')
        assert callable(VmapEnvAdapter.close)

    def test_adapter_implements_env_is_wrapped(self):
        """Test adapter has env_is_wrapped check method."""
        assert hasattr(VmapEnvAdapter, 'env_is_wrapped')
        assert callable(VmapEnvAdapter.env_is_wrapped)

    def test_adapter_implements_get_attr(self):
        """Test adapter has get_attr for accessing environment attributes."""
        assert hasattr(VmapEnvAdapter, 'get_attr')
        assert callable(VmapEnvAdapter.get_attr)

    def test_adapter_implements_set_attr(self):
        """Test adapter has set_attr for setting environment attributes."""
        assert hasattr(VmapEnvAdapter, 'set_attr')
        assert callable(VmapEnvAdapter.set_attr)

    def test_adapter_implements_env_method(self):
        """Test adapter has env_method for calling environment methods."""
        assert hasattr(VmapEnvAdapter, 'env_method')
        assert callable(VmapEnvAdapter.env_method)


class TestCurriculumLevelStructure:
    """Test CurriculumLevel dataclass structure and validation."""

    def test_level_stores_opponent_list(self):
        """Test curriculum level stores list of opponent file paths."""
        level = CurriculumLevel(
            name="Multi-Opponent Level",
            difficulty=DifficultyLevel.INTERMEDIATE,
            opponents=[
                "fighters/test_dummies/atomic/stationary_neutral.py",
                "fighters/test_dummies/atomic/approach_slow.py",
                "fighters/test_dummies/atomic/circle_left.py"
            ]
        )

        assert len(level.opponents) == 3
        assert all(isinstance(opp, str) for opp in level.opponents)

    def test_level_stores_graduation_requirements(self):
        """Test level stores graduation win rate and episode requirements."""
        level = CurriculumLevel(
            name="Strict Graduation",
            difficulty=DifficultyLevel.ADVANCED,
            opponents=["test.py"],
            graduation_win_rate=0.85,
            graduation_episodes=30
        )

        assert level.graduation_win_rate == 0.85
        assert level.graduation_episodes == 30

    def test_level_stores_episode_limits(self):
        """Test level stores minimum and maximum episode limits."""
        level = CurriculumLevel(
            name="Episode Limits",
            difficulty=DifficultyLevel.EXPERT,
            opponents=["expert.py"],
            min_episodes=500,
            max_episodes=5000
        )

        assert level.min_episodes == 500
        assert level.max_episodes == 5000
        assert level.max_episodes > level.min_episodes

    def test_level_optional_description(self):
        """Test level description is optional with default."""
        level_no_desc = CurriculumLevel(
            name="No Description",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["test.py"]
        )

        level_with_desc = CurriculumLevel(
            name="With Description",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["test.py"],
            description="This level tests fundamentals of combat"
        )

        assert level_no_desc.description == ""
        assert level_with_desc.description == "This level tests fundamentals of combat"


class TestTrainingProgressStateTracking:
    """Test TrainingProgress tracks training state correctly."""

    def test_progress_tracks_current_level_index(self):
        """Test progress tracks which curriculum level is active."""
        progress = TrainingProgress(current_level=3)

        assert progress.current_level == 3

    def test_progress_tracks_level_specific_stats(self):
        """Test progress tracks stats for current level separately."""
        progress = TrainingProgress(
            current_level=2,
            episodes_at_level=250,
            wins_at_level=180
        )

        level_win_rate = progress.wins_at_level / progress.episodes_at_level

        assert level_win_rate == 0.72  # 180/250

    def test_progress_tracks_cumulative_stats(self):
        """Test progress tracks cumulative stats across all levels."""
        progress = TrainingProgress(
            current_level=3,
            episodes_at_level=100,
            wins_at_level=70,
            total_episodes=600,
            total_wins=450
        )

        overall_win_rate = progress.total_wins / progress.total_episodes

        assert overall_win_rate == 0.75  # 450/600
        assert progress.total_episodes > progress.episodes_at_level
        assert progress.total_wins > progress.wins_at_level

    def test_progress_tracks_recent_episode_outcomes(self):
        """Test progress maintains sliding window of recent results."""
        recent_results = [True] * 15 + [False] * 5  # 15 wins, 5 losses

        progress = TrainingProgress(
            current_level=1,
            recent_episodes=recent_results
        )

        assert len(progress.recent_episodes) == 20
        recent_wins = sum(progress.recent_episodes)
        assert recent_wins == 15

    def test_progress_tracks_graduated_levels_list(self):
        """Test progress maintains list of completed levels."""
        progress = TrainingProgress(
            current_level=4,
            graduated_levels=[
                "Level 1: Fundamentals",
                "Level 2: Basic Skills",
                "Level 3: Intermediate",
                "Level 4: Advanced"
            ]
        )

        assert len(progress.graduated_levels) == 4
        assert progress.current_level == 4
        assert all(isinstance(level_name, str) for level_name in progress.graduated_levels)

    def test_progress_tracks_training_start_time(self):
        """Test progress records when training started."""
        import time

        before = time.time()
        progress = TrainingProgress()
        after = time.time()

        assert progress.start_time >= before
        assert progress.start_time <= after

    def test_progress_empty_state_at_start(self):
        """Test progress starts with empty/zero state."""
        progress = TrainingProgress()

        assert progress.current_level == 0
        assert progress.episodes_at_level == 0
        assert progress.wins_at_level == 0
        assert progress.total_episodes == 0
        assert progress.total_wins == 0
        assert len(progress.recent_episodes) == 0
        assert len(progress.graduated_levels) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
