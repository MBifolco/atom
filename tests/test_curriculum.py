"""
Tests for curriculum training system to boost coverage.
"""

import pytest
import time
from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    CurriculumLevel,
    TrainingProgress,
    VmapEnvAdapter
)


def simple_policy_func(state):
    """Simple policy for curriculum level testing."""
    return {"acceleration": 0.0, "stance": "neutral"}


class TestDifficultyLevel:
    """Test difficulty level enum."""

    def test_difficulty_levels_defined(self):
        """Test all difficulty levels are defined."""
        assert DifficultyLevel.FUNDAMENTALS.value == "fundamentals"
        assert DifficultyLevel.BASIC_SKILLS.value == "basic_skills"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"
        assert DifficultyLevel.POPULATION.value == "population"

    def test_all_levels_accessible(self):
        """Test can access all difficulty level members."""
        levels = list(DifficultyLevel)
        assert len(levels) == 6


class TestCurriculumLevel:
    """Test CurriculumLevel dataclass."""

    def test_curriculum_level_creation(self):
        """Test creating a curriculum level."""
        level = CurriculumLevel(
            name="Test Level",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponent_policy=simple_policy_func,
            opponent_mass=70.0,
            max_episodes=1000
        )

        assert level.name == "Test Level"
        assert level.difficulty == DifficultyLevel.FUNDAMENTALS
        assert level.opponent_mass == 70.0
        assert level.max_episodes == 1000

    def test_curriculum_level_defaults(self):
        """Test curriculum level default values."""
        level = CurriculumLevel(
            name="Test",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponent_policy=simple_policy_func,
            opponent_mass=70.0
        )

        assert level.min_episodes == 100  # Default
        assert level.graduation_win_rate == 0.7  # Default
        assert level.graduation_episodes == 20  # Default

    def test_curriculum_level_custom_values(self):
        """Test curriculum level with custom graduation settings."""
        level = CurriculumLevel(
            name="Custom",
            difficulty=DifficultyLevel.ADVANCED,
            opponent_policy=simple_policy_func,
            opponent_mass=75.0,
            max_episodes=2000,
            min_episodes=200,
            graduation_win_rate=0.8,
            graduation_episodes=30,
            description="Custom test level"
        )

        assert level.graduation_win_rate == 0.8
        assert level.graduation_episodes == 30
        assert level.description == "Custom test level"


class TestTrainingProgress:
    """Test TrainingProgress dataclass."""

    def test_training_progress_creation(self):
        """Test creating training progress tracker."""
        progress = TrainingProgress(
            current_level=0,
            episodes_at_level=50,
            wins_at_level=30,
            recent_episodes=[True, False, True, True],
            graduated_levels=["Level 1"],
            total_episodes=100,
            total_wins=60
        )

        assert progress.current_level == 0
        assert progress.episodes_at_level == 50
        assert progress.wins_at_level == 30
        assert len(progress.recent_episodes) == 4
        assert progress.total_episodes == 100

    def test_training_progress_defaults(self):
        """Test training progress default values."""
        progress = TrainingProgress()

        assert progress.current_level == 0
        assert progress.episodes_at_level == 0
        assert progress.wins_at_level == 0
        assert progress.recent_episodes == []
        assert progress.graduated_levels == []
        assert progress.total_episodes == 0
        assert progress.total_wins == 0

    def test_training_progress_start_time(self):
        """Test training progress has start time."""
        progress = TrainingProgress()

        assert hasattr(progress, 'start_time')
        assert progress.start_time > 0


class TestVmapEnvAdapter:
    """Test vmap environment adapter."""

    def test_vmap_adapter_class_exists(self):
        """Test vmap adapter can be imported."""
        assert VmapEnvAdapter is not None

    def test_vmap_adapter_has_methods(self):
        """Test vmap adapter has required VecEnv methods."""
        assert hasattr(VmapEnvAdapter, 'reset')
        assert hasattr(VmapEnvAdapter, 'step_async')
        assert hasattr(VmapEnvAdapter, 'step_wait')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

