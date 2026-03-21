"""
Comprehensive trainer tests focusing on data structures and simple methods.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.trainers.curriculum_trainer import (
    VmapEnvAdapter,
    DifficultyLevel,
    CurriculumLevel,
    TrainingProgress
)
from src.training.trainers.population.population_trainer import (
    PopulationFighter,
    PopulationCallback
)
from src.atom.training.trainers.population.elo_tracker import EloTracker, FighterStats


class TestVmapEnvAdapterMethods:
    """Test VmapEnvAdapter VecEnv interface methods."""

    def test_adapter_has_env_is_wrapped(self):
        """Test adapter implements env_is_wrapped."""
        assert hasattr(VmapEnvAdapter, 'env_is_wrapped')
        assert callable(VmapEnvAdapter.env_is_wrapped)

    def test_adapter_has_get_attr(self):
        """Test adapter implements get_attr."""
        assert hasattr(VmapEnvAdapter, 'get_attr')
        assert callable(VmapEnvAdapter.get_attr)

    def test_adapter_has_set_attr(self):
        """Test adapter implements set_attr."""
        assert hasattr(VmapEnvAdapter, 'set_attr')
        assert callable(VmapEnvAdapter.set_attr)

    def test_adapter_has_env_method(self):
        """Test adapter implements env_method."""
        assert hasattr(VmapEnvAdapter, 'env_method')
        assert callable(VmapEnvAdapter.env_method)

    def test_adapter_has_close(self):
        """Test adapter implements close."""
        assert hasattr(VmapEnvAdapter, 'close')
        assert callable(VmapEnvAdapter.close)


class TestPopulationCallbackMethods:
    """Test PopulationCallback methods."""

    def test_population_callback_initialization(self):
        """Test population callback initializes correctly."""
        tracker = EloTracker()

        callback = PopulationCallback(
            fighter_name="TestFighter",
            elo_tracker=tracker,
            verbose=1
        )

        assert callback.fighter_name == "TestFighter"
        assert callback.elo_tracker == tracker
        assert callback.episode_count == 0
        assert hasattr(callback, 'recent_rewards')

    def test_population_callback_has_on_step(self):
        """Test callback has _on_step method."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        assert hasattr(callback, '_on_step')
        assert callable(callback._on_step)

    def test_population_callback_tracks_rewards(self):
        """Test callback has reward tracking."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker, verbose=0)

        assert hasattr(callback, 'recent_rewards')
        assert isinstance(callback.recent_rewards, list)


class TestCurriculumLevelVariations:
    """Test curriculum level with various configurations."""

    def simple_policy(state):
        return {"acceleration": 0.0, "stance": "neutral"}

    def test_curriculum_level_minimum_fields(self):
        """Test curriculum level with only required fields."""
        level = CurriculumLevel(
            name="Minimal",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["test.py"]
        )

        # Should have defaults
        assert level.min_episodes > 0
        assert 0 < level.graduation_win_rate <= 1.0

    def test_curriculum_level_high_graduation_rate(self):
        """Test level with high graduation requirement."""
        level = CurriculumLevel(
            name="Hard",
            difficulty=DifficultyLevel.EXPERT,
            opponents=["expert.py"],
            graduation_win_rate=0.90,
            graduation_episodes=50
        )

        assert level.graduation_win_rate == 0.90
        assert level.graduation_episodes == 50

    def test_curriculum_level_with_description(self):
        """Test level description field."""
        level = CurriculumLevel(
            name="Described",
            difficulty=DifficultyLevel.INTERMEDIATE,
            opponents=["test.py"],
            description="This is a test level with a description"
        )

        assert level.description == "This is a test level with a description"

    def test_curriculum_level_with_multiple_opponents(self):
        """Test levels with multiple opponent files."""
        opponents = ["opp1.py", "opp2.py", "opp3.py", "opp4.py", "opp5.py"]

        level = CurriculumLevel(
            name="MultiOpponents",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=opponents
        )

        assert len(level.opponents) == 5


class TestTrainingProgressVariations:
    """Test training progress in different scenarios."""

    def test_progress_at_start_of_training(self):
        """Test progress at very beginning."""
        progress = TrainingProgress()

        assert progress.current_level == 0
        assert progress.total_episodes == 0
        assert progress.total_wins == 0
        assert progress.episodes_at_level == 0
        assert len(progress.graduated_levels) == 0

    def test_progress_mid_training(self):
        """Test progress in middle of training."""
        progress = TrainingProgress(
            current_level=2,
            episodes_at_level=150,
            wins_at_level=105,
            total_episodes=500,
            total_wins=350,
            graduated_levels=["Level 1", "Level 2"]
        )

        assert progress.current_level == 2
        assert len(progress.graduated_levels) == 2

    def test_progress_near_graduation(self):
        """Test progress approaching graduation."""
        # 18 of last 20 wins = 90% win rate
        recent = [True] * 18 + [False] * 2

        progress = TrainingProgress(
            current_level=3,
            episodes_at_level=300,
            wins_at_level=240,  # 80% overall
            recent_episodes=recent,
            total_episodes=800,
            total_wins=600
        )

        wins_recent = sum(progress.recent_episodes)
        recent_win_rate = wins_recent / len(progress.recent_episodes)

        assert recent_win_rate == 0.9  # 90%
        assert progress.wins_at_level / progress.episodes_at_level == 0.8  # 80% overall


class TestFighterStatsCalculations:
    """Test fighter stats property calculations."""

    def test_stats_perfect_record(self):
        """Test stats for undefeated fighter."""
        stats = FighterStats(
            name="Undefeated",
            wins=10,
            losses=0,
            draws=0
        )

        assert stats.matches_played == 10
        assert stats.win_rate == 1.0

    def test_stats_winless_record(self):
        """Test stats for fighter with no wins."""
        stats = FighterStats(
            name="Winless",
            wins=0,
            losses=8,
            draws=2
        )

        assert stats.matches_played == 10
        assert stats.win_rate == 0.0

    def test_stats_balanced_record(self):
        """Test stats for balanced fighter."""
        stats = FighterStats(
            name="Balanced",
            wins=5,
            losses=4,
            draws=1
        )

        assert stats.matches_played == 10
        assert stats.win_rate == 0.5

    def test_stats_damage_tracking_over_time(self):
        """Test cumulative damage tracking."""
        stats = FighterStats(
            name="Tracker",
            total_damage_dealt=500.0,
            total_damage_taken=300.0
        )

        assert stats.damage_ratio == pytest.approx(500.0 / 300.0)
        assert stats.damage_ratio > 1.0  # Dealing more than taking


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
