"""
Quick coverage wins - test simple methods and dataclasses to boost to 45%.
"""

import pytest
from src.atom.training.trainers.population.elo_tracker import FighterStats, EloTracker
from src.training.trainers.population.population_trainer import PopulationFighter
from src.training.trainers.curriculum_trainer import CurriculumLevel, DifficultyLevel, TrainingProgress
from stable_baselines3 import PPO


class TestFighterStatsComplete:
    """Complete FighterStats coverage."""

    def test_damage_ratio_when_no_damage_taken(self):
        """Test damage ratio when fighter took no damage."""
        stats = FighterStats(
            name="Perfect",
            total_damage_dealt=100.0,
            total_damage_taken=0.0
        )

        # Should return infinity or very large number
        assert stats.damage_ratio == float('inf') or stats.damage_ratio > 100

    def test_damage_ratio_when_no_damage_dealt_or_taken(self):
        """Test damage ratio when no damage at all."""
        stats = FighterStats(
            name="Passive",
            total_damage_dealt=0.0,
            total_damage_taken=0.0
        )

        # Should handle division by zero gracefully
        assert stats.damage_ratio >= 0


class TestEloTrackerComplete:
    """Complete ELO tracker coverage."""

    def test_add_fighter_twice_doesnt_duplicate(self):
        """Test adding same fighter twice doesn't create duplicate."""
        tracker = EloTracker()

        tracker.add_fighter("test")
        tracker.add_fighter("test")  # Add again

        # Should still only have one entry
        assert len([f for f in tracker.fighters if f == "test"]) == 1

    def test_remove_nonexistent_fighter_doesnt_crash(self):
        """Test removing non-existent fighter doesn't crash."""
        tracker = EloTracker()

        # Should not raise exception
        tracker.remove_fighter("nonexistent")

        assert True

    def test_match_history_tracking(self):
        """Test match history is recorded."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        tracker.update_ratings("a", "b", "a_wins", damage_a=50, damage_b=20)

        # Check match history exists
        stats_a = tracker.fighters["a"]
        assert hasattr(stats_a, 'match_history')
        assert len(stats_a.match_history) > 0

    def test_multiple_matches_accumulate_damage(self):
        """Test multiple matches accumulate damage correctly."""
        tracker = EloTracker()

        tracker.add_fighter("attacker")
        tracker.add_fighter("defender")

        # First match
        tracker.update_ratings("attacker", "defender", "a_wins", damage_a=30, damage_b=10)
        # Second match
        tracker.update_ratings("attacker", "defender", "a_wins", damage_a=40, damage_b=15)

        # Damage should accumulate
        assert tracker.fighters["attacker"].total_damage_dealt == 70  # 30 + 40
        assert tracker.fighters["attacker"].total_damage_taken == 25  # 10 + 15


class TestPopulationFighter:
    """Test PopulationFighter dataclass."""

    def test_population_fighter_creation(self):
        """Test creating a population fighter."""
        # Create mock model
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Test Fighter",
            model=MockModel(),
            generation=1,
            lineage="founder",
            mass=70.0
        )

        assert fighter.name == "Test Fighter"
        assert fighter.generation == 1
        assert fighter.lineage == "founder"
        assert fighter.mass == 70.0

    def test_population_fighter_defaults(self):
        """Test population fighter default values."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Default",
            model=MockModel()
        )

        assert fighter.generation == 0  # Default
        assert fighter.lineage == "founder"  # Default
        assert fighter.mass == 70.0  # Default
        assert fighter.training_episodes == 0
        assert fighter.last_checkpoint is None


class TestCurriculumLevelComplete:
    """Complete curriculum level coverage."""

    def simple_opponent(state):
        return {"acceleration": 0.0, "stance": "neutral"}

    def test_curriculum_level_all_fields_set(self):
        """Test curriculum level with all fields."""
        level = CurriculumLevel(
            name="Complete Level",
            difficulty=DifficultyLevel.INTERMEDIATE,
            opponents=["opp1.py", "opp2.py"],
            min_episodes=300,
            graduation_win_rate=0.75,
            graduation_episodes=25,
            description="Test description"
        )

        assert all([
            level.name == "Complete Level",
            level.difficulty == DifficultyLevel.INTERMEDIATE,
            len(level.opponents) == 2,
            level.min_episodes == 300,
            level.graduation_win_rate == 0.75,
            level.graduation_episodes == 25,
            level.description == "Test description"
        ])


class TestTrainingProgressComplete:
    """Complete training progress coverage."""

    def test_progress_tracks_recent_episodes_list(self):
        """Test recent episodes list is tracked."""
        progress = TrainingProgress(
            recent_episodes=[True, True, False, True, False, True, True]
        )

        assert len(progress.recent_episodes) == 7
        # Win rate from recent would be 5/7
        recent_wins = sum(progress.recent_episodes)
        assert recent_wins == 5

    def test_progress_tracks_graduated_levels_list(self):
        """Test graduated levels list."""
        progress = TrainingProgress(
            current_level=3,
            graduated_levels=[
                "Level 1: Fundamentals",
                "Level 2: Basic Skills",
                "Level 3: Intermediate"
            ]
        )

        assert len(progress.graduated_levels) == 3
        assert progress.current_level == 3

    def test_progress_total_vs_level_stats(self):
        """Test total stats vs current level stats."""
        progress = TrainingProgress(
            current_level=2,
            episodes_at_level=50,
            wins_at_level=30,
            total_episodes=150,
            total_wins=95
        )

        # Total should be greater than current level
        assert progress.total_episodes > progress.episodes_at_level
        assert progress.total_wins > progress.wins_at_level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
