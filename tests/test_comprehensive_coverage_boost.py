"""
Comprehensive tests to push coverage from 41% to 45%+.
Focuses on testable code paths in training modules.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.arena import WorldConfig
from src.atom.training.trainers.population.elo_tracker import EloTracker, FighterStats
from src.atom.training.trainers.population.fighter_loader import (
    load_fighter,
    validate_fighter,
    FighterLoadError,
    load_hardcoded_fighters
)
from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    CurriculumLevel,
    TrainingProgress,
    CurriculumCallback
)


class TestEloTrackerEdgeCases:
    """Test ELO tracker edge cases and branches."""

    def test_elo_ranking_with_ties(self):
        """Test rankings when fighters have same ELO."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")
        tracker.add_fighter("c")

        # All have same ELO
        tracker.fighters["a"].elo = 1500
        tracker.fighters["b"].elo = 1500
        tracker.fighters["c"].elo = 1500

        rankings = tracker.get_rankings()

        # Should return all 3
        assert len(rankings) == 3

    def test_expected_score_with_large_rating_difference(self):
        """Test expected score with very different ratings."""
        tracker = EloTracker()

        # Huge difference
        expected_high = tracker.expected_score(2000, 1000)
        assert expected_high > 0.95  # Almost certain win

        expected_low = tracker.expected_score(1000, 2000)
        assert expected_low < 0.05  # Almost certain loss

    def test_update_ratings_b_wins(self):
        """Test updating ratings when B wins."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        initial_a = tracker.fighters["a"].elo
        initial_b = tracker.fighters["b"].elo

        # B wins
        tracker.update_ratings("a", "b", "b_wins", damage_a=20, damage_b=60)

        # A should lose rating
        assert tracker.fighters["a"].elo < initial_a
        # B should gain rating
        assert tracker.fighters["b"].elo > initial_b
        # B should have a win
        assert tracker.fighters["b"].wins == 1
        assert tracker.fighters["a"].losses == 1

    def test_fighter_stats_with_many_matches(self):
        """Test stats accumulation over many matches."""
        tracker = EloTracker()

        tracker.add_fighter("veteran")
        tracker.add_fighter("rookie")

        # Simulate 10 matches
        for i in range(10):
            result = "a_wins" if i % 3 != 0 else "b_wins"
            tracker.update_ratings("veteran", "rookie", result,
                                 damage_a=30+i, damage_b=20+i)

        veteran = tracker.fighters["veteran"]

        # Should have played 10 matches
        assert veteran.matches_played == 10
        # Should have wins + losses = 10
        assert veteran.wins + veteran.losses + veteran.draws == 10


class TestFighterLoaderEdgeCases:
    """Test fighter loader edge cases."""

    def test_load_fighter_with_invalid_python(self):
        """Test loading file with syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write invalid Python
            f.write("def decide(state\n")  # Missing closing paren
            f.write("    return {}\n")
            temp_path = f.name

        try:
            with pytest.raises(FighterLoadError):
                load_fighter(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validate_fighter_with_invalid_action_format(self):
        """Test validation catches invalid action format."""
        def bad_fighter(state):
            return {"wrong_key": 0.5}  # Missing required keys

        is_valid = validate_fighter(bad_fighter, verbose=False)

        assert not is_valid

    def test_validate_fighter_with_invalid_acceleration_type(self):
        """Test validation catches non-numeric acceleration."""
        def bad_fighter(state):
            return {"acceleration": "not a number", "stance": "neutral"}

        is_valid = validate_fighter(bad_fighter, verbose=False)

        assert not is_valid

    def test_validate_fighter_with_invalid_stance(self):
        """Test validation catches invalid stance values."""
        def bad_fighter(state):
            return {"acceleration": 0.5, "stance": "invalid_stance"}

        is_valid = validate_fighter(bad_fighter, verbose=False)

        assert not is_valid

    def test_load_hardcoded_fighters_with_nonexistent_path(self):
        """Test loading from non-existent directory."""
        fighters = load_hardcoded_fighters(
            base_path="/tmp/nonexistent_fighters_xyz",
            verbose=False
        )

        # Should return empty dict or handle gracefully
        assert isinstance(fighters, dict)



class TestCurriculumComponentsComplete:
    """Complete curriculum component testing."""

    def test_difficulty_level_enum_iteration(self):
        """Test iterating over difficulty levels."""
        levels = list(DifficultyLevel)

        assert len(levels) == 6
        assert DifficultyLevel.FUNDAMENTALS in levels
        assert DifficultyLevel.POPULATION in levels

    def test_curriculum_level_stores_opponent_paths(self):
        """Test curriculum level stores list of opponent file paths."""
        level = CurriculumLevel(
            name="Test",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["opponent1.py", "opponent2.py", "opponent3.py"]
        )

        # Should store list of opponents
        assert isinstance(level.opponents, list)
        assert len(level.opponents) == 3
        assert all(isinstance(opp, str) for opp in level.opponents)

    def test_training_progress_empty_recent_episodes(self):
        """Test training progress with no recent episodes."""
        progress = TrainingProgress(
            current_level=0,
            episodes_at_level=0,
            recent_episodes=[]
        )

        assert len(progress.recent_episodes) == 0
        assert progress.episodes_at_level == 0

    def test_training_progress_full_recent_window(self):
        """Test training progress with full recent window."""
        # Typical recent window size is 20
        recent = [True] * 15 + [False] * 5

        progress = TrainingProgress(
            current_level=1,
            recent_episodes=recent,
            episodes_at_level=20,
            wins_at_level=15
        )

        assert len(progress.recent_episodes) == 20
        # Win rate: 15/20 = 0.75
        wins = sum(progress.recent_episodes)
        assert wins == 15

    def test_curriculum_callback_tracks_progress(self):
        """Test curriculum callback is created for tracking."""
        # Mock trainer
        class MockTrainer:
            pass

        callback = CurriculumCallback(
            curriculum_trainer=MockTrainer(),
            verbose=1
        )

        assert callback.curriculum_trainer is not None
        assert hasattr(callback, 'episode_rewards')
        assert hasattr(callback, 'episode_wins')


class TestPopulationTrainerDataStructures:
    """Test population trainer data structures."""

    def test_population_fighter_tracks_training_episodes(self):
        """Test PopulationFighter tracks episode count."""
        from src.training.trainers.population.population_trainer import PopulationFighter

        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Tracker",
            model=MockModel(),
            training_episodes=150
        )

        assert fighter.training_episodes == 150

    def test_population_fighter_tracks_last_checkpoint(self):
        """Test PopulationFighter tracks checkpoint path."""
        from src.training.trainers.population.population_trainer import PopulationFighter

        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Checkpointed",
            model=MockModel(),
            last_checkpoint="/path/to/checkpoint.zip"
        )

        assert fighter.last_checkpoint == "/path/to/checkpoint.zip"

    def test_population_fighter_lineage_tracking(self):
        """Test PopulationFighter tracks lineage."""
        from src.training.trainers.population.population_trainer import PopulationFighter

        class MockModel:
            pass

        # Founder
        founder = PopulationFighter(
            name="Founder",
            model=MockModel(),
            lineage="founder"
        )

        assert founder.lineage == "founder"

        # Child
        child = PopulationFighter(
            name="Child",
            model=MockModel(),
            generation=2,
            lineage="founder->child"
        )

        assert child.lineage == "founder->child"
        assert child.generation == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
