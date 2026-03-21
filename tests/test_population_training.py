"""
Tests for population training system components.
"""

import pytest
from src.atom.training.trainers.population.elo_tracker import EloTracker, FighterStats


class TestFighterStats:
    """Test FighterStats dataclass."""

    def test_fighter_stats_creation(self):
        """Test creating fighter stats."""
        stats = FighterStats(name="TestFighter", elo=1500.0)

        assert stats.name == "TestFighter"
        assert stats.elo == 1500.0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0

    def test_matches_played_property(self):
        """Test matches_played calculates correctly."""
        stats = FighterStats(name="Test", wins=5, losses=3, draws=2)

        assert stats.matches_played == 10

    def test_win_rate_property(self):
        """Test win rate calculation."""
        stats = FighterStats(name="Test", wins=7, losses=3)

        assert stats.win_rate == 0.7

    def test_win_rate_zero_matches(self):
        """Test win rate when no matches played."""
        stats = FighterStats(name="Test")

        assert stats.win_rate == 0.0

    def test_damage_ratio_property(self):
        """Test damage ratio calculation."""
        stats = FighterStats(
            name="Test",
            total_damage_dealt=100.0,
            total_damage_taken=50.0
        )

        assert stats.damage_ratio == 2.0


class TestEloTracker:
    """Test ELO rating system for fighters."""

    def test_elo_tracker_initialization(self):
        """Test ELO tracker initializes correctly."""
        tracker = EloTracker(k_factor=32, initial_elo=1500)

        assert tracker.k_factor == 32
        assert tracker.initial_elo == 1500
        assert len(tracker.fighters) == 0

    def test_add_fighter(self):
        """Test adding a new fighter."""
        tracker = EloTracker()

        tracker.add_fighter("fighter1")

        assert "fighter1" in tracker.fighters
        assert tracker.fighters["fighter1"].elo == 1500  # Default

    def test_remove_fighter(self):
        """Test removing a fighter."""
        tracker = EloTracker()

        tracker.add_fighter("test")
        assert "test" in tracker.fighters

        tracker.remove_fighter("test")
        assert "test" not in tracker.fighters

    def test_expected_score_calculation(self):
        """Test expected score calculation between fighters."""
        tracker = EloTracker()

        # Strong vs weak fighter
        expected = tracker.expected_score(1700, 1300)
        assert 0.5 < expected < 1.0  # Strong fighter favored

        # Weak vs strong
        expected_weak = tracker.expected_score(1300, 1700)
        assert 0.0 < expected_weak < 0.5  # Weak fighter underdog

        # Equal fighters
        expected_equal = tracker.expected_score(1500, 1500)
        assert abs(expected_equal - 0.5) < 0.01  # 50/50

    def test_update_ratings_on_match(self):
        """Test updating ratings after a match."""
        tracker = EloTracker()

        tracker.add_fighter("winner")
        tracker.add_fighter("loser")

        initial_winner_elo = tracker.fighters["winner"].elo
        initial_loser_elo = tracker.fighters["loser"].elo

        # Record match (winner wins)
        tracker.update_ratings("winner", "loser", "a_wins", damage_a=50.0, damage_b=20.0)

        # Winner should gain ELO
        assert tracker.fighters["winner"].elo > initial_winner_elo
        # Loser should lose ELO
        assert tracker.fighters["loser"].elo < initial_loser_elo

    def test_update_ratings_on_draw(self):
        """Test updating ratings on a draw."""
        tracker = EloTracker()

        tracker.add_fighter("fighter_a")
        tracker.add_fighter("fighter_b")

        initial_a = tracker.fighters["fighter_a"].elo
        initial_b = tracker.fighters["fighter_b"].elo

        # Record draw
        tracker.update_ratings("fighter_a", "fighter_b", "draw", damage_a=30.0, damage_b=30.0)

        # Ratings should stay roughly equal for equal fighters drawing
        # (might change slightly based on implementation)
        new_a = tracker.fighters["fighter_a"].elo
        new_b = tracker.fighters["fighter_b"].elo

        assert abs(new_a - new_b) < 50  # Should remain close

    def test_win_loss_draw_counts(self):
        """Test win/loss/draw counting."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        # A wins
        tracker.update_ratings("a", "b", "a_wins")
        assert tracker.fighters["a"].wins == 1
        assert tracker.fighters["b"].losses == 1

        # Draw
        tracker.update_ratings("a", "b", "draw")
        assert tracker.fighters["a"].draws == 1
        assert tracker.fighters["b"].draws == 1

        # B wins
        tracker.update_ratings("a", "b", "b_wins")
        assert tracker.fighters["b"].wins == 1
        assert tracker.fighters["a"].losses == 1

    def test_damage_tracking(self):
        """Test damage is tracked across matches."""
        tracker = EloTracker()

        tracker.add_fighter("attacker")
        tracker.add_fighter("defender")

        tracker.update_ratings("attacker", "defender", "a_wins", damage_a=80.0, damage_b=20.0)

        assert tracker.fighters["attacker"].total_damage_dealt == 80.0
        assert tracker.fighters["attacker"].total_damage_taken == 20.0
        assert tracker.fighters["defender"].total_damage_dealt == 20.0
        assert tracker.fighters["defender"].total_damage_taken == 80.0

    def test_get_rankings(self):
        """Test getting ranked list of fighters."""
        tracker = EloTracker()

        tracker.add_fighter("best")
        tracker.add_fighter("middle")
        tracker.add_fighter("worst")

        # Manually set ELOs
        tracker.fighters["best"].elo = 1800
        tracker.fighters["middle"].elo = 1500
        tracker.fighters["worst"].elo = 1200

        rankings = tracker.get_rankings()

        assert len(rankings) == 3
        assert rankings[0].name == "best"
        assert rankings[1].name == "middle"
        assert rankings[2].name == "worst"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
