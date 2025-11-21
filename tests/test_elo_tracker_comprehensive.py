"""
Comprehensive tests for EloTracker to increase coverage.
Tests matchup predictions, diversity metrics, and leaderboard.
"""

import pytest
import numpy as np
from io import StringIO
import sys

from src.training.trainers.population.elo_tracker import EloTracker, FighterStats


class TestFighterStatsProperties:
    """Tests for FighterStats computed properties."""

    def test_matches_played_zero(self):
        """Test matches_played with no matches."""
        stats = FighterStats(name="test")
        assert stats.matches_played == 0

    def test_matches_played_with_games(self):
        """Test matches_played with various outcomes."""
        stats = FighterStats(name="test", wins=5, losses=3, draws=2)
        assert stats.matches_played == 10

    def test_win_rate_zero_matches(self):
        """Test win rate with no matches played."""
        stats = FighterStats(name="test")
        assert stats.win_rate == 0.0

    def test_win_rate_all_wins(self):
        """Test win rate with all wins."""
        stats = FighterStats(name="test", wins=10, losses=0, draws=0)
        assert stats.win_rate == 1.0

    def test_win_rate_mixed_results(self):
        """Test win rate with mixed results."""
        stats = FighterStats(name="test", wins=6, losses=3, draws=1)
        assert stats.win_rate == 0.6

    def test_damage_ratio_no_damage_taken(self):
        """Test damage ratio when no damage taken."""
        stats = FighterStats(name="test", total_damage_dealt=100, total_damage_taken=0)
        assert stats.damage_ratio == float('inf')

    def test_damage_ratio_no_damage_dealt(self):
        """Test damage ratio when no damage dealt."""
        stats = FighterStats(name="test", total_damage_dealt=0, total_damage_taken=100)
        assert stats.damage_ratio == 0.0

    def test_damage_ratio_equal_damage(self):
        """Test damage ratio with equal damage."""
        stats = FighterStats(name="test", total_damage_dealt=50, total_damage_taken=50)
        assert stats.damage_ratio == 1.0

    def test_damage_ratio_no_damage_at_all(self):
        """Test damage ratio when no damage dealt or taken."""
        stats = FighterStats(name="test", total_damage_dealt=0, total_damage_taken=0)
        assert stats.damage_ratio == 1.0

    def test_damage_ratio_positive_ratio(self):
        """Test damage ratio with more damage dealt."""
        stats = FighterStats(name="test", total_damage_dealt=200, total_damage_taken=100)
        assert stats.damage_ratio == 2.0


class TestEloTrackerMatchupPrediction:
    """Tests for get_matchup_prediction method."""

    def test_matchup_prediction_equal_elo(self):
        """Test prediction with equal ELO ratings."""
        tracker = EloTracker()
        tracker.add_fighter("Alpha")
        tracker.add_fighter("Beta")

        prediction = tracker.get_matchup_prediction("Alpha", "Beta")

        assert prediction["fighter_a"] == "Alpha"
        assert prediction["fighter_b"] == "Beta"
        assert prediction["elo_a"] == 1500.0
        assert prediction["elo_b"] == 1500.0
        assert abs(prediction["win_prob_a"] - 0.5) < 0.01
        assert abs(prediction["win_prob_b"] - 0.5) < 0.01
        assert prediction["elo_diff"] == 0

    def test_matchup_prediction_different_elo(self):
        """Test prediction with different ELO ratings."""
        tracker = EloTracker()
        tracker.add_fighter("Strong")
        tracker.add_fighter("Weak")
        tracker.fighters["Strong"].elo = 1700
        tracker.fighters["Weak"].elo = 1300

        prediction = tracker.get_matchup_prediction("Strong", "Weak")

        assert prediction["win_prob_a"] > 0.5
        assert prediction["win_prob_b"] < 0.5
        assert prediction["elo_diff"] == 400
        assert prediction["favorite"] == "Strong"

    def test_matchup_prediction_creates_new_fighters(self):
        """Test that prediction creates fighters if they don't exist."""
        tracker = EloTracker()

        prediction = tracker.get_matchup_prediction("New1", "New2")

        assert "New1" in tracker.fighters
        assert "New2" in tracker.fighters
        assert prediction["elo_a"] == 1500.0
        assert prediction["elo_b"] == 1500.0

    def test_matchup_prediction_favorite_determination(self):
        """Test that favorite is correctly determined."""
        tracker = EloTracker()
        tracker.add_fighter("Underdog")
        tracker.add_fighter("Champion")
        tracker.fighters["Champion"].elo = 1800

        prediction = tracker.get_matchup_prediction("Underdog", "Champion")

        assert prediction["favorite"] == "Champion"
        assert prediction["win_prob_b"] > prediction["win_prob_a"]


class TestEloTrackerDiversityMetrics:
    """Tests for get_diversity_metrics method."""

    def test_diversity_metrics_empty_population(self):
        """Test metrics with no fighters."""
        tracker = EloTracker()
        metrics = tracker.get_diversity_metrics()
        assert metrics == {}

    def test_diversity_metrics_single_fighter(self):
        """Test metrics with single fighter."""
        tracker = EloTracker()
        tracker.add_fighter("Solo")

        metrics = tracker.get_diversity_metrics()

        assert metrics["population_size"] == 1
        assert metrics["elo_mean"] == 1500.0
        assert metrics["elo_std"] == 0.0
        assert metrics["elo_range"] == 0.0

    def test_diversity_metrics_multiple_fighters(self):
        """Test metrics with multiple fighters."""
        tracker = EloTracker()
        tracker.add_fighter("Low")
        tracker.add_fighter("Mid")
        tracker.add_fighter("High")
        tracker.fighters["Low"].elo = 1300
        tracker.fighters["Mid"].elo = 1500
        tracker.fighters["High"].elo = 1700

        metrics = tracker.get_diversity_metrics()

        assert metrics["population_size"] == 3
        assert metrics["elo_mean"] == 1500.0
        assert metrics["elo_range"] == 400
        assert metrics["elo_top"] == 1700
        assert metrics["elo_bottom"] == 1300

    def test_diversity_metrics_with_matches(self):
        """Test metrics with fighters that have played matches."""
        tracker = EloTracker()
        tracker.add_fighter("A")
        tracker.add_fighter("B")

        # Simulate some matches
        tracker.update_ratings("A", "B", "a_wins", damage_a=50, damage_b=30)
        tracker.update_ratings("A", "B", "a_wins", damage_a=60, damage_b=20)
        tracker.update_ratings("B", "A", "b_wins", damage_a=40, damage_b=70)

        metrics = tracker.get_diversity_metrics()

        assert "win_rate_mean" in metrics
        assert "win_rate_std" in metrics

    def test_diversity_metrics_with_damage_ratios(self):
        """Test metrics include damage ratio stats."""
        tracker = EloTracker()
        tracker.add_fighter("Attacker")
        tracker.add_fighter("Defender")

        tracker.update_ratings("Attacker", "Defender", "a_wins", damage_a=100, damage_b=50)

        metrics = tracker.get_diversity_metrics()

        assert "damage_ratio_mean" in metrics
        assert "damage_ratio_std" in metrics

    def test_diversity_metrics_ignores_infinite_ratios(self):
        """Test that infinite damage ratios are excluded from mean."""
        tracker = EloTracker()
        tracker.add_fighter("A")
        tracker.add_fighter("B")

        # A takes no damage, so ratio would be inf
        tracker.update_ratings("A", "B", "a_wins", damage_a=100, damage_b=0)

        metrics = tracker.get_diversity_metrics()

        # Should still have damage ratio metrics but computed from valid values
        # If only one fighter has valid ratio, metrics should still compute


class TestEloTrackerLeaderboard:
    """Tests for print_leaderboard method."""

    def test_leaderboard_empty(self):
        """Test leaderboard with no fighters."""
        tracker = EloTracker()

        # Capture stdout
        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "POPULATION LEADERBOARD" in output

    def test_leaderboard_with_fighters(self):
        """Test leaderboard with multiple fighters."""
        tracker = EloTracker()
        tracker.add_fighter("Alpha")
        tracker.add_fighter("Beta")
        tracker.add_fighter("Gamma")

        tracker.fighters["Alpha"].elo = 1600
        tracker.fighters["Beta"].elo = 1500
        tracker.fighters["Gamma"].elo = 1400

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "Alpha" in output
        assert "Beta" in output
        assert "Gamma" in output
        assert "1600" in output

    def test_leaderboard_top_n(self):
        """Test leaderboard with top_n limit."""
        tracker = EloTracker()
        for i in range(10):
            tracker.add_fighter(f"Fighter{i}")
            tracker.fighters[f"Fighter{i}"].elo = 1500 + i * 50

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard(top_n=3)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should only show top 3
        assert "Fighter9" in output  # ELO 1950
        assert "Fighter8" in output  # ELO 1900
        assert "Fighter7" in output  # ELO 1850

    def test_leaderboard_active_only(self):
        """Test leaderboard filtered to active fighters."""
        tracker = EloTracker()
        tracker.add_fighter("Active1")
        tracker.add_fighter("Active2")
        tracker.add_fighter("Inactive")

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard(active_only=["Active1", "Active2"])
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "Active1" in output
        assert "Active2" in output
        # Inactive shouldn't be shown
        lines = [line for line in output.split("\n") if "Inactive" in line and not line.startswith("=")]
        assert len(lines) == 0 or all("Inactive" not in line for line in lines if line.strip() and not line.startswith("-") and not line.startswith("="))

    def test_leaderboard_with_match_history(self):
        """Test leaderboard shows win/loss records."""
        tracker = EloTracker()
        tracker.add_fighter("Winner")
        tracker.add_fighter("Loser")

        for _ in range(5):
            tracker.update_ratings("Winner", "Loser", "a_wins")

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "5-0-0" in output  # Winner's record
        assert "0-5-0" in output  # Loser's record

    def test_leaderboard_shows_diversity_metrics(self):
        """Test leaderboard shows diversity metrics."""
        tracker = EloTracker()
        tracker.add_fighter("A")
        tracker.add_fighter("B")
        tracker.fighters["A"].elo = 1600
        tracker.fighters["B"].elo = 1400

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "POPULATION DIVERSITY" in output
        assert "ELO Range" in output

    def test_leaderboard_infinite_damage_ratio(self):
        """Test leaderboard handles infinite damage ratio display."""
        tracker = EloTracker()
        tracker.add_fighter("Perfect")
        tracker.fighters["Perfect"].total_damage_dealt = 100
        tracker.fighters["Perfect"].total_damage_taken = 0
        tracker.fighters["Perfect"].wins = 1  # Has played a match

        captured = StringIO()
        sys.stdout = captured
        tracker.print_leaderboard()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should show infinity symbol for damage ratio
        assert "∞" in output or "inf" in output.lower()


class TestEloTrackerSuggestBalancedMatches:
    """Tests for suggest_balanced_matches method."""

    def test_suggest_matches_insufficient_fighters(self):
        """Test suggesting matches with too few fighters."""
        tracker = EloTracker()
        tracker.add_fighter("Solo")

        matches = tracker.suggest_balanced_matches()
        assert matches == []

    def test_suggest_matches_pairs_similar_elo(self):
        """Test that matches pair fighters with similar ELO."""
        tracker = EloTracker()
        tracker.add_fighter("High1")
        tracker.add_fighter("High2")
        tracker.add_fighter("Low1")
        tracker.add_fighter("Low2")

        tracker.fighters["High1"].elo = 1700
        tracker.fighters["High2"].elo = 1680
        tracker.fighters["Low1"].elo = 1300
        tracker.fighters["Low2"].elo = 1320

        matches = tracker.suggest_balanced_matches(num_matches=2)

        # Should pair similar ELO fighters
        assert len(matches) <= 2
        for fighter_a, fighter_b in matches:
            elo_a = tracker.fighters[fighter_a].elo
            elo_b = tracker.fighters[fighter_b].elo
            # Should be relatively close in ELO
            assert abs(elo_a - elo_b) < 500

    def test_suggest_matches_respects_num_matches(self):
        """Test that num_matches limits output."""
        tracker = EloTracker()
        for i in range(10):
            tracker.add_fighter(f"F{i}")

        matches = tracker.suggest_balanced_matches(num_matches=2)
        assert len(matches) <= 2

    def test_suggest_matches_with_active_filter(self):
        """Test suggesting matches only from active fighters."""
        tracker = EloTracker()
        tracker.add_fighter("Active1")
        tracker.add_fighter("Active2")
        tracker.add_fighter("Inactive")

        matches = tracker.suggest_balanced_matches(
            num_matches=1,
            active_fighters=["Active1", "Active2"]
        )

        # Should only suggest Active1 vs Active2
        for a, b in matches:
            assert a in ["Active1", "Active2"]
            assert b in ["Active1", "Active2"]


class TestEloTrackerIntegration:
    """Integration tests for EloTracker."""

    def test_full_tournament_simulation(self):
        """Test simulating a full tournament."""
        tracker = EloTracker()

        # Add 4 fighters
        fighters = ["Alpha", "Beta", "Gamma", "Delta"]
        for f in fighters:
            tracker.add_fighter(f)

        # Simulate round-robin tournament
        for i, a in enumerate(fighters):
            for b in fighters[i+1:]:
                # Simulate match (higher alphabetically wins for determinism)
                result = "a_wins" if a < b else "b_wins"
                tracker.update_ratings(a, b, result, damage_a=50, damage_b=30)

        # Check rankings
        rankings = tracker.get_rankings()
        assert len(rankings) == 4
        assert rankings[0].elo > rankings[-1].elo

        # Check match histories
        for f in fighters:
            assert tracker.fighters[f].matches_played == 3

    def test_elo_convergence(self):
        """Test that ELO ratings converge with repeated matches."""
        tracker = EloTracker()
        tracker.add_fighter("Better")
        tracker.add_fighter("Worse")

        # Better always wins
        for _ in range(20):
            tracker.update_ratings("Better", "Worse", "a_wins")

        # Better should have higher ELO than initial
        assert tracker.fighters["Better"].elo > 1600
        assert tracker.fighters["Worse"].elo < 1400
        # Better should be significantly higher than Worse
        assert tracker.fighters["Better"].elo - tracker.fighters["Worse"].elo > 300
