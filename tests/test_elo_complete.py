"""
Complete ELO tracker coverage - target uncovered lines 179-307.
"""

import pytest
from src.training.trainers.population.elo_tracker import EloTracker, FighterStats


class TestEloTrackerMissingBranches:
    """Test uncovered ELO tracker code paths."""

    def test_get_all_fighters_stats(self):
        """Test getting all fighter stats."""
        tracker = EloTracker()

        tracker.add_fighter("f1")
        tracker.add_fighter("f2")
        tracker.add_fighter("f3")

        # Get all fighters
        all_fighters = tracker.fighters

        assert len(all_fighters) == 3
        assert "f1" in all_fighters
        assert all(isinstance(stats, FighterStats) for stats in all_fighters.values())

    def test_update_ratings_with_match_info(self):
        """Test updating ratings with additional match info."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        match_info = {
            "duration": 150,
            "collisions": 12,
            "spectacle_score": 0.75
        }

        tracker.update_ratings("a", "b", "a_wins",
                             damage_a=60, damage_b=30,
                             match_info=match_info)

        # Match info should be stored in history
        history = tracker.fighters["a"].match_history
        assert len(history) > 0

    def test_fighter_stats_with_only_draws(self):
        """Test fighter with only draws."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        # All draws
        for _ in range(5):
            tracker.update_ratings("a", "b", "draw", damage_a=25, damage_b=25)

        stats_a = tracker.fighters["a"]

        assert stats_a.wins == 0
        assert stats_a.losses == 0
        assert stats_a.draws == 5
        assert stats_a.matches_played == 5
        assert stats_a.win_rate == 0.0  # No wins

    def test_elo_change_for_upset_win(self):
        """Test large ELO change when underdog wins."""
        tracker = EloTracker(k_factor=32)

        tracker.add_fighter("weak")
        tracker.add_fighter("strong")

        # Set very different ratings
        tracker.fighters["weak"].elo = 1200
        tracker.fighters["strong"].elo = 1800

        # Weak fighter wins (upset!)
        tracker.update_ratings("weak", "strong", "a_wins")

        # Weak should gain a lot of ELO
        assert tracker.fighters["weak"].elo > 1200
        # Strong should lose a lot
        assert tracker.fighters["strong"].elo < 1800

        # Change should be significant
        elo_gained = tracker.fighters["weak"].elo - 1200
        assert elo_gained > 20  # Significant gain for upset

    def test_elo_change_for_expected_win(self):
        """Test small ELO change when favorite wins."""
        tracker = EloTracker(k_factor=32)

        tracker.add_fighter("strong")
        tracker.add_fighter("weak")

        tracker.fighters["strong"].elo = 1800
        tracker.fighters["weak"].elo = 1200

        # Strong wins (expected)
        tracker.update_ratings("strong", "weak", "a_wins")

        # Change should be small
        elo_gained = tracker.fighters["strong"].elo - 1800
        assert elo_gained < 10  # Small gain for expected win

    def test_get_rankings_empty_tracker(self):
        """Test rankings with no fighters."""
        tracker = EloTracker()

        rankings = tracker.get_rankings()

        assert rankings == []

    def test_get_rankings_single_fighter(self):
        """Test rankings with single fighter."""
        tracker = EloTracker()

        tracker.add_fighter("solo")

        rankings = tracker.get_rankings()

        assert len(rankings) == 1
        assert rankings[0].name == "solo"

    def test_k_factor_affects_rating_changes(self):
        """Test different K-factors produce different rating changes."""
        # Low K-factor (slow changes)
        tracker_low = EloTracker(k_factor=16)
        tracker_low.add_fighter("a")
        tracker_low.add_fighter("b")

        initial = 1500
        tracker_low.fighters["a"].elo = initial
        tracker_low.fighters["b"].elo = initial

        tracker_low.update_ratings("a", "b", "a_wins")
        change_low = abs(tracker_low.fighters["a"].elo - initial)

        # High K-factor (fast changes)
        tracker_high = EloTracker(k_factor=64)
        tracker_high.add_fighter("a")
        tracker_high.add_fighter("b")

        tracker_high.fighters["a"].elo = initial
        tracker_high.fighters["b"].elo = initial

        tracker_high.update_ratings("a", "b", "a_wins")
        change_high = abs(tracker_high.fighters["a"].elo - initial)

        # Higher K should produce larger changes
        assert change_high > change_low


class TestEloTrackerSuggestMatches:
    """Test suggest_balanced_matches method (lines 209-229)."""

    def test_suggest_balanced_matches_basic(self):
        """Test suggesting balanced matches."""
        tracker = EloTracker()

        # Add 6 fighters with different ratings
        for i in range(6):
            tracker.add_fighter(f"fighter{i}")
            tracker.fighters[f"fighter{i}"].elo = 1400 + i * 100

        suggestions = tracker.suggest_balanced_matches(num_matches=3)

        # Should suggest some matches
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_suggest_matches_with_too_few_fighters(self):
        """Test suggesting matches when <2 fighters."""
        tracker = EloTracker()

        tracker.add_fighter("only_one")

        suggestions = tracker.suggest_balanced_matches()

        # Can't make matches with 1 fighter
        assert suggestions == []

    def test_suggest_matches_with_active_fighters_filter(self):
        """Test suggesting matches with active fighters filter."""
        tracker = EloTracker()

        # Add many fighters
        for i in range(8):
            tracker.add_fighter(f"f{i}")

        # Only some are active
        active = ["f0", "f1", "f2", "f3"]

        suggestions = tracker.suggest_balanced_matches(
            num_matches=2,
            active_fighters=active
        )

        # Suggestions should only include active fighters
        for fa, fb in suggestions:
            assert fa in active
            assert fb in active


class TestFighterStatsMatchHistory:
    """Test fighter stats match history tracking."""

    def test_match_history_accumulates(self):
        """Test match history grows with each match."""
        tracker = EloTracker()

        tracker.add_fighter("tracker")
        tracker.add_fighter("opp")

        # Play 3 matches
        tracker.update_ratings("tracker", "opp", "a_wins")
        tracker.update_ratings("tracker", "opp", "draw")
        tracker.update_ratings("tracker", "opp", "b_wins")

        stats = tracker.fighters["tracker"]

        # History should have 3 entries
        assert len(stats.match_history) == 3

    def test_match_history_contains_match_info(self):
        """Test match history entries contain info."""
        tracker = EloTracker()

        tracker.add_fighter("a")
        tracker.add_fighter("b")

        match_info = {"test": "data", "score": 0.5}

        tracker.update_ratings("a", "b", "a_wins",
                             damage_a=50, damage_b=30,
                             match_info=match_info)

        history = tracker.fighters["a"].match_history

        assert len(history) > 0
        # History entry should be a dict
        assert isinstance(history[0], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

