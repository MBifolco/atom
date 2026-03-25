"""Tests for per-generation ELO reset in the population ELO tracker."""

import pytest

from src.atom.training.trainers.population.elo_tracker import EloTracker, FighterStats


class TestEloReset:
    """Tests for EloTracker.reset_ratings()."""

    def _make_tracker_with_history(self) -> EloTracker:
        """Create a tracker with 3 fighters that have accumulated stats."""
        tracker = EloTracker(k_factor=32.0, initial_elo=1500.0)
        tracker.add_fighter("Alpha")
        tracker.add_fighter("Beta")
        tracker.add_fighter("Charlie")

        # Simulate some matches so ELOs diverge
        tracker.update_ratings("Alpha", "Beta", "a_wins", 10.0, 5.0)
        tracker.update_ratings("Alpha", "Charlie", "a_wins", 15.0, 3.0)
        tracker.update_ratings("Beta", "Charlie", "b_wins", 8.0, 12.0)

        return tracker

    def test_reset_sets_all_to_initial_elo(self):
        tracker = self._make_tracker_with_history()

        # ELOs should have diverged from 1500
        elos_before = {f.name: f.elo for f in tracker.get_rankings()}
        assert any(e != 1500.0 for e in elos_before.values())

        tracker.reset_ratings()

        for stats in tracker.fighters.values():
            assert stats.elo == 1500.0

    def test_reset_preserves_wins_losses_draws(self):
        tracker = self._make_tracker_with_history()

        # Capture stats before reset
        stats_before = {
            name: (s.wins, s.losses, s.draws)
            for name, s in tracker.fighters.items()
        }

        tracker.reset_ratings()

        for name, stats in tracker.fighters.items():
            assert (stats.wins, stats.losses, stats.draws) == stats_before[name]

    def test_reset_preserves_damage_stats(self):
        tracker = self._make_tracker_with_history()

        damage_before = {
            name: (s.total_damage_dealt, s.total_damage_taken)
            for name, s in tracker.fighters.items()
        }

        tracker.reset_ratings()

        for name, stats in tracker.fighters.items():
            assert (stats.total_damage_dealt, stats.total_damage_taken) == damage_before[name]

    def test_reset_preserves_match_history(self):
        tracker = self._make_tracker_with_history()

        history_lengths = {
            name: len(s.match_history)
            for name, s in tracker.fighters.items()
        }

        tracker.reset_ratings()

        for name, stats in tracker.fighters.items():
            assert len(stats.match_history) == history_lengths[name]

    def test_diversity_metrics_show_zero_spread_after_reset(self):
        tracker = self._make_tracker_with_history()

        # Before reset: nonzero spread
        metrics_before = tracker.get_diversity_metrics()
        assert metrics_before["elo_range"] > 0

        tracker.reset_ratings()

        metrics_after = tracker.get_diversity_metrics()
        assert metrics_after["elo_range"] == 0.0
        assert metrics_after["elo_std"] == 0.0

    def test_rankings_reflect_only_post_reset_results(self):
        """After reset + new evaluation, rankings should reflect new results only."""
        tracker = self._make_tracker_with_history()

        # Alpha was dominant before reset
        rankings_before = [s.name for s in tracker.get_rankings()]
        assert rankings_before[0] == "Alpha"

        tracker.reset_ratings()

        # Now Charlie wins everything
        tracker.update_ratings("Charlie", "Alpha", "a_wins", 20.0, 2.0)
        tracker.update_ratings("Charlie", "Beta", "a_wins", 18.0, 3.0)
        tracker.update_ratings("Beta", "Alpha", "a_wins", 12.0, 8.0)

        rankings_after = [s.name for s in tracker.get_rankings()]
        assert rankings_after[0] == "Charlie"

    def test_new_and_incumbent_fighters_start_equal_after_reset(self):
        """A child added after reset should have the same ELO as incumbents."""
        tracker = self._make_tracker_with_history()

        tracker.reset_ratings()

        # Add a new child
        tracker.add_fighter("NewChild")

        elos = {s.name: s.elo for s in tracker.get_rankings()}
        assert elos["NewChild"] == elos["Alpha"] == elos["Beta"] == 1500.0

    def test_reset_on_empty_tracker_is_noop(self):
        tracker = EloTracker()
        tracker.reset_ratings()  # should not raise
        assert len(tracker.fighters) == 0

    def test_reset_with_custom_initial_elo(self):
        tracker = EloTracker(initial_elo=1200.0)
        tracker.add_fighter("X")
        tracker.add_fighter("Y")
        tracker.update_ratings("X", "Y", "a_wins", 5.0, 3.0)

        tracker.reset_ratings()

        assert tracker.fighters["X"].elo == 1200.0
        assert tracker.fighters["Y"].elo == 1200.0
