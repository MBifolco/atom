"""
Comprehensive tests for PopulationCallback.
Tests callback lifecycle and episode tracking.
"""

import pytest
from src.training.trainers.population.population_trainer import PopulationCallback
from src.atom.training.trainers.population.elo_tracker import EloTracker


class TestPopulationCallbackInitialization:
    """Test PopulationCallback initialization."""

    def test_callback_stores_fighter_name(self):
        """Test callback stores the fighter name being trained."""
        tracker = EloTracker()
        callback = PopulationCallback(
            fighter_name="TestFighter",
            elo_tracker=tracker,
            verbose=0
        )

        assert callback.fighter_name == "TestFighter"

    def test_callback_stores_elo_tracker_reference(self):
        """Test callback stores reference to ELO tracker."""
        tracker = EloTracker()
        callback = PopulationCallback(
            fighter_name="Test",
            elo_tracker=tracker,
            verbose=1
        )

        assert callback.elo_tracker is tracker

    def test_callback_initializes_episode_counter(self):
        """Test callback initializes episode counter to zero."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        assert callback.episode_count == 0

    def test_callback_initializes_rewards_list(self):
        """Test callback initializes recent rewards tracking."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker, verbose=0)

        assert hasattr(callback, 'recent_rewards')
        assert isinstance(callback.recent_rewards, list)
        assert len(callback.recent_rewards) == 0

    def test_callback_with_different_verbosity_levels(self):
        """Test callback accepts different verbosity levels."""
        tracker = EloTracker()

        callback_quiet = PopulationCallback("Fighter1", tracker, verbose=0)
        callback_normal = PopulationCallback("Fighter2", tracker, verbose=1)
        callback_debug = PopulationCallback("Fighter3", tracker, verbose=2)

        assert callback_quiet.verbose == 0
        assert callback_normal.verbose == 1
        assert callback_debug.verbose == 2


class TestPopulationCallbackStepMethod:
    """Test PopulationCallback _on_step method."""

    def test_on_step_method_exists(self):
        """Test callback has _on_step method."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        assert hasattr(callback, '_on_step')
        assert callable(callback._on_step)

    def test_on_step_returns_boolean(self):
        """Test _on_step returns boolean to continue training."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        # Mock the locals that would be provided during training
        callback.locals = {"infos": []}

        result = callback._on_step()

        assert isinstance(result, bool)
        assert result is True  # Should return True to continue


class TestPopulationCallbackEpisodeTracking:
    """Test PopulationCallback episode tracking functionality."""

    def test_callback_processes_episode_info(self):
        """Test callback processes episode completion info."""
        tracker = EloTracker()
        callback = PopulationCallback("TestFighter", tracker, verbose=0)

        # Simulate episode completion
        callback.locals = {
            "infos": [
                {"episode": {"r": 150.5, "l": 120}}
            ]
        }

        initial_count = callback.episode_count

        callback._on_step()

        # Episode count should increment
        assert callback.episode_count == initial_count + 1

        # Reward should be tracked
        assert len(callback.recent_rewards) == 1
        assert callback.recent_rewards[0] == 150.5

    def test_callback_handles_multiple_episodes_in_step(self):
        """Test callback handles multiple environment episodes in one step."""
        tracker = EloTracker()
        callback = PopulationCallback("Multi", tracker, verbose=0)

        # Multiple environments finish episodes
        callback.locals = {
            "infos": [
                {"episode": {"r": 100.0, "l": 80}},
                {"episode": {"r": 120.0, "l": 90}},
                {"episode": {"r": 110.0, "l": 85}}
            ]
        }

        callback._on_step()

        # Should have processed all 3 episodes
        assert callback.episode_count == 3
        assert len(callback.recent_rewards) == 3

    def test_callback_handles_no_episode_completion(self):
        """Test callback handles step with no episode completions."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        # No episodes completed
        callback.locals = {"infos": []}

        initial_count = callback.episode_count

        callback._on_step()

        # Count should not change
        assert callback.episode_count == initial_count

    def test_callback_limits_recent_rewards_list(self):
        """Test callback limits recent rewards to last 100."""
        tracker = EloTracker()
        callback = PopulationCallback("Test", tracker)

        # Simulate 150 episodes
        for i in range(150):
            callback.locals = {
                "infos": [{"episode": {"r": float(i), "l": 100}}]
            }
            callback._on_step()

        # Should only keep last 100
        assert len(callback.recent_rewards) == 100
        # Most recent should be from latest episodes
        assert callback.recent_rewards[-1] == 149.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
