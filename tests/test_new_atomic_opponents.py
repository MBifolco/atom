#!/usr/bin/env python3
"""
Test the newly created atomic opponent files to ensure they work correctly.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the opponent modules
from fighters.test_dummies.atomic import distance_keeper_3m
from fighters.test_dummies.atomic import stamina_waster
from fighters.test_dummies.atomic import forward_mover
from fighters.test_dummies.atomic import backward_mover
from fighters.test_dummies.atomic import sideways_mover_smooth
from fighters.test_dummies.atomic import strategic_retreater


class TestNewAtomicOpponents:
    """Test all newly created atomic opponents."""

    def create_state(self, distance=2.0, position=0.0, tick=0):
        """Helper to create a test state."""
        return {
            "tick": tick,
            "you": {
                "position": position,
                "velocity": 0.0,
                "stamina": 100.0,
                "max_stamina": 100.0,
                "hp": 100.0,
                "max_hp": 100.0,
                "stance": "neutral"
            },
            "opponent": {
                "distance": distance,
                "direction": 1.0 if distance > 0 else -1.0 if distance < 0 else 0.0,
                "velocity": 0.0,
                "stamina": 100.0,
                "hp": 100.0,
                "stance": "neutral"
            }
        }

    def test_distance_keeper_3m(self):
        """Test that distance_keeper_3m maintains 3m distance."""
        # Test when too close (should back away)
        state = self.create_state(distance=1.0)
        decision = distance_keeper_3m.decide(state)
        assert decision["acceleration"] < 0  # Should back away from right opponent
        assert decision["stance"] == "neutral"

        # Test when too far (should approach)
        state = self.create_state(distance=5.0)
        decision = distance_keeper_3m.decide(state)
        assert decision["acceleration"] > 0  # Should approach right opponent
        assert decision["stance"] == "neutral"

        # Test at optimal distance (should stay)
        state = self.create_state(distance=3.0)
        decision = distance_keeper_3m.decide(state)
        assert decision["acceleration"] == 0.0  # Should stay put
        assert decision["stance"] == "neutral"

        # Test with opponent on left
        state = self.create_state(distance=-3.5)
        decision = distance_keeper_3m.decide(state)
        assert decision["acceleration"] < 0  # Should approach left opponent

    def test_stamina_waster(self):
        """Test that stamina_waster always uses extended stance."""
        # Test various states - should always return extended stance
        for distance in [0.5, 2.0, 5.0, -2.0]:
            state = self.create_state(distance=distance)
            decision = stamina_waster.decide(state)
            assert decision["acceleration"] == 0.0  # Should be stationary
            assert decision["stance"] == "extended"  # Always extended

    def test_forward_mover(self):
        """Test that forward_mover always moves toward opponent."""
        # Test with opponent on right
        state = self.create_state(distance=3.0)
        decision = forward_mover.decide(state)
        assert decision["acceleration"] > 0  # Should move right
        assert decision["stance"] in ["neutral", "extended"]

        # Test with opponent on left
        state = self.create_state(distance=-3.0)
        decision = forward_mover.decide(state)
        assert decision["acceleration"] < 0  # Should move left
        assert decision["stance"] in ["neutral", "extended"]

        # Test close range (should use extended)
        state = self.create_state(distance=1.0)
        decision = forward_mover.decide(state)
        assert decision["acceleration"] > 0  # Should move toward opponent
        assert decision["stance"] == "extended"  # Aggressive at close range

    def test_backward_mover(self):
        """Test that backward_mover always moves away from opponent."""
        # Test with opponent on right
        state = self.create_state(distance=2.0)
        decision = backward_mover.decide(state)
        assert decision["acceleration"] < 0  # Should move left (away)
        assert decision["stance"] in ["neutral", "defending"]

        # Test with opponent on left
        state = self.create_state(distance=-2.0)
        decision = backward_mover.decide(state)
        assert decision["acceleration"] > 0  # Should move right (away)
        assert decision["stance"] in ["neutral", "defending"]

        # Test close range (should use defending)
        state = self.create_state(distance=1.5)
        decision = backward_mover.decide(state)
        assert decision["acceleration"] < 0  # Should move away
        assert decision["stance"] == "defending"  # Defensive at close range

    def test_sideways_mover_smooth(self):
        """Test that sideways_mover_smooth oscillates smoothly."""
        # Test at different tick points in the cycle (60 ticks per second)
        accelerations = []
        for tick in [0, 30, 60, 90, 120, 150, 180, 210]:
            state = self.create_state(distance=2.0, tick=tick)
            decision = sideways_mover_smooth.decide(state)
            accelerations.append(decision["acceleration"])
            assert decision["stance"] in ["neutral", "extended", "defending"]

        # Check that it oscillates (changes direction)
        assert max(accelerations) > 0  # Sometimes moves right
        assert min(accelerations) < 0  # Sometimes moves left

        # Test stance variation by distance
        state = self.create_state(distance=0.5)
        decision = sideways_mover_smooth.decide(state)
        assert decision["stance"] == "defending"  # Close range

        state = self.create_state(distance=1.5)
        decision = sideways_mover_smooth.decide(state)
        assert decision["stance"] == "extended"  # Medium range

        state = self.create_state(distance=3.0)
        decision = sideways_mover_smooth.decide(state)
        assert decision["stance"] == "neutral"  # Far range

    def test_strategic_retreater(self):
        """Test that strategic_retreater retreats intelligently."""
        # Test danger zone (< 1m)
        state = self.create_state(distance=0.8)
        decision = strategic_retreater.decide(state)
        assert decision["acceleration"] < 0  # Quick retreat from right opponent
        assert abs(decision["acceleration"]) >= 3.0  # Quick retreat
        assert decision["stance"] == "defending"

        # Test medium range (1-3m)
        state = self.create_state(distance=2.0)
        decision = strategic_retreater.decide(state)
        assert decision["acceleration"] < 0  # Moderate retreat
        assert abs(decision["acceleration"]) == 1.5  # Controlled retreat
        assert decision["stance"] == "neutral"

        # Test safe zone (> 3m)
        state = self.create_state(distance=4.0)
        decision = strategic_retreater.decide(state)
        assert decision["acceleration"] == 0.0  # Can stop retreating
        assert decision["stance"] == "extended"  # Can be aggressive

        # Test with opponent on left
        state = self.create_state(distance=-0.8)
        decision = strategic_retreater.decide(state)
        assert decision["acceleration"] > 0  # Quick retreat right
        assert abs(decision["acceleration"]) >= 3.0  # Quick retreat
        assert decision["stance"] == "defending"

    def test_all_return_valid_actions(self):
        """Test that all opponents return valid action dictionaries."""
        opponents = [
            distance_keeper_3m,
            stamina_waster,
            forward_mover,
            backward_mover,
            sideways_mover_smooth,
            strategic_retreater
        ]

        for opponent in opponents:
            state = self.create_state(distance=2.0)
            decision = opponent.decide(state)

            # Check required fields
            assert "acceleration" in decision, f"{opponent.__name__} missing acceleration"
            assert "stance" in decision, f"{opponent.__name__} missing stance"

            # Check valid ranges
            assert -5.0 <= decision["acceleration"] <= 5.0, \
                f"{opponent.__name__} acceleration out of range: {decision['acceleration']}"
            assert decision["stance"] in ["neutral", "extended", "defending"], \
                f"{opponent.__name__} invalid stance: {decision['stance']}"

    def test_opponents_handle_edge_cases(self):
        """Test that opponents handle edge cases gracefully."""
        opponents = [
            distance_keeper_3m,
            stamina_waster,
            forward_mover,
            backward_mover,
            sideways_mover_smooth,
            strategic_retreater
        ]

        # Test with very large distance
        state_far = self.create_state(distance=100.0)
        # Test with very small distance
        state_close = self.create_state(distance=0.1)
        # Test with negative distance
        state_negative = self.create_state(distance=-50.0)

        for opponent in opponents:
            # Should not crash on edge cases
            decision_far = opponent.decide(state_far)
            assert isinstance(decision_far, dict)

            decision_close = opponent.decide(state_close)
            assert isinstance(decision_close, dict)

            decision_negative = opponent.decide(state_negative)
            assert isinstance(decision_negative, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])