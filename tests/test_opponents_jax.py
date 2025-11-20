"""
Tests for JAX-compatible opponent policies.
"""

import pytest
import jax.numpy as jnp
from src.training.opponents_jax import (
    stationary_neutral_jax,
    stationary_extended_jax,
    stationary_defending_jax,
    approach_slow_jax,
    flee_always_jax,
    circle_left_jax,
    circle_right_jax
)
from src.arena import WorldConfig
from src.arena.arena_1d_jax_jit import FighterStateJAX, ArenaStateJAX


class TestStationaryOpponents:
    """Test stationary opponent policies."""

    def test_stationary_neutral(self):
        """Test stationary neutral opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = stationary_neutral_jax(state, config)

        assert action[0] == 0.0  # No acceleration
        assert action[1] == 0  # Neutral stance

    def test_stationary_extended(self):
        """Test stationary extended opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = stationary_extended_jax(state, config)

        assert action[0] == 0.0  # No acceleration
        assert action[1] == 1  # Extended stance

    def test_stationary_defending(self):
        """Test stationary defending opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = stationary_defending_jax(state, config)

        assert action[0] == 0.0  # No acceleration
        assert int(action[1]) == 2  # Defending stance (was 3, now 2 after removing retracted)


class TestMovementOpponents:
    """Test moving opponent policies."""

    def test_approach_slow(self):
        """Test slow approach opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = approach_slow_jax(state, config)

        # Should return valid action
        assert action.shape == (2,)
        assert action[1] == 0  # Neutral stance

    def test_flee_always(self):
        """Test flee always opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = flee_always_jax(state, config)

        # Should return valid action
        assert action.shape == (2,)

    def test_circle_left(self):
        """Test circle left opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = circle_left_jax(state, config)

        # Should return valid action
        assert action.shape == (2,)
        # Should be moving left (negative acceleration) unless near wall
        assert action[0] != 0  # Some movement

    def test_circle_right(self):
        """Test circle right opponent."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=5.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        action = circle_right_jax(state, config)

        # Should return valid action
        assert action.shape == (2,)
        # Should be moving right (positive acceleration) unless near wall
        assert action[0] != 0  # Some movement


class TestOpponentPoliciesAreJaxCompatible:
    """Test that policies work with JAX."""

    def test_policies_return_jax_arrays(self):
        """Test all policies return JAX arrays."""
        config = WorldConfig()
        fighter = FighterStateJAX(
            mass=70.0, position=6.0, velocity=0.0,
            hp=100.0, max_hp=100.0, stamina=10.0, max_stamina=10.0,
            stance=0, last_hit_tick=-999
        )
        state = ArenaStateJAX(fighter, fighter, 0)

        policies = [
            stationary_neutral_jax,
            stationary_extended_jax,
            stationary_defending_jax,
            approach_slow_jax,
            flee_always_jax,
            circle_left_jax,
            circle_right_jax
        ]

        for policy in policies:
            action = policy(state, config)
            # Should be JAX array
            assert isinstance(action, jnp.ndarray)
            assert action.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
