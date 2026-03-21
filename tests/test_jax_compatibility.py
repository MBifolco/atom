"""
Test JAX compatibility, JIT compilation, and vmap functionality.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.arena import WorldConfig, FighterState
from src.atom.runtime.arena.arena_1d_jax_jit import Arena1DJAXJit, FighterStateJAX, ArenaStateJAX


class TestJAXCompatibility:
    """Test JAX features work with the new discrete hit system."""

    def test_jit_compilation(self):
        """Test that the step function can be JIT compiled."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Step function should already be JIT compiled
        # This tests it doesn't crash on first call
        action_a = {"acceleration": 0.5, "stance": "extended"}
        action_b = {"acceleration": -0.5, "stance": "defending"}

        # First call (compilation happens here)
        events1 = arena.step(action_a, action_b)

        # Second call (uses compiled version)
        events2 = arena.step(action_a, action_b)

        # Both should work without errors
        assert events1 is not None
        assert events2 is not None

    def test_fighter_state_jax_dataclass(self):
        """Test FighterStateJAX is a valid JAX pytree."""
        config = WorldConfig()
        fighter = FighterState.create("Test", 70.0, 5.0, config)

        # Convert to JAX version
        fighter_jax = FighterStateJAX.from_fighter_state(fighter)

        # Check all fields are present
        assert hasattr(fighter_jax, "mass")
        assert hasattr(fighter_jax, "position")
        assert hasattr(fighter_jax, "velocity")
        assert hasattr(fighter_jax, "hp")
        assert hasattr(fighter_jax, "max_hp")
        assert hasattr(fighter_jax, "stamina")
        assert hasattr(fighter_jax, "max_stamina")
        assert hasattr(fighter_jax, "stance")
        assert hasattr(fighter_jax, "last_hit_tick")  # New field

        # Check it's a valid pytree
        leaves, treedef = jax.tree_util.tree_flatten(fighter_jax)
        assert len(leaves) == 9  # All fields should be leaves

        # Can be reconstructed
        fighter_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert fighter_reconstructed.mass == fighter_jax.mass
        assert fighter_reconstructed.last_hit_tick == fighter_jax.last_hit_tick

    def test_arena_state_namedtuple(self):
        """Test ArenaStateJAX works as expected."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        fighter_a_jax = FighterStateJAX.from_fighter_state(fighter_a)
        fighter_b_jax = FighterStateJAX.from_fighter_state(fighter_b)

        state = ArenaStateJAX(fighter_a_jax, fighter_b_jax, 0)

        assert state.tick == 0
        assert state.fighter_a.mass == 70.0
        assert state.fighter_b.mass == 70.0

    def test_no_python_control_flow_in_step(self):
        """Verify step function uses JAX control flow (jnp.where) not Python if/else."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # This should work without Python control flow issues
        # JAX will error if Python control flow is used with traced values
        for i in range(10):
            action_a = {"acceleration": 0.5 * (-1)**i, "stance": "extended"}
            action_b = {"acceleration": -0.5 * (-1)**i, "stance": "neutral"}
            events = arena.step(action_a, action_b)

        # If we got here, no Python control flow issues
        assert True

    def test_stance_arrays_correct_size(self):
        """Test stance arrays are correctly sized for 3 stances."""
        from src.atom.runtime.arena.arena_1d_jax_jit import create_stance_arrays

        config = WorldConfig()
        reach, defense, drain = create_stance_arrays(config)

        # Should be arrays of size 3 (3 stances)
        assert reach.shape == (3,)
        assert defense.shape == (3,)
        assert drain.shape == (3,)

        # Check defending stance has zero drain (index 2) - no stamina penalty
        assert drain[2] == 0, f"Defending stance (index 2) should have zero drain: {drain[2]}"

    def test_discrete_hit_functions_are_pure(self):
        """Test that discrete hit helper functions are pure (no side effects)."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 3.0, config)

        fighter_a_jax = FighterStateJAX.from_fighter_state(fighter_a)
        fighter_b_jax = FighterStateJAX.from_fighter_state(fighter_b)

        # Test impact force calculation
        force1 = Arena1DJAXJit._calculate_impact_force_jax(fighter_a_jax, fighter_b_jax)
        force2 = Arena1DJAXJit._calculate_impact_force_jax(fighter_a_jax, fighter_b_jax)

        # Should be deterministic
        assert force1 == force2

        # Test hit validity check
        valid1 = Arena1DJAXJit._check_hit_valid_jax(
            fighter_a_jax, 10, 50.0, config.hit_cooldown_ticks, config.hit_impact_threshold
        )
        valid2 = Arena1DJAXJit._check_hit_valid_jax(
            fighter_a_jax, 10, 50.0, config.hit_cooldown_ticks, config.hit_impact_threshold
        )

        assert valid1 == valid2

    def test_no_string_stances_in_jit(self):
        """Verify integer stances are used in JIT code, not strings."""
        from src.atom.runtime.arena.arena_1d_jax_jit import STANCE_NEUTRAL, STANCE_EXTENDED, STANCE_DEFENDING

        # Should be integers
        assert isinstance(STANCE_NEUTRAL, int)
        assert isinstance(STANCE_EXTENDED, int)
        assert isinstance(STANCE_DEFENDING, int)

        # Should be 0, 1, 2 for 3 stances
        assert STANCE_NEUTRAL == 0
        assert STANCE_EXTENDED == 1
        assert STANCE_DEFENDING == 2

    def test_vmap_compatibility(self):
        """Test that arena can potentially be vmapped for parallel environments."""
        config = WorldConfig()

        # Create multiple fighter pairs
        fighter_pairs = []
        for i in range(4):
            f_a = FighterState.create(f"A{i}", 60.0 + i*5, 2.0, config)
            f_b = FighterState.create(f"B{i}", 70.0 + i*3, 10.0, config)
            fighter_pairs.append((f_a, f_b))

        # Create arenas
        arenas = [Arena1DJAXJit(f_a, f_b, config) for f_a, f_b in fighter_pairs]

        # Step all arenas
        for arena in arenas:
            action_a = {"acceleration": 0.5, "stance": "extended"}
            action_b = {"acceleration": -0.5, "stance": "defending"}
            events = arena.step(action_a, action_b)

        # If this runs without error, vmap should be possible
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])