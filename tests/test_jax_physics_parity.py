"""
Physics Parity Tests: JAX vs Python Implementation

Validates that Arena1DJAX produces identical results to Arena1D.

Test Coverage:
- Single step parity (position, velocity, HP, stamina match)
- Full episode parity (complete fight trajectory)
- Statistical parity (1000 episodes produce same win/loss distribution)
- Edge cases (wall collisions, stamina depletion, draws)
"""

import pytest
import numpy as np
from src.arena.arena_1d import Arena1D
from src.arena.arena_1d_jax import Arena1DJAX, FighterStateJAX
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig


class TestSingleStepParity:
    """Test that single physics steps produce identical results."""

    def test_identical_initialization(self):
        """JAX and Python arenas initialize with same state."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Check initial state matches
        assert arena_py.tick == arena_jax.tick == 0
        assert abs(arena_py.fighter_a.hp - arena_jax.fighter_a.hp) < 1e-6
        assert abs(arena_py.fighter_b.hp - arena_jax.fighter_b.hp) < 1e-6
        assert abs(arena_py.fighter_a.stamina - arena_jax.fighter_a.stamina) < 1e-6
        assert abs(arena_py.fighter_b.stamina - arena_jax.fighter_b.stamina) < 1e-6

    def test_single_step_no_collision(self):
        """Single step without collision produces identical state."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Both fighters move toward each other
        action_a = {"acceleration": 1.0, "stance": "neutral"}
        action_b = {"acceleration": -1.0, "stance": "neutral"}

        arena_py.step(action_a, action_b)
        arena_jax.step(action_a, action_b)

        # Verify states match
        assert arena_py.tick == arena_jax.tick == 1
        assert abs(arena_py.fighter_a.position - arena_jax.fighter_a.position) < 1e-6
        assert abs(arena_py.fighter_b.position - arena_jax.fighter_b.position) < 1e-6
        assert abs(arena_py.fighter_a.velocity - arena_jax.fighter_a.velocity) < 1e-6
        assert abs(arena_py.fighter_b.velocity - arena_jax.fighter_b.velocity) < 1e-6
        assert abs(arena_py.fighter_a.stamina - arena_jax.fighter_a.stamina) < 1e-6
        assert abs(arena_py.fighter_b.stamina - arena_jax.fighter_b.stamina) < 1e-6

    def test_single_step_with_collision(self):
        """Single step with collision produces identical damage."""
        config = WorldConfig()
        # Start fighters close together for immediate collision
        fighter_a = FighterState.create("Alice", 70.0, 5.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 6.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Both attack
        action_a = {"acceleration": 1.0, "stance": "extended"}
        action_b = {"acceleration": -1.0, "stance": "extended"}

        events_py = arena_py.step(action_a, action_b)
        events_jax = arena_jax.step(action_a, action_b)

        # Verify HP matches (damage was applied)
        # Note: JAX uses float32 by default, so use slightly larger tolerance
        assert abs(arena_py.fighter_a.hp - arena_jax.fighter_a.hp) < 1e-5
        assert abs(arena_py.fighter_b.hp - arena_jax.fighter_b.hp) < 1e-5

        # Verify events match
        if events_py and events_jax:
            assert abs(events_py[0]["damage_to_a"] - events_jax[0]["damage_to_a"]) < 0.1
            assert abs(events_py[0]["damage_to_b"] - events_jax[0]["damage_to_b"]) < 0.1

    def test_wall_collision_parity(self):
        """Wall collisions handled identically."""
        config = WorldConfig()
        # Start very close to wall so we hit it quickly
        fighter_a = FighterState.create("Alice", 70.0, 0.08, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Push into left wall
        action_a = {"acceleration": -2.0, "stance": "neutral"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}

        # Take 3 steps to actually hit the wall
        for _ in range(3):
            arena_py.step(action_a, action_b)
            arena_jax.step(action_a, action_b)

        # Should hit wall and stop
        assert abs(arena_py.fighter_a.position - arena_jax.fighter_a.position) < 1e-5
        assert arena_py.fighter_a.position == 0.0
        assert float(arena_jax.fighter_a.position) == 0.0
        assert abs(arena_py.fighter_a.velocity - arena_jax.fighter_a.velocity) < 1e-5

    def test_stamina_depletion_parity(self):
        """Stamina depletion handled identically."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 5.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        # Manually set low stamina
        fighter_a.stamina = 0.1
        fighter_b.stamina = 0.1

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Try to attack with no stamina
        action_a = {"acceleration": 2.0, "stance": "extended"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}

        arena_py.step(action_a, action_b)
        arena_jax.step(action_a, action_b)

        # Stamina enforcement should match
        assert arena_py.fighter_a.stance == arena_jax.fighter_a.stance
        assert abs(arena_py.fighter_a.stamina - arena_jax.fighter_a.stamina) < 1e-6


class TestFullEpisodeParity:
    """Test that full episodes produce identical trajectories."""

    def test_full_episode_trajectory(self):
        """100-step episode produces identical state at each step."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Run 100 steps with simple actions
        for step in range(100):
            action_a = {"acceleration": 1.0 if step % 2 == 0 else -0.5, "stance": "neutral"}
            action_b = {"acceleration": -0.8, "stance": "neutral"}

            arena_py.step(action_a, action_b)
            arena_jax.step(action_a, action_b)

            # Verify state matches at every step
            assert abs(arena_py.fighter_a.position - arena_jax.fighter_a.position) < 1e-5, \
                f"Position mismatch at step {step}"
            assert abs(arena_py.fighter_a.velocity - arena_jax.fighter_a.velocity) < 1e-5
            assert abs(arena_py.fighter_a.hp - arena_jax.fighter_a.hp) < 1e-5
            assert abs(arena_py.fighter_a.stamina - arena_jax.fighter_a.stamina) < 1e-5
            assert abs(arena_py.fighter_b.position - arena_jax.fighter_b.position) < 1e-5
            assert abs(arena_py.fighter_b.velocity - arena_jax.fighter_b.velocity) < 1e-5
            assert abs(arena_py.fighter_b.hp - arena_jax.fighter_b.hp) < 1e-5
            assert abs(arena_py.fighter_b.stamina - arena_jax.fighter_b.stamina) < 1e-5

    def test_fight_to_completion_parity(self):
        """Complete fight produces same winner."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Fight until someone wins (max 250 ticks)
        for _ in range(250):
            # Simple aggressive strategy
            action_a = {"acceleration": 1.0, "stance": "extended"}
            action_b = {"acceleration": -1.0, "stance": "extended"}

            arena_py.step(action_a, action_b)
            arena_jax.step(action_a, action_b)

            if arena_py.is_finished() or arena_jax.is_finished():
                break

        # Both should finish at same time with same winner
        assert arena_py.is_finished() == arena_jax.is_finished()
        assert arena_py.get_winner() == arena_jax.get_winner()


class TestStatisticalParity:
    """Test that 1000+ episodes produce same statistical outcomes."""

    @pytest.mark.slow
    def test_1000_episodes_same_outcomes(self):
        """1000 random episodes produce same win/loss statistics."""
        config = WorldConfig()
        num_episodes = 1000
        max_ticks = 250

        py_wins_a = 0
        py_wins_b = 0
        py_draws = 0

        jax_wins_a = 0
        jax_wins_b = 0
        jax_draws = 0

        # Run 1000 parallel episodes
        for episode in range(num_episodes):
            # Use episode number as seed for reproducibility
            seed = episode

            fighter_a = FighterState.create("Alice", 70.0, 2.0, config)
            fighter_b = FighterState.create("Bob", 75.0, 10.0, config)

            arena_py = Arena1D(fighter_a, fighter_b, config, seed=seed)
            arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=seed)

            # Run episode with deterministic actions based on seed
            np.random.seed(seed)
            for tick in range(max_ticks):
                accel_a = np.random.uniform(-1, 1)
                accel_b = np.random.uniform(-1, 1)
                stance_a = np.random.choice(["neutral", "extended", "retracted", "defending"])
                stance_b = np.random.choice(["neutral", "extended", "retracted", "defending"])

                action_a = {"acceleration": accel_a, "stance": stance_a}
                action_b = {"acceleration": accel_b, "stance": stance_b}

                arena_py.step(action_a, action_b)
                arena_jax.step(action_a, action_b)

                if arena_py.is_finished():
                    break

            # Tally results
            py_winner = arena_py.get_winner()
            jax_winner = arena_jax.get_winner()

            # Both should have same winner
            assert py_winner == jax_winner, \
                f"Episode {episode}: Python winner={py_winner}, JAX winner={jax_winner}"

            if py_winner == "Alice":
                py_wins_a += 1
                jax_wins_a += 1
            elif py_winner == "Bob":
                py_wins_b += 1
                jax_wins_b += 1
            elif py_winner == "draw":
                py_draws += 1
                jax_draws += 1

        # Statistics should be identical (since we use same seed)
        assert py_wins_a == jax_wins_a
        assert py_wins_b == jax_wins_b
        assert py_draws == jax_draws

        print(f"\nStatistical Parity Results ({num_episodes} episodes):")
        print(f"  Alice wins: {py_wins_a} (Python) vs {jax_wins_a} (JAX)")
        print(f"  Bob wins: {py_wins_b} (Python) vs {jax_wins_b} (JAX)")
        print(f"  Draws: {py_draws} (Python) vs {jax_draws} (JAX)")


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_simultaneous_ko_parity(self):
        """Both fighters KO simultaneously produces same result."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 5.0, config)
        fighter_b = FighterState.create("Bob", 70.0, 6.0, config)

        # Set both to very low HP
        fighter_a.hp = 1.0
        fighter_b.hp = 1.0

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # Both attack for mutual KO
        action_a = {"acceleration": 2.0, "stance": "extended"}
        action_b = {"acceleration": -2.0, "stance": "extended"}

        arena_py.step(action_a, action_b)
        arena_jax.step(action_a, action_b)

        # Both should register as draw
        assert arena_py.get_winner() == arena_jax.get_winner() == "draw"

    def test_defensive_stance_parity(self):
        """Defensive stance damage reduction matches."""
        config = WorldConfig()
        fighter_a = FighterState.create("Alice", 70.0, 5.0, config)
        fighter_b = FighterState.create("Bob", 75.0, 6.0, config)

        arena_py = Arena1D(fighter_a, fighter_b, config, seed=42)
        arena_jax = Arena1DJAX(fighter_a, fighter_b, config, seed=42)

        # A attacks, B defends
        action_a = {"acceleration": 1.0, "stance": "extended"}
        action_b = {"acceleration": -0.5, "stance": "defending"}

        arena_py.step(action_a, action_b)
        arena_jax.step(action_a, action_b)

        # HP should match (defensive stance applied same damage reduction)
        assert abs(arena_py.fighter_b.hp - arena_jax.fighter_b.hp) < 1e-6


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
