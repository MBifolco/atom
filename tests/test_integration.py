"""
Integration tests for the complete boxing combat system.
Tests the full flow from fighter creation through combat to conclusion.
"""

import pytest
import numpy as np
from src.arena import WorldConfig, FighterState
from src.atom.runtime.arena.arena_1d_jax_jit import Arena1DJAXJit
from src.atom.training.gym_env import AtomCombatEnv


class TestIntegration:
    """Full integration tests."""

    def test_complete_fight_with_boxing_archetypes(self):
        """Test a complete fight between two boxing archetypes."""
        from fighters.examples.boxer import decide as boxer_action
        from fighters.examples.slugger import decide as slugger_action

        config = WorldConfig()
        boxer = FighterState.create("Boxer", 65.0, 2.0, config)
        slugger = FighterState.create("Slugger", 80.0, 10.0, config)

        arena = Arena1DJAXJit(boxer, slugger, config)

        # Run a fight for up to 100 ticks
        for tick in range(100):
            # Get fighter states for AI
            distance = abs(arena.fighter_a.position - arena.fighter_b.position)
            direction_a = 1.0 if arena.fighter_a.position < arena.fighter_b.position else -1.0
            direction_b = -direction_a

            state_boxer = {
                "you": arena.fighter_a.to_dict("Boxer"),
                "opponent": {
                    **arena.fighter_b.to_dict("Slugger"),
                    "distance": distance,
                    "direction": direction_a
                }
            }

            state_slugger = {
                "you": arena.fighter_b.to_dict("Slugger"),
                "opponent": {
                    **arena.fighter_a.to_dict("Boxer"),
                    "distance": distance,
                    "direction": direction_b
                }
            }

            # Get actions
            action_boxer = boxer_action(state_boxer)
            action_slugger = slugger_action(state_slugger)

            # Step
            events = arena.step(action_boxer, action_slugger)

            # Check if fight is over
            if arena.fighter_a.hp <= 0 or arena.fighter_b.hp <= 0:
                break

        # Someone should have taken damage
        assert arena.fighter_a.hp < boxer.hp or arena.fighter_b.hp < slugger.hp, \
            "Damage should be dealt in a fight"

        # Fight should have engaged (not just avoid each other)
        total_damage = (boxer.hp - arena.fighter_a.hp) + (slugger.hp - arena.fighter_b.hp)
        assert total_damage > 0, "Fighters should deal damage to each other"

    def test_gym_env_with_new_system(self):
        """Test the Gym environment works with 3-stance system."""
        config = WorldConfig()

        # Import a test opponent
        from fighters.examples.boxer import decide as opponent_action

        env = AtomCombatEnv(
            opponent_decision_func=opponent_action,
            fighter_mass=70.0,
            opponent_mass=70.0,
            config=config,
        )

        obs, info = env.reset()
        assert obs.shape == (13,)  # Check enhanced observation shape

        done = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated) and steps < 100:
            # Random action in correct range
            action = np.array([
                np.random.uniform(-1, 1),  # acceleration
                np.random.uniform(0, 2.99)  # stance (3 stances)
            ])

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Check observation is valid
            assert not np.any(np.isnan(obs)), f"NaN in observation at step {steps}"
            assert not np.any(np.isinf(obs)), f"Inf in observation at step {steps}"

        # Should have completed episode
        assert done or truncated or steps == 100

    def test_hit_event_generation(self):
        """Test that hit events are generated correctly."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 5.5, config)  # Start close
        fighter_b = FighterState.create("B", 70.0, 6.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Get them moving toward each other
        for _ in range(2):
            arena.step(
                {"acceleration": 0.5, "stance": "neutral"},
                {"acceleration": -0.5, "stance": "neutral"}
            )

        # Attack each other
        events = arena.step(
            {"acceleration": 0.5, "stance": "extended"},
            {"acceleration": -0.5, "stance": "extended"}
        )

        # Should generate events
        assert events is not None
        assert len(events) > 0

        # Check event structure
        for event in events:
            assert isinstance(event, dict)
            # Events should have type and other data
            if "damage" in event:
                assert event["damage"] >= 0

    def test_stamina_exhaustion_mechanics(self):
        """Test what happens when fighters run out of stamina."""
        config = WorldConfig()
        fighter = FighterState.create("Exhausted", 70.0, 5.0, config)
        opponent = FighterState.create("Fresh", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter, opponent, config)

        # Exhaust the fighter by modifying state
        initial_velocity = 2.0
        arena.state = arena.state._replace(
            fighter_a=arena.state.fighter_a.replace(stamina=0.0, velocity=initial_velocity)
        )

        # Step with extended stance (requires stamina)
        arena.step(
            {"acceleration": 1.0, "stance": "extended"},
            {"acceleration": 0.0, "stance": "neutral"}
        )

        # Velocity should be reduced when exhausted
        assert arena.fighter_a.velocity < initial_velocity, \
            "Exhausted fighter should have reduced velocity"

        # Should still be able to defend
        for _ in range(5):
            arena.step(
                {"acceleration": 0.0, "stance": "defending"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        # Defending should regenerate stamina
        assert arena.fighter_a.stamina > 0, \
            "Defending should regenerate stamina even when exhausted"

    def test_arena_boundary_physics(self):
        """Test fighters can't leave arena boundaries."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 1.0, config)  # Near left edge
        fighter_b = FighterState.create("B", 70.0, 11.0, config)  # Near right edge

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Try to move out of bounds
        for _ in range(20):
            arena.step(
                {"acceleration": -1.0, "stance": "neutral"},  # A tries to go left
                {"acceleration": 1.0, "stance": "neutral"}    # B tries to go right
            )

        # Should be clamped to arena bounds
        assert arena.fighter_a.position >= 0, "Fighter A should not go below 0"
        assert arena.fighter_b.position <= config.arena_width, \
            f"Fighter B should not exceed arena width {config.arena_width}"

    def test_discrete_hits_not_continuous(self):
        """Verify damage is discrete, not applied every tick."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 5.0, config)
        fighter_b = FighterState.create("B", 70.0, 5.5, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        damages = []
        hp_history = []

        # Continuous collision for 20 ticks
        for _ in range(20):
            hp_before = arena.fighter_b.hp
            arena.step(
                {"acceleration": 0.0, "stance": "extended"},
                {"acceleration": 0.0, "stance": "neutral"}
            )
            hp_after = arena.fighter_b.hp

            hp_history.append(hp_after)
            if hp_before > hp_after:
                damages.append(hp_before - hp_after)

        # Should have discrete hits, not continuous damage
        # Damage events should be separated by cooldown
        if len(damages) > 1:
            # Not every tick should have damage
            assert len(damages) < 20 / config.hit_cooldown_ticks + 1, \
                f"Too many hits: {len(damages)} in 20 ticks (cooldown: {config.hit_cooldown_ticks})"

    def test_different_mass_dynamics(self):
        """Test that mass affects combat dynamics correctly."""
        config = WorldConfig()

        # Light vs Heavy
        light = FighterState.create("Light", 45.0, 5.0, config)
        heavy = FighterState.create("Heavy", 85.0, 7.0, config)

        arena = Arena1DJAXJit(light, heavy, config)

        # Light fighter should be faster
        for _ in range(5):
            arena.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 1.0, "stance": "neutral"}
            )

        # Light fighter should accelerate faster (same force, less mass)
        assert abs(arena.fighter_a.velocity) > abs(arena.fighter_b.velocity), \
            "Light fighter should accelerate faster"

        # Check stamina differences
        light_stamina_max = light.max_stamina
        heavy_stamina_max = heavy.max_stamina

        assert light_stamina_max > heavy_stamina_max, \
            "Light fighter should have more stamina"

        # Check HP differences
        assert light.max_hp < heavy.max_hp, \
            "Heavy fighter should have more HP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])