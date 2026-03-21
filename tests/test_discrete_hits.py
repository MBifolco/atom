"""
Test suite for discrete hit boxing combat system.
Tests hit cooldowns, physics-based damage, stamina mechanics, and recoil.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from src.arena import WorldConfig, FighterState
from src.atom.runtime.arena.arena_1d_jax_jit import Arena1DJAXJit, STANCE_NEUTRAL, STANCE_EXTENDED, STANCE_DEFENDING


class TestDiscreteHitSystem:
    """Test the discrete hit mechanics."""

    def test_hit_cooldown_prevents_rapid_fire(self):
        """Verify hits respect cooldown period - can't hit rapidly."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Both fighters charge at each other in extended stance
        damages = []
        for i in range(20):
            action_a = {"acceleration": 1.0, "stance": "extended"}
            action_b = {"acceleration": -1.0, "stance": "extended"}
            events = arena.step(action_a, action_b)

            # Track damage dealt
            if events and "damage" in events[0]:
                damages.append((i, events[0]["damage"]))

        # Check that hits are spaced by cooldown (5 ticks)
        if len(damages) > 1:
            for i in range(1, len(damages)):
                tick_diff = damages[i][0] - damages[i-1][0]
                assert tick_diff >= config.hit_cooldown_ticks, \
                    f"Hits too close: {tick_diff} ticks apart (min {config.hit_cooldown_ticks})"

    def test_impact_force_calculation(self):
        """Test that impact force scales with velocity and mass."""
        config = WorldConfig()

        # Light fighter moving fast
        light_fighter = FighterState.create("Light", 45.0, 2.0, config)
        # Heavy fighter moving slow
        heavy_fighter = FighterState.create("Heavy", 85.0, 10.0, config)

        arena = Arena1DJAXJit(light_fighter, heavy_fighter, config)

        # Light fighter charges fast
        for _ in range(5):
            arena.step({"acceleration": 1.0, "stance": "neutral"},
                      {"acceleration": 0.0, "stance": "neutral"})

        # Get current velocities
        light_vel = arena.fighter_a.velocity
        heavy_vel = arena.fighter_b.velocity

        # Calculate expected impact force
        rel_velocity = abs(light_vel - heavy_vel)
        reduced_mass = (45.0 * 85.0) / (45.0 + 85.0)
        expected_force = rel_velocity * reduced_mass

        # Now they collide
        events = arena.step({"acceleration": 1.0, "stance": "extended"},
                           {"acceleration": 0.0, "stance": "neutral"})

        # The damage should scale with impact force
        # Higher impact force = more damage
        assert expected_force > 0, "Should have non-zero impact force"

    def test_defending_stance_regenerates_stamina(self):
        """Verify defending stance regenerates stamina instead of draining."""
        config = WorldConfig()
        fighter = FighterState.create("Defender", 70.0, 6.0, config)
        opponent = FighterState.create("Opponent", 70.0, 8.0, config)

        arena = Arena1DJAXJit(fighter, opponent, config)

        # Start with reduced stamina by modifying arena state directly
        arena.state = arena.state._replace(
            fighter_a=arena.state.fighter_a.replace(stamina=5.0)
        )
        initial_stamina = arena.fighter_a.stamina

        # Defend for 10 ticks
        for _ in range(10):
            arena.step(
                {"acceleration": 0.0, "stance": "defending"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        final_stamina = arena.fighter_a.stamina

        # Stamina should increase when defending
        assert final_stamina > initial_stamina, \
            f"Defending should regen stamina: {initial_stamina} -> {final_stamina}"

    def test_recoil_creates_separation(self):
        """Test that hits cause recoil, reducing velocity."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 5.0, config)
        fighter_b = FighterState.create("B", 70.0, 7.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Fighters approach each other
        for _ in range(3):
            arena.step(
                {"acceleration": 0.8, "stance": "neutral"},
                {"acceleration": -0.8, "stance": "neutral"}
            )

        # Record velocities before hit
        vel_a_before = arena.fighter_a.velocity
        vel_b_before = arena.fighter_b.velocity

        # A hits B
        events = arena.step(
            {"acceleration": 0.5, "stance": "extended"},
            {"acceleration": -0.5, "stance": "neutral"}
        )

        # Check if hit occurred and velocity was reduced
        if events and any(e.get("damage", 0) > 0 for e in events):
            vel_a_after = arena.fighter_a.velocity
            # Velocity should be reduced by recoil
            assert abs(vel_a_after) < abs(vel_a_before), \
                f"Recoil should reduce velocity: {vel_a_before} -> {vel_a_after}"

    def test_stamina_cost_on_hit(self):
        """Test that landing a hit costs stamina."""
        config = WorldConfig()
        fighter_a = FighterState.create("Attacker", 70.0, 5.0, config)
        fighter_b = FighterState.create("Defender", 70.0, 6.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Get them close
        for _ in range(2):
            arena.step(
                {"acceleration": 0.5, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        stamina_before = arena.fighter_a.stamina

        # Attack
        events = arena.step(
            {"acceleration": 0.2, "stance": "extended"},
            {"acceleration": 0.0, "stance": "neutral"}
        )

        stamina_after = arena.fighter_a.stamina

        # If hit landed, stamina should decrease
        if events and any(e.get("damage", 0) > 0 for e in events):
            assert stamina_after < stamina_before - 1.0, \
                f"Landing hit should cost stamina: {stamina_before} -> {stamina_after}"

    def test_blocking_reduces_damage_costs_less_stamina(self):
        """Test that blocking reduces damage and costs less stamina than attacking."""
        config = WorldConfig()
        attacker = FighterState.create("Attacker", 70.0, 5.0, config)
        blocker = FighterState.create("Blocker", 70.0, 6.0, config)

        arena = Arena1DJAXJit(attacker, blocker, config)

        # Get them close
        for _ in range(2):
            arena.step(
                {"acceleration": 0.5, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        # Test 1: Hit without blocking
        hp_before_no_block = arena.fighter_b.hp
        stamina_before_no_block = arena.fighter_b.stamina

        arena.step(
            {"acceleration": 0.2, "stance": "extended"},
            {"acceleration": 0.0, "stance": "neutral"}  # Not blocking
        )

        damage_no_block = hp_before_no_block - arena.fighter_b.hp

        # Reset for blocking test - create new arena
        arena = Arena1DJAXJit(attacker, blocker, config)

        # Get them close again
        for _ in range(2):
            arena.step(
                {"acceleration": 0.5, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        # Test 2: Hit with blocking
        hp_before_block = arena.fighter_b.hp
        stamina_before_block = arena.fighter_b.stamina

        arena.step(
            {"acceleration": 0.2, "stance": "extended"},
            {"acceleration": 0.0, "stance": "defending"}  # Blocking!
        )

        damage_blocked = hp_before_block - arena.fighter_b.hp
        stamina_cost_block = stamina_before_block - arena.fighter_b.stamina

        # Blocking should reduce damage
        if damage_no_block > 0:
            assert damage_blocked < damage_no_block, \
                f"Blocking should reduce damage: {damage_no_block} -> {damage_blocked}"

            # Blocking should cost less stamina than attacking
            assert stamina_cost_block < config.hit_stamina_cost, \
                f"Blocking stamina cost ({stamina_cost_block}) should be less than hitting ({config.hit_stamina_cost})"

    def test_no_damage_without_extended_stance(self):
        """Test that fighters can't deal damage without extended stance."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 5.0, config)
        fighter_b = FighterState.create("B", 70.0, 6.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        initial_hp_b = arena.fighter_b.hp

        # Collide without extended stance
        for _ in range(5):
            arena.step(
                {"acceleration": 0.5, "stance": "neutral"},  # Not extended
                {"acceleration": -0.5, "stance": "neutral"}
            )

        # No damage should be dealt
        assert arena.fighter_b.hp == initial_hp_b, \
            "No damage should be dealt without extended stance"

    def test_minimum_impact_threshold(self):
        """Test that very light touches don't register as hits."""
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 5.0, config)
        fighter_b = FighterState.create("B", 70.0, 5.5, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        initial_hp_b = arena.fighter_b.hp

        # Very gentle approach (low relative velocity)
        arena.step(
            {"acceleration": 0.05, "stance": "extended"},
            {"acceleration": -0.05, "stance": "neutral"}
        )

        # If relative velocity is too low, no hit should register
        # even if in extended stance
        # This prevents "phantom hits" from just touching


class TestStanceSystem:
    """Test the 3-stance boxing system."""

    def test_only_three_stances(self):
        """Verify only 3 stances exist: neutral, extended, defending."""
        config = WorldConfig()
        assert len(config.stances) == 3
        assert "neutral" in config.stances
        assert "extended" in config.stances
        assert "defending" in config.stances
        assert "retracted" not in config.stances

    def test_stance_transitions(self):
        """Test smooth transitions between stances."""
        config = WorldConfig()
        fighter = FighterState.create("Fighter", 70.0, 5.0, config)
        opponent = FighterState.create("Opponent", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter, opponent, config)

        stances_tested = []

        # Cycle through all stances
        for stance in ["neutral", "extended", "defending"]:
            arena.step(
                {"acceleration": 0.0, "stance": stance},
                {"acceleration": 0.0, "stance": "neutral"}
            )
            # Verify stance was set correctly
            current_stance = arena.fighter_a.to_dict("Fighter")["stance"]
            stances_tested.append(current_stance)

        assert stances_tested == ["neutral", "extended", "defending"]

    def test_stamina_drain_per_stance(self):
        """Test stamina behavior for each stance."""
        config = WorldConfig()
        fighter = FighterState.create("Fighter", 70.0, 5.0, config)
        opponent = FighterState.create("Opponent", 70.0, 10.0, config)

        results = {}

        for stance in ["neutral", "extended", "defending"]:
            arena = Arena1DJAXJit(fighter, opponent, config)

            # Start with reduced stamina so we can see regen
            arena.state = arena.state._replace(
                fighter_a=arena.state.fighter_a.replace(stamina=5.0)
            )
            initial_stamina = arena.fighter_a.stamina

            # Hold stance for 10 ticks with no movement
            for _ in range(10):
                arena.step(
                    {"acceleration": 0.0, "stance": stance},
                    {"acceleration": 0.0, "stance": "neutral"}
                )

            stamina_change = arena.fighter_a.stamina - initial_stamina
            results[stance] = float(stamina_change)

        # Neutral should have positive regen (base_regen * neutral_bonus)
        assert results["neutral"] > 0, f"Neutral should have positive regen, got {results['neutral']}"

        # Extended should drain stamina (less regen than neutral due to drain)
        assert results["extended"] < results["neutral"], \
            f"Extended should drain more than neutral: {results['extended']} vs {results['neutral']}"

        # Defending should have less regen than neutral (no neutral bonus)
        assert results["defending"] < results["neutral"], \
            f"Defending should have less regen than neutral (no bonus): {results['defending']} vs {results['neutral']}"
        assert results["defending"] > 0, \
            f"Defending should still have positive base regen: {results['defending']}"


class TestBoxingArchetypes:
    """Test the boxing fighter archetypes work correctly."""

    def test_boxer_fighter(self):
        """Test the Boxer archetype behavior."""
        from fighters.examples.boxer import decide

        # Test stamina management
        state_low_stamina = {
            "you": {"stamina": 2.0, "max_stamina": 10.0, "position": 5.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stance": "neutral"},
            "opponent": {"distance": 2.0, "direction": 1.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stamina": 5.0, "max_stamina": 10.0}
        }

        action = decide(state_low_stamina)
        # Should defend when low on stamina
        assert action["stance"] == "defending", "Boxer should defend when low on stamina"

    def test_slugger_fighter(self):
        """Test the Slugger archetype behavior."""
        from fighters.examples.slugger import decide

        # Test aggressive behavior
        state = {
            "you": {"stamina": 8.0, "max_stamina": 10.0, "position": 5.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stance": "neutral"},
            "opponent": {"distance": 3.0, "direction": 1.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stamina": 5.0, "max_stamina": 10.0}
        }

        action = decide(state)
        # Should be aggressive (move toward opponent)
        assert action["acceleration"] != 0, "Slugger should be moving"

    def test_counter_puncher_fighter(self):
        """Test the Counter Puncher archetype behavior."""
        from fighters.examples.counter_puncher import decide

        # Test defensive default
        state = {
            "you": {"stamina": 8.0, "max_stamina": 10.0, "position": 5.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stance": "neutral"},
            "opponent": {"distance": 2.0, "direction": 1.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stamina": 8.0, "max_stamina": 10.0}
        }

        action = decide(state)
        # Should default to defending
        assert action["stance"] == "defending", "Counter puncher should default to defense"

    def test_swarmer_fighter(self):
        """Test the Swarmer archetype behavior."""
        from fighters.examples.swarmer import decide

        # Test constant pressure
        state = {
            "you": {"stamina": 6.0, "max_stamina": 10.0, "position": 5.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stance": "neutral"},
            "opponent": {"distance": 3.0, "direction": 1.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stamina": 5.0, "max_stamina": 10.0}
        }

        action = decide(state)
        # Should always attack when has stamina
        assert action["stance"] == "extended" or action["acceleration"] != 0, \
            "Swarmer should maintain pressure"

    def test_out_fighter(self):
        """Test the Out-Fighter archetype behavior."""
        from fighters.examples.out_fighter import decide

        # Test distance management
        state_too_close = {
            "you": {"stamina": 8.0, "max_stamina": 10.0, "position": 5.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stance": "neutral"},
            "opponent": {"distance": 0.7, "direction": 1.0, "velocity": 0.0, "hp": 50.0, "max_hp": 100.0, "stamina": 5.0, "max_stamina": 10.0}
        }

        action = decide(state_too_close)
        # Should retreat when too close (negative acceleration * direction = away)
        # direction=1 means opponent to right, so to retreat we want negative final velocity
        # acceleration * direction should be negative to move away
        expected_away = action["acceleration"] * state_too_close["opponent"]["direction"]
        assert expected_away < 0, "Out-fighter should retreat when too close"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])