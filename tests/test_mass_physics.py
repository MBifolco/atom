"""
Test mass-based physics mechanics.

Tests for bug fix: Mass wasn't affecting acceleration.
Light fighters should accelerate faster than heavy fighters for same force.
"""

import pytest
from src.arena import WorldConfig, FighterState, Arena1DJAXJit


class TestMassPhysics:
    """Test that mass correctly affects fighter physics."""

    def test_light_fighters_accelerate_faster(self):
        """
        Bug fix test: Verify light fighters accelerate faster than heavy ones.

        Previously, velocity update didn't account for mass:
            new_velocity = velocity + accel * dt

        Fixed to use F=ma physics:
            actual_accel = accel * (70.0 / mass)
            new_velocity = velocity + actual_accel * dt

        This makes 70kg the baseline - lighter accelerate faster, heavier slower.
        """
        config = WorldConfig()

        # Light fighter (45kg)
        light = FighterState.create("Light", 45.0, 5.0, config)
        # Heavy fighter (85kg)
        heavy = FighterState.create("Heavy", 85.0, 7.0, config)

        arena = Arena1DJAXJit(light, heavy, config)

        # Both accelerate with same input
        for _ in range(5):
            arena.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 1.0, "stance": "neutral"}
            )

        light_velocity = float(arena.fighter_a.velocity)
        heavy_velocity = float(arena.fighter_b.velocity)

        # Light fighter should have higher velocity
        assert light_velocity > heavy_velocity, \
            f"Light fighter (45kg) should accelerate faster than heavy (85kg): " \
            f"{light_velocity:.6f} vs {heavy_velocity:.6f}"

        # Check expected ratio (roughly 70/45 vs 70/85 = 1.56 vs 0.82)
        # Light should be ~1.9x faster than heavy
        ratio = light_velocity / heavy_velocity
        assert 1.7 < ratio < 2.1, \
            f"Velocity ratio should be ~1.9x (70/45 ÷ 70/85 = 1.89): got {ratio:.2f}x"

    def test_medium_fighter_baseline(self):
        """Test that 70kg fighter is the baseline (mass_factor = 1.0)."""
        config = WorldConfig()

        # 70kg fighter (baseline)
        baseline = FighterState.create("Baseline", 70.0, 5.0, config)
        arena = Arena1DJAXJit(baseline, FighterState.create("Dummy", 70.0, 10.0, config), config)

        # Accelerate for 3 ticks
        for _ in range(3):
            arena.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        baseline_velocity = float(arena.fighter_a.velocity)

        # Now test light fighter (should be faster)
        light = FighterState.create("Light", 45.0, 5.0, config)
        arena2 = Arena1DJAXJit(light, FighterState.create("Dummy", 70.0, 10.0, config), config)

        for _ in range(3):
            arena2.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        light_velocity = float(arena2.fighter_a.velocity)

        assert light_velocity > baseline_velocity, \
            f"45kg fighter should accelerate faster than 70kg baseline"

    def test_mass_affects_stamina_cost(self):
        """Test that mass affects stamina cost of acceleration."""
        config = WorldConfig()

        # Light fighter (uses less stamina per acceleration)
        light = FighterState.create("Light", 45.0, 5.0, config)
        # Heavy fighter (uses more stamina per acceleration)
        heavy = FighterState.create("Heavy", 85.0, 7.0, config)

        arena_light = Arena1DJAXJit(light, FighterState.create("D", 70.0, 10.0, config), config)
        arena_heavy = Arena1DJAXJit(heavy, FighterState.create("D", 70.0, 10.0, config), config)

        # Both start with same stamina percentage
        arena_light.state = arena_light.state._replace(
            fighter_a=arena_light.state.fighter_a.replace(stamina=5.0)
        )
        arena_heavy.state = arena_heavy.state._replace(
            fighter_a=arena_heavy.state.fighter_a.replace(stamina=5.0)
        )

        initial_light = arena_light.fighter_a.stamina
        initial_heavy = arena_heavy.fighter_a.stamina

        # Both accelerate the same amount
        for _ in range(5):
            arena_light.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )
            arena_heavy.step(
                {"acceleration": 1.0, "stance": "neutral"},
                {"acceleration": 0.0, "stance": "neutral"}
            )

        stamina_used_light = initial_light - arena_light.fighter_a.stamina
        stamina_used_heavy = initial_heavy - arena_heavy.fighter_a.stamina

        # Heavy fighter should use MORE stamina (mass_factor in stamina cost)
        assert stamina_used_heavy > stamina_used_light, \
            f"Heavy fighter should use more stamina: " \
            f"Light used {stamina_used_light:.4f}, Heavy used {stamina_used_heavy:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
