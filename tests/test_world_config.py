"""
Test WorldConfig with new discrete hit parameters.
"""

import pytest
from src.arena.world_config import WorldConfig, StanceConfig


class TestWorldConfig:
    """Test the world configuration with new hit system parameters."""

    def test_discrete_hit_parameters_exist(self):
        """Verify all discrete hit parameters are present."""
        config = WorldConfig()

        # Check new parameters exist
        assert hasattr(config, "hit_cooldown_ticks")
        assert hasattr(config, "hit_impact_threshold")
        assert hasattr(config, "hit_recoil_multiplier")
        assert hasattr(config, "hit_stamina_cost")
        assert hasattr(config, "block_stamina_cost")

        # Check default values are reasonable
        assert config.hit_cooldown_ticks > 0
        assert config.hit_impact_threshold > 0
        assert 0 < config.hit_recoil_multiplier < 1
        assert config.hit_stamina_cost > 0
        assert config.block_stamina_cost > 0
        assert config.block_stamina_cost < config.hit_stamina_cost

    def test_three_stances_only(self):
        """Verify only 3 stances exist."""
        config = WorldConfig()
        assert len(config.stances) == 3
        assert set(config.stances.keys()) == {"neutral", "extended", "defending"}

    def test_defending_stance_regenerates(self):
        """Verify defending stance has zero drain (no stamina penalty)."""
        config = WorldConfig()
        defending_stance = config.stances["defending"]

        assert defending_stance.drain == 0, \
            f"Defending stance should have zero drain (no penalty), got {defending_stance.drain}"

    def test_stance_defense_values(self):
        """Test stance defense multipliers are reasonable."""
        config = WorldConfig()

        neutral_def = config.stances["neutral"].defense
        extended_def = config.stances["extended"].defense
        defending_def = config.stances["defending"].defense

        # Extended should be most vulnerable
        assert extended_def < neutral_def, "Extended should be more vulnerable than neutral"

        # Defending should have best defense
        assert defending_def > neutral_def, "Defending should have better defense than neutral"

        # Reasonable ranges
        assert 0.5 < extended_def < 1.0, f"Extended defense out of range: {extended_def}"
        assert 1.0 < defending_def < 2.0, f"Defending defense out of range: {defending_def}"

    def test_fighter_stats_calculation(self):
        """Test HP and stamina calculation from mass."""
        config = WorldConfig()

        # Light fighter
        light_stats = config.calculate_fighter_stats(45.0)
        # Heavy fighter
        heavy_stats = config.calculate_fighter_stats(85.0)

        # HP increases with mass
        assert heavy_stats["max_hp"] > light_stats["max_hp"], \
            "Heavy fighter should have more HP"

        # Stamina decreases with mass
        assert heavy_stats["max_stamina"] < light_stats["max_stamina"], \
            "Heavy fighter should have less stamina"

    def test_save_and_load_config(self):
        """Test configuration can be saved and loaded."""
        import tempfile
        import json
        import os

        config = WorldConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_json(f.name)
            temp_file = f.name

        try:
            # Load it back
            loaded_config = WorldConfig.load_from_json(temp_file)

            # Check key parameters match
            assert loaded_config.hit_cooldown_ticks == config.hit_cooldown_ticks
            assert loaded_config.hit_impact_threshold == config.hit_impact_threshold
            assert loaded_config.hit_recoil_multiplier == config.hit_recoil_multiplier
            assert len(loaded_config.stances) == 3
            assert loaded_config.stances["defending"].drain == 0

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])