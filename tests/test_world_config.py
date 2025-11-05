"""
Tests for WorldConfig and StanceConfig.

Tests cover:
- StanceConfig dataclass creation
- WorldConfig initialization with defaults
- calculate_fighter_stats for HP and stamina
- to_dict and from_dict conversion
- JSON loading and saving
- YAML loading and saving
"""

import pytest
import json
import yaml
import tempfile
import os
from src.arena.world_config import WorldConfig, StanceConfig, SPECTACLE_CONFIG


class TestStanceConfig:
    """Test StanceConfig dataclass."""

    def test_stance_config_creation(self):
        """StanceConfig can be created with all fields."""
        stance = StanceConfig(reach=0.5, width=0.3, drain=0.1, defense=1.2)

        assert stance.reach == 0.5
        assert stance.width == 0.3
        assert stance.drain == 0.1
        assert stance.defense == 1.2


class TestWorldConfigDefaults:
    """Test WorldConfig default initialization."""

    def test_world_config_has_default_values(self):
        """WorldConfig initializes with default values."""
        config = WorldConfig()

        assert config.arena_width > 0
        assert config.friction > 0
        assert config.max_acceleration > 0
        assert config.max_velocity > 0
        assert config.dt > 0

    def test_world_config_has_default_stances(self):
        """WorldConfig has default stances."""
        config = WorldConfig()

        assert "neutral" in config.stances
        assert "extended" in config.stances
        assert "retracted" in config.stances
        assert "defending" in config.stances

    def test_spectacle_config_is_world_config(self):
        """SPECTACLE_CONFIG is a valid WorldConfig instance."""
        assert isinstance(SPECTACLE_CONFIG, WorldConfig)


class TestCalculateFighterStats:
    """Test calculate_fighter_stats method."""

    def test_calculate_stats_for_min_mass(self):
        """Calculate stats for minimum mass fighter."""
        config = WorldConfig()
        stats = config.calculate_fighter_stats(config.min_mass)

        # Should be close to hp_min and stamina_max
        assert abs(stats["max_hp"] - config.hp_min) < 1.0
        assert abs(stats["max_stamina"] - config.stamina_max) < 1.0

    def test_calculate_stats_for_max_mass(self):
        """Calculate stats for maximum mass fighter."""
        config = WorldConfig()
        stats = config.calculate_fighter_stats(config.max_mass)

        # Should be close to hp_max and stamina_min
        assert abs(stats["max_hp"] - config.hp_max) < 1.0
        assert abs(stats["max_stamina"] - config.stamina_min) < 1.0

    def test_calculate_stats_for_mid_mass(self):
        """Calculate stats for mid-range mass fighter."""
        config = WorldConfig()
        mid_mass = (config.min_mass + config.max_mass) / 2
        stats = config.calculate_fighter_stats(mid_mass)

        # Should be mid-range HP and stamina
        mid_hp = (config.hp_min + config.hp_max) / 2
        mid_stamina = (config.stamina_min + config.stamina_max) / 2

        assert abs(stats["max_hp"] - mid_hp) < 5.0
        assert abs(stats["max_stamina"] - mid_stamina) < 1.0

    def test_heavier_fighter_has_more_hp(self):
        """Heavier fighters have more HP."""
        config = WorldConfig()
        light_stats = config.calculate_fighter_stats(50.0)
        heavy_stats = config.calculate_fighter_stats(80.0)

        assert heavy_stats["max_hp"] > light_stats["max_hp"]

    def test_heavier_fighter_has_less_stamina(self):
        """Heavier fighters have less stamina."""
        config = WorldConfig()
        light_stats = config.calculate_fighter_stats(50.0)
        heavy_stats = config.calculate_fighter_stats(80.0)

        assert heavy_stats["max_stamina"] < light_stats["max_stamina"]


class TestDictConversion:
    """Test to_dict and from_dict methods."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all configuration fields."""
        config = WorldConfig()
        data = config.to_dict()

        assert "arena_width" in data
        assert "friction" in data
        assert "max_acceleration" in data
        assert "max_velocity" in data
        assert "stances" in data

    def test_to_dict_stances_are_dicts(self):
        """to_dict converts stances to nested dictionaries."""
        config = WorldConfig()
        data = config.to_dict()

        assert isinstance(data["stances"], dict)
        assert isinstance(data["stances"]["neutral"], dict)
        assert "reach" in data["stances"]["neutral"]
        assert "width" in data["stances"]["neutral"]
        assert "drain" in data["stances"]["neutral"]
        assert "defense" in data["stances"]["neutral"]

    def test_from_dict_recreates_config(self):
        """from_dict creates equivalent WorldConfig."""
        config = WorldConfig()
        data = config.to_dict()
        restored = WorldConfig.from_dict(data)

        assert restored.arena_width == config.arena_width
        assert restored.friction == config.friction
        assert restored.max_acceleration == config.max_acceleration
        assert "neutral" in restored.stances
        assert "extended" in restored.stances

    def test_from_dict_with_custom_values(self):
        """from_dict works with custom values."""
        data = {
            "arena_width": 15.0,
            "friction": 0.5,
            "max_acceleration": 5.0,
            "max_velocity": 3.0,
            "dt": 0.1,
            "stances": {
                "neutral": {"reach": 0.3, "width": 0.4, "drain": 0.0, "defense": 1.0}
            }
        }
        config = WorldConfig.from_dict(data)

        assert config.arena_width == 15.0
        assert config.friction == 0.5
        assert config.max_acceleration == 5.0
        assert "neutral" in config.stances
        assert config.stances["neutral"].reach == 0.3

    def test_roundtrip_preserves_values(self):
        """to_dict -> from_dict roundtrip preserves values."""
        config = WorldConfig(arena_width=20.0, friction=0.4)
        data = config.to_dict()
        restored = WorldConfig.from_dict(data)

        assert restored.arena_width == config.arena_width
        assert restored.friction == config.friction


class TestJSONSerialization:
    """Test JSON loading and saving."""

    def test_save_to_json_creates_file(self):
        """save_to_json creates a valid JSON file."""
        config = WorldConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_json(filepath)
            assert os.path.exists(filepath)

            # Verify it's valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert "arena_width" in data
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_from_json_reads_file(self):
        """load_from_json reads and parses JSON file."""
        config = WorldConfig(arena_width=25.0, friction=0.6)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_json(filepath)
            loaded = WorldConfig.load_from_json(filepath)

            assert loaded.arena_width == 25.0
            assert loaded.friction == 0.6
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_json_roundtrip_preserves_stances(self):
        """JSON roundtrip preserves stance configurations."""
        config = WorldConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_json(filepath)
            loaded = WorldConfig.load_from_json(filepath)

            assert "neutral" in loaded.stances
            assert "extended" in loaded.stances
            assert loaded.stances["neutral"].reach == config.stances["neutral"].reach
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestYAMLSerialization:
    """Test YAML loading and saving."""

    def test_save_to_yaml_creates_file(self):
        """save_to_yaml creates a valid YAML file."""
        config = WorldConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_yaml(filepath)
            assert os.path.exists(filepath)

            # Verify it's valid YAML
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            assert "arena_width" in data
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_load_from_yaml_reads_file(self):
        """load_from_yaml reads and parses YAML file."""
        config = WorldConfig(arena_width=30.0, friction=0.7)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_yaml(filepath)
            loaded = WorldConfig.load_from_yaml(filepath)

            assert loaded.arena_width == 30.0
            assert loaded.friction == 0.7
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_yaml_roundtrip_preserves_stances(self):
        """YAML roundtrip preserves stance configurations."""
        config = WorldConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            config.save_to_yaml(filepath)
            loaded = WorldConfig.load_from_yaml(filepath)

            assert "defending" in loaded.stances
            assert "retracted" in loaded.stances
            assert loaded.stances["defending"].defense == config.stances["defending"].defense
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
