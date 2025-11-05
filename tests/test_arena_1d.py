"""
Tests for Arena1D edge cases.

Tests cover:
- Fighter validation (mass constraints)
- get_winner edge cases (draw, ongoing, individual deaths)
"""

import pytest
from src.arena.arena_1d import Arena1D
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig


class TestFighterValidation:
    """Test fighter validation during arena initialization."""

    def test_valid_mass_fighters_accepted(self):
        """Fighters with valid mass are accepted."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)

        # Should not raise
        arena = Arena1D(fighter_a, fighter_b, config)
        assert arena is not None

    def test_min_mass_fighter_accepted(self):
        """Fighter at minimum mass is accepted."""
        config = WorldConfig()
        fighter_a = FighterState.create("LightFighter", mass=config.min_mass, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)

        # Should not raise
        arena = Arena1D(fighter_a, fighter_b, config)
        assert arena is not None

    def test_max_mass_fighter_accepted(self):
        """Fighter at maximum mass is accepted."""
        config = WorldConfig()
        fighter_a = FighterState.create("HeavyFighter", mass=config.max_mass, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)

        # Should not raise
        arena = Arena1D(fighter_a, fighter_b, config)
        assert arena is not None

    def test_too_light_fighter_rejected(self):
        """Fighter below minimum mass is rejected."""
        config = WorldConfig()

        # Create fighter with invalid mass (bypass the create method)
        fighter_a = FighterState(
            name="TooLight",
            mass=config.min_mass - 1.0,  # Below minimum
            position=2.0,
            velocity=0.0,
            hp=100.0,
            max_hp=100.0,
            stamina=10.0,
            max_stamina=10.0,
            stance="neutral"
        )
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)

        # Should raise ValueError
        with pytest.raises(ValueError, match="mass.*outside legal range"):
            Arena1D(fighter_a, fighter_b, config)

    def test_too_heavy_fighter_rejected(self):
        """Fighter above maximum mass is rejected."""
        config = WorldConfig()

        # Create fighter with invalid mass (bypass the create method)
        fighter_a = FighterState(
            name="TooHeavy",
            mass=config.max_mass + 1.0,  # Above maximum
            position=2.0,
            velocity=0.0,
            hp=100.0,
            max_hp=100.0,
            stamina=10.0,
            max_stamina=10.0,
            stance="neutral"
        )
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)

        # Should raise ValueError
        with pytest.raises(ValueError, match="mass.*outside legal range"):
            Arena1D(fighter_a, fighter_b, config)


class TestGetWinner:
    """Test get_winner method edge cases."""

    def test_get_winner_ongoing_match(self):
        """get_winner returns 'ongoing' when both fighters alive."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Both alive
        assert arena.get_winner() == "ongoing"

    def test_get_winner_fighter_a_wins(self):
        """get_winner returns fighter_a name when fighter_b dies."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Kill fighter_b
        fighter_b.hp = 0.0

        assert arena.get_winner() == "FighterA"

    def test_get_winner_fighter_b_wins(self):
        """get_winner returns fighter_b name when fighter_a dies."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Kill fighter_a
        fighter_a.hp = 0.0

        assert arena.get_winner() == "FighterB"

    def test_get_winner_draw(self):
        """get_winner returns 'draw' when both fighters die simultaneously."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Kill both fighters
        fighter_a.hp = 0.0
        fighter_b.hp = 0.0

        assert arena.get_winner() == "draw"

    def test_get_winner_negative_hp_treated_as_zero(self):
        """get_winner treats negative HP as death."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Give fighter_a negative HP (overkill)
        fighter_a.hp = -5.0

        assert arena.get_winner() == "FighterB"

    def test_is_finished_when_fighter_dies(self):
        """is_finished returns True when a fighter dies."""
        config = WorldConfig()
        fighter_a = FighterState.create("FighterA", mass=70.0, position=2.0, world_config=config)
        fighter_b = FighterState.create("FighterB", mass=70.0, position=10.0, world_config=config)
        arena = Arena1D(fighter_a, fighter_b, config)

        # Initially ongoing
        assert not arena.is_finished()

        # Kill fighter_b
        fighter_b.hp = 0.0

        # Should be finished
        assert arena.is_finished()
