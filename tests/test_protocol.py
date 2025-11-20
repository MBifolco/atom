"""
Tests for combat protocol (Action, Snapshot, ProtocolValidator).
Increase coverage of src/protocol/combat_protocol.py.
"""

import pytest
from src.protocol.combat_protocol import Action, ProtocolValidator, generate_snapshot
from src.arena import WorldConfig, FighterState


class TestAction:
    """Test Action dataclass."""

    def test_action_to_dict(self):
        """Test Action.to_dict() conversion."""
        action = Action(acceleration=2.5, stance="extended")
        d = action.to_dict()

        assert d["acceleration"] == 2.5
        assert d["stance"] == "extended"

    def test_action_from_dict(self):
        """Test Action.from_dict() conversion."""
        d = {"acceleration": -1.5, "stance": "defending"}
        action = Action.from_dict(d)

        assert action.acceleration == -1.5
        assert action.stance == "defending"


class TestProtocolValidator:
    """Test ProtocolValidator for action validation."""

    def test_valid_action_passes(self):
        """Test that valid actions pass validation."""
        validator = ProtocolValidator(
            max_acceleration=4.5,
            valid_stances=["neutral", "extended", "defending"]
        )

        action = Action(acceleration=2.0, stance="extended")
        is_valid, error = validator.validate_action(action)

        assert is_valid, f"Valid action should pass: {error}"
        assert error == ""

    def test_excessive_acceleration_fails(self):
        """Test that excessive acceleration fails validation."""
        validator = ProtocolValidator(
            max_acceleration=4.5,
            valid_stances=["neutral", "extended", "defending"]
        )

        action = Action(acceleration=10.0, stance="neutral")
        is_valid, error = validator.validate_action(action)

        assert not is_valid
        assert "exceeds max" in error.lower()

    def test_invalid_stance_fails(self):
        """Test that invalid stance fails validation."""
        validator = ProtocolValidator(
            max_acceleration=4.5,
            valid_stances=["neutral", "extended", "defending"]
        )

        action = Action(acceleration=2.0, stance="retracted")  # Invalid!
        is_valid, error = validator.validate_action(action)

        assert not is_valid
        assert "invalid stance" in error.lower()

    def test_clamp_action_fixes_acceleration(self):
        """Test that clamp_action fixes excessive acceleration."""
        validator = ProtocolValidator(
            max_acceleration=4.5,
            valid_stances=["neutral", "extended", "defending"]
        )

        # Excessive positive acceleration
        action = Action(acceleration=10.0, stance="neutral")
        clamped = validator.clamp_action(action)

        assert clamped.acceleration == 4.5
        assert clamped.stance == "neutral"

        # Excessive negative acceleration
        action2 = Action(acceleration=-10.0, stance="extended")
        clamped2 = validator.clamp_action(action2)

        assert clamped2.acceleration == -4.5

    def test_clamp_action_fixes_stance(self):
        """Test that clamp_action fixes invalid stance."""
        validator = ProtocolValidator(
            max_acceleration=4.5,
            valid_stances=["neutral", "extended", "defending"]
        )

        action = Action(acceleration=2.0, stance="retracted")  # Invalid!
        clamped = validator.clamp_action(action)

        assert clamped.stance == "neutral"  # Falls back to neutral
        assert clamped.acceleration == 2.0  # Acceleration unchanged


class TestSnapshotGeneration:
    """Test snapshot generation for fighters."""

    def test_snapshot_direction_left(self):
        """Test direction when opponent is to the left."""
        config = WorldConfig()
        # Fighter on right (10.0), opponent on left (2.0)
        fighter = FighterState.create("Me", 70.0, 10.0, config)
        opponent = FighterState.create("Opp", 70.0, 2.0, config)

        snapshot = generate_snapshot(fighter, opponent, 0, config.arena_width)

        # Opponent to the left = direction -1
        assert snapshot["opponent"]["direction"] == -1.0

    def test_snapshot_direction_right(self):
        """Test direction when opponent is to the right."""
        config = WorldConfig()
        # Fighter on left (2.0), opponent on right (10.0)
        fighter = FighterState.create("Me", 70.0, 2.0, config)
        opponent = FighterState.create("Opp", 70.0, 10.0, config)

        snapshot = generate_snapshot(fighter, opponent, 0, config.arena_width)

        # Opponent to the right = direction +1
        assert snapshot["opponent"]["direction"] == 1.0

    def test_snapshot_direction_same_position(self):
        """Test direction when fighters at same position."""
        config = WorldConfig()
        fighter = FighterState.create("Me", 70.0, 5.0, config)
        opponent = FighterState.create("Opp", 70.0, 5.0, config)

        snapshot = generate_snapshot(fighter, opponent, 0, config.arena_width)

        # Same position = direction 0
        assert snapshot["opponent"]["direction"] == 0.0

    def test_snapshot_includes_tick(self):
        """Test that snapshot includes current tick."""
        config = WorldConfig()
        fighter = FighterState.create("Me", 70.0, 5.0, config)
        opponent = FighterState.create("Opp", 70.0, 10.0, config)

        snapshot = generate_snapshot(fighter, opponent, 42, config.arena_width)

        assert snapshot["tick"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
