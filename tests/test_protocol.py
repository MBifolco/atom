"""
Tests for the combat protocol module.

Tests cover:
- Action dataclass creation and serialization
- ProtocolValidator validation and clamping
- Snapshot generation with correct relative velocity
"""

from src.protocol.combat_protocol import Action, ProtocolValidator, generate_snapshot
from src.arena.fighter import FighterState
from src.arena.world_config import WorldConfig


class TestAction:
    """Test Action dataclass serialization."""

    def test_action_to_dict(self):
        """Action can be converted to dictionary."""
        action = Action(acceleration=2.5, stance="extended")
        result = action.to_dict()

        assert result["acceleration"] == 2.5
        assert result["stance"] == "extended"

    def test_action_from_dict(self):
        """Action can be created from dictionary."""
        data = {"acceleration": -1.5, "stance": "defending"}
        action = Action.from_dict(data)

        assert action.acceleration == -1.5
        assert action.stance == "defending"

    def test_action_roundtrip(self):
        """Action survives to_dict/from_dict roundtrip."""
        original = Action(acceleration=3.0, stance="retracted")
        data = original.to_dict()
        restored = Action.from_dict(data)

        assert restored.acceleration == original.acceleration
        assert restored.stance == original.stance


class TestProtocolValidator:
    """Test ProtocolValidator for action validation and clamping."""

    def setup_method(self):
        """Create validator for each test."""
        self.validator = ProtocolValidator(
            max_acceleration=4.0,
            valid_stances=["neutral", "extended", "retracted", "defending"]
        )

    def test_valid_action_passes(self):
        """Valid action passes validation."""
        action = Action(acceleration=2.0, stance="extended")
        is_valid, error = self.validator.validate_action(action)

        assert is_valid
        assert error == ""

    def test_acceleration_too_high_fails(self):
        """Acceleration above max fails validation."""
        action = Action(acceleration=5.0, stance="neutral")
        is_valid, error = self.validator.validate_action(action)

        assert not is_valid
        assert "exceeds max" in error

    def test_acceleration_too_low_fails(self):
        """Acceleration below -max fails validation."""
        action = Action(acceleration=-6.0, stance="neutral")
        is_valid, error = self.validator.validate_action(action)

        assert not is_valid
        assert "exceeds max" in error

    def test_invalid_stance_fails(self):
        """Invalid stance fails validation."""
        action = Action(acceleration=2.0, stance="invalid_stance")
        is_valid, error = self.validator.validate_action(action)

        assert not is_valid
        assert "Invalid stance" in error

    def test_clamp_high_acceleration(self):
        """Clamping reduces too-high acceleration to max."""
        action = Action(acceleration=10.0, stance="neutral")
        clamped = self.validator.clamp_action(action)

        assert clamped.acceleration == 4.0
        assert clamped.stance == "neutral"

    def test_clamp_low_acceleration(self):
        """Clamping increases too-low acceleration to -max."""
        action = Action(acceleration=-10.0, stance="neutral")
        clamped = self.validator.clamp_action(action)

        assert clamped.acceleration == -4.0
        assert clamped.stance == "neutral"

    def test_clamp_invalid_stance(self):
        """Clamping converts invalid stance to neutral."""
        action = Action(acceleration=2.0, stance="bad_stance")
        clamped = self.validator.clamp_action(action)

        assert clamped.acceleration == 2.0
        assert clamped.stance == "neutral"

    def test_clamp_valid_action_unchanged(self):
        """Clamping valid action leaves it unchanged."""
        action = Action(acceleration=2.0, stance="extended")
        clamped = self.validator.clamp_action(action)

        assert clamped.acceleration == 2.0
        assert clamped.stance == "extended"


class TestSnapshotGeneration:
    """Test snapshot generation with correct relative positioning."""

    def setup_method(self):
        """Create fighters and config for each test."""
        self.config = WorldConfig()
        self.fighter_a = FighterState.create("FighterA", 70.0, 2.0, self.config)
        self.fighter_b = FighterState.create("FighterB", 75.0, 10.0, self.config)

    def test_snapshot_contains_all_required_fields(self):
        """Snapshot contains tick, you, opponent, and arena."""
        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=5, arena_width=12.5
        )

        assert "tick" in snapshot
        assert "you" in snapshot
        assert "opponent" in snapshot
        assert "arena" in snapshot
        assert snapshot["tick"] == 5

    def test_snapshot_you_fields(self):
        """Snapshot 'you' section contains fighter's own state."""
        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        you = snapshot["you"]
        assert you["position"] == self.fighter_a.position
        assert you["velocity"] == self.fighter_a.velocity
        assert you["hp"] == self.fighter_a.hp
        assert you["max_hp"] == self.fighter_a.max_hp
        assert you["stamina"] == self.fighter_a.stamina
        assert you["max_stamina"] == self.fighter_a.max_stamina
        assert you["stance"] == self.fighter_a.stance

    def test_snapshot_opponent_distance(self):
        """Snapshot opponent distance is absolute distance between fighters."""
        # Fighter A at 2.0, Fighter B at 10.0 -> distance should be 8.0
        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        expected_distance = abs(10.0 - 2.0)
        assert snapshot["opponent"]["distance"] == expected_distance

    def test_snapshot_relative_velocity_approaching(self):
        """Relative velocity is negative when fighters are approaching."""
        # Fighter A at left (2.0) moving right (+1.0 m/s)
        # Fighter B at right (10.0) moving left (-1.0 m/s)
        # They're approaching, so relative velocity should be negative
        self.fighter_a.velocity = 1.0
        self.fighter_b.velocity = -1.0

        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        # From A's perspective: B is moving left at -1, A is moving right at +1
        # Relative velocity = B_vel - A_vel = -1 - 1 = -2 (approaching)
        assert snapshot["opponent"]["velocity"] == -2.0

    def test_snapshot_relative_velocity_separating(self):
        """Relative velocity is positive when fighters are separating."""
        # Fighter A at left (2.0) moving left (-1.0 m/s)
        # Fighter B at right (10.0) moving right (+1.0 m/s)
        # They're separating, so relative velocity should be positive
        self.fighter_a.velocity = -1.0
        self.fighter_b.velocity = 1.0

        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        # From A's perspective: B is moving right at +1, A is moving left at -1
        # Relative velocity = B_vel - A_vel = 1 - (-1) = 2 (separating)
        assert snapshot["opponent"]["velocity"] == 2.0

    def test_snapshot_relative_velocity_reversed_positions(self):
        """Relative velocity calculation works when fighter positions are reversed."""
        # Fighter A at right (10.0), Fighter B at left (2.0)
        # Swap their positions
        self.fighter_a.position = 10.0
        self.fighter_b.position = 2.0

        self.fighter_a.velocity = 1.0
        self.fighter_b.velocity = -1.0

        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        # From A's perspective (A on right, B on left):
        # Relative velocity = A_vel - B_vel = 1 - (-1) = 2 (separating)
        assert snapshot["opponent"]["velocity"] == 2.0

    def test_snapshot_arena_width(self):
        """Snapshot includes correct arena width."""
        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=15.0
        )

        assert snapshot["arena"]["width"] == 15.0

    def test_snapshot_opponent_includes_all_stats(self):
        """Snapshot opponent section includes all fighter stats."""
        snapshot = generate_snapshot(
            self.fighter_a, self.fighter_b, tick=0, arena_width=12.5
        )

        opponent = snapshot["opponent"]
        assert "distance" in opponent
        assert "velocity" in opponent
        assert "hp" in opponent
        assert "max_hp" in opponent
        assert "stamina" in opponent
        assert "max_stamina" in opponent
        assert "stance_hint" in opponent

        # Verify opponent stats match fighter B
        assert opponent["hp"] == self.fighter_b.hp
        assert opponent["max_hp"] == self.fighter_b.max_hp
        assert opponent["stamina"] == self.fighter_b.stamina
        assert opponent["max_stamina"] == self.fighter_b.max_stamina
        assert opponent["stance_hint"] == self.fighter_b.stance
