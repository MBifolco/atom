"""
Test fighter behavior and movement.
Ensures fighters receive correct state and produce valid actions.
"""

import pytest
import numpy as np
from src.arena import WorldConfig, FighterState, Arena1DJAXJit


class TestFighterActions:
    """Test that fighters produce valid actions."""

    def test_boxer_produces_valid_actions(self):
        """Test boxer fighter produces valid acceleration and stance."""
        from fighters.examples.boxer import decide

        # Create realistic state
        state = {
            "you": {
                "position": 5.0,
                "velocity": 0.0,
                "hp": 86.0,
                "max_hp": 86.0,
                "stamina": 9.1,
                "max_stamina": 9.1,
                "stance": "neutral"
            },
            "opponent": {
                "distance": 5.0,
                "direction": 1.0,  # Opponent to the right
                "velocity": 0.0,
                "hp": 86.0,
                "max_hp": 86.0,
                "stamina": 9.1,
                "max_stamina": 9.1
            },
            "arena": {
                "width": 12.476
            }
        }

        action = decide(state)

        # Verify action structure
        assert "acceleration" in action, "Action must have acceleration"
        assert "stance" in action, "Action must have stance"

        # Verify acceleration is valid number
        assert isinstance(action["acceleration"], (int, float)), "Acceleration must be numeric"
        assert -5.0 <= action["acceleration"] <= 5.0, f"Acceleration {action['acceleration']} out of valid range"

        # Verify stance is valid
        valid_stances = ["neutral", "extended", "defending"]
        assert action["stance"] in valid_stances, f"Invalid stance: {action['stance']}"

    def test_all_fighters_produce_valid_actions(self):
        """Test all fighter archetypes produce valid actions."""
        fighters = [
            ("boxer", "fighters.examples.boxer"),
            ("slugger", "fighters.examples.slugger"),
            ("counter_puncher", "fighters.examples.counter_puncher"),
            ("swarmer", "fighters.examples.swarmer"),
            ("out_fighter", "fighters.examples.out_fighter")
        ]

        state = {
            "you": {
                "position": 5.0,
                "velocity": 0.0,
                "hp": 86.0,
                "max_hp": 86.0,
                "stamina": 9.1,
                "max_stamina": 9.1,
                "stance": "neutral"
            },
            "opponent": {
                "distance": 3.0,
                "direction": 1.0,  # Opponent to the right
                "velocity": 0.5,
                "hp": 86.0,
                "max_hp": 86.0,
                "stamina": 9.1,
                "max_stamina": 9.1
            },
            "arena": {
                "width": 12.476
            }
        }

        for name, module_path in fighters:
            # Import fighter
            parts = module_path.split(".")
            module = __import__(module_path, fromlist=[parts[-1]])
            decide = module.decide

            # Get action
            action = decide(state)

            # Verify
            assert "acceleration" in action, f"{name}: missing acceleration"
            assert "stance" in action, f"{name}: missing stance"
            assert isinstance(action["acceleration"], (int, float)), f"{name}: acceleration not numeric"
            assert action["stance"] in ["neutral", "extended", "defending"], f"{name}: invalid stance"


class TestFighterMovement:
    """Test that fighters actually move in the arena."""

    def test_fighters_change_position_over_time(self):
        """Verify fighters actually move when arena steps."""
        from fighters.examples.boxer import decide as boxer_decide
        from fighters.examples.slugger import decide as slugger_decide
        from src.atom.runtime.protocol.combat_protocol import generate_snapshot

        config = WorldConfig()
        boxer = FighterState.create("Boxer", 70.0, 3.0, config)
        slugger = FighterState.create("Slugger", 70.0, 9.0, config)

        arena = Arena1DJAXJit(boxer, slugger, config)

        # Record initial positions
        initial_pos_a = arena.fighter_a.position
        initial_pos_b = arena.fighter_b.position

        # Run 10 steps with fighters trying to approach
        for tick in range(10):
            # Generate snapshots using protocol (includes direction field)
            snapshot_a = generate_snapshot(arena.fighter_a, arena.fighter_b, tick, config.arena_width)
            snapshot_b = generate_snapshot(arena.fighter_b, arena.fighter_a, tick, config.arena_width)

            # Get actions
            action_a = boxer_decide(snapshot_a)
            action_b = slugger_decide(snapshot_b)

            # Step arena
            arena.step(action_a, action_b)

        # Verify positions changed
        final_pos_a = arena.fighter_a.position
        final_pos_b = arena.fighter_b.position

        # At least one fighter should have moved
        moved_a = abs(final_pos_a - initial_pos_a) > 0.01
        moved_b = abs(final_pos_b - initial_pos_b) > 0.01

        assert moved_a or moved_b, f"Fighters didn't move! A: {initial_pos_a}->{final_pos_a}, B: {initial_pos_b}->{final_pos_b}"

    def test_aggressive_fighters_approach(self):
        """Test that aggressive fighters move toward each other."""
        from fighters.examples.slugger import decide
        from src.atom.runtime.protocol.combat_protocol import generate_snapshot

        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        initial_distance = abs(arena.fighter_a.position - arena.fighter_b.position)

        # Run with both sluggers (aggressive)
        for tick in range(20):
            # Use proper snapshot generation
            snapshot_a = generate_snapshot(arena.fighter_a, arena.fighter_b, tick, config.arena_width)
            snapshot_b = generate_snapshot(arena.fighter_b, arena.fighter_a, tick, config.arena_width)

            action_a = decide(snapshot_a)
            action_b = decide(snapshot_b)

            arena.step(action_a, action_b)

        final_distance = abs(arena.fighter_a.position - arena.fighter_b.position)

        # Distance should decrease (fighters approach)
        assert final_distance < initial_distance, \
            f"Aggressive fighters should approach each other: {initial_distance} -> {final_distance}"


class TestStateFormat:
    """Test that state is provided in correct format to fighters."""

    def test_protocol_state_format(self):
        """Verify state format matches protocol specification."""
        from src.atom.runtime.protocol.combat_protocol import generate_snapshot
        from src.arena import WorldConfig, FighterState

        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)
        fighter_b = FighterState.create("B", 70.0, 10.0, config)

        # Generate snapshot like orchestrator does
        snapshot = generate_snapshot(fighter_a, fighter_b, 0, config.arena_width)

        # Verify structure
        assert "you" in snapshot
        assert "opponent" in snapshot
        assert "arena" in snapshot

        # Verify "you" fields
        you_fields = ["position", "velocity", "hp", "max_hp", "stamina", "max_stamina", "stance"]
        for field in you_fields:
            assert field in snapshot["you"], f"Missing field in 'you': {field}"

        # Verify "opponent" fields
        opp_fields = ["distance", "direction", "velocity", "hp", "max_hp", "stamina", "max_stamina"]
        for field in opp_fields:
            assert field in snapshot["opponent"], f"Missing field in 'opponent': {field}"

        # Verify "arena" fields
        assert "width" in snapshot["arena"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
