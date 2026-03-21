"""
Test orchestrator state management bugs.

Tests for bugs fixed:
1. Orchestrator was using stale fighter states (not updating from arena)
2. Missing direction field in snapshot (fighters didn't know which way to move)
"""

import pytest
from src.arena import WorldConfig, FighterState, Arena1DJAXJit
from src.orchestrator import MatchOrchestrator
from src.atom.runtime.protocol.combat_protocol import generate_snapshot


class TestOrchestratorStateUpdates:
    """Test that orchestrator uses current arena state, not stale initial state."""

    def test_snapshot_uses_current_arena_state(self):
        """
        Bug: Orchestrator was generating snapshots from initial fighter states,
        not the updated states from the arena.

        This caused fighters to always see starting HP/stamina/position.
        """
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 3.0, config)
        fighter_b = FighterState.create("B", 70.0, 9.0, config)

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Step the arena (states should change)
        action = {"acceleration": 1.0, "stance": "neutral"}
        arena.step(action, action)

        # Get current state from arena
        current_a = arena.fighter_a

        # Position should have changed
        assert current_a.position != fighter_a.position, \
            "Arena state should update after step"

        # Snapshot should reflect CURRENT state, not initial
        snapshot = generate_snapshot(current_a, arena.fighter_b, 1, config.arena_width)

        assert abs(snapshot["you"]["position"] - current_a.position) < 0.001, \
            "Snapshot should use current position from arena, not initial position"

    def test_direction_field_in_snapshot(self):
        """
        Bug: Snapshot didn't include direction to opponent.

        Fighters only got distance but not which direction to move (left or right).
        This caused all fighters to move in same direction.
        """
        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 3.0, config)  # Left side
        fighter_b = FighterState.create("B", 70.0, 9.0, config)  # Right side

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        # Generate snapshot for fighter A (on left, opponent on right)
        snapshot_a = generate_snapshot(arena.fighter_a, arena.fighter_b, 0, config.arena_width)

        # Should have direction field
        assert "direction" in snapshot_a["opponent"], "Snapshot must include opponent direction"

        # Direction should be +1 (opponent to the right)
        assert snapshot_a["opponent"]["direction"] == 1.0, \
            f"Fighter on left should see opponent to right (direction=1), got {snapshot_a['opponent']['direction']}"

        # Generate snapshot for fighter B (on right, opponent on left)
        snapshot_b = generate_snapshot(arena.fighter_b, arena.fighter_a, 0, config.arena_width)

        # Direction should be -1 (opponent to the left)
        assert snapshot_b["opponent"]["direction"] == -1.0, \
            f"Fighter on right should see opponent to left (direction=-1), got {snapshot_b['opponent']['direction']}"

    def test_fighters_approach_with_direction(self):
        """Test that fighters use direction field to move toward each other."""
        from fighters.examples.slugger import decide

        config = WorldConfig()
        fighter_a = FighterState.create("A", 70.0, 2.0, config)  # Left
        fighter_b = FighterState.create("B", 70.0, 10.0, config)  # Right

        arena = Arena1DJAXJit(fighter_a, fighter_b, config)

        initial_distance = abs(arena.fighter_a.position - arena.fighter_b.position)

        # Run steps
        for _ in range(20):
            snapshot_a = generate_snapshot(arena.fighter_a, arena.fighter_b, 0, config.arena_width)
            snapshot_b = generate_snapshot(arena.fighter_b, arena.fighter_a, 0, config.arena_width)

            action_a = decide(snapshot_a)
            action_b = decide(snapshot_b)

            arena.step(action_a, action_b)

        final_distance = abs(arena.fighter_a.position - arena.fighter_b.position)

        # Aggressive sluggers should close distance
        assert final_distance < initial_distance, \
            f"Aggressive fighters should approach: {initial_distance:.2f} -> {final_distance:.2f}"

    def test_orchestrator_updates_state_each_tick(self):
        """
        Integration test: orchestrator should use current arena state,
        not stale initial state.
        """
        from fighters.examples.boxer import decide as boxer_decide

        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=20, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "Boxer", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "Boxer", "mass": 70.0, "position": 9.0},
            decision_func_a=boxer_decide,
            decision_func_b=boxer_decide,
            seed=42
        )

        # Check telemetry shows changing positions
        ticks = result.telemetry["ticks"]

        if len(ticks) > 2:
            pos_tick_0 = ticks[0]["fighter_a"]["position"]
            pos_tick_10 = ticks[10]["fighter_a"]["position"] if len(ticks) > 10 else ticks[-1]["fighter_a"]["position"]

            assert abs(pos_tick_10 - pos_tick_0) > 0.01, \
                f"Position should change over ticks: {pos_tick_0} -> {pos_tick_10}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
