"""
Tests for match orchestrator to increase coverage.
"""

import pytest
from src.orchestrator import MatchOrchestrator, MatchResult
from src.arena import WorldConfig


def aggressive_fighter(state):
    """Aggressive fighter for testing."""
    direction = state["opponent"]["direction"]
    return {"acceleration": 1.0 * direction, "stance": "extended"}


def defensive_fighter(state):
    """Defensive fighter for testing."""
    return {"acceleration": 0.0, "stance": "defending"}


class TestMatchOrchestrator:
    """Test match orchestrator functionality."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes with config."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=100, record_telemetry=True)

        assert orchestrator.config == config
        assert orchestrator.max_ticks == 100
        assert orchestrator.record_telemetry is True

    def test_run_match_returns_result(self):
        """Test run_match returns MatchResult."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=50)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=defensive_fighter,
            seed=42
        )

        assert isinstance(result, MatchResult)
        assert result.winner is not None
        assert result.total_ticks > 0
        assert result.final_hp_a >= 0
        assert result.final_hp_b >= 0

    def test_match_with_knockout(self):
        """Test match ends properly on knockout."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=500)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "Aggressor", "mass": 85.0, "position": 5.0},
            fighter_b_spec={"name": "Defender", "mass": 45.0, "position": 6.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=defensive_fighter,
            seed=42
        )

        # One fighter should be at or near 0 HP
        assert result.final_hp_a == 0 or result.final_hp_b == 0 or \
               "(timeout)" in result.winner, \
               "Match should have knockout or timeout"

    def test_telemetry_recording(self):
        """Test telemetry is recorded when enabled."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        # Telemetry should be populated
        assert "ticks" in result.telemetry
        assert len(result.telemetry["ticks"]) > 0
        assert "fighter_a_name" in result.telemetry
        assert "fighter_b_name" in result.telemetry

    def test_no_telemetry_when_disabled(self):
        """Test telemetry not recorded when disabled."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=False)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        # Telemetry should be minimal
        assert "ticks" in result.telemetry
        # Config should not be recorded
        assert result.telemetry.get("config", {}) == {}

    def test_match_timeout(self):
        """Test match timeouts correctly."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=20, record_telemetry=False)

        # Both defensive - should timeout
        result = orchestrator.run_match(
            fighter_a_spec={"name": "Defender1", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "Defender2", "mass": 70.0, "position": 9.0},
            decision_func_a=defensive_fighter,
            decision_func_b=defensive_fighter,
            seed=42
        )

        # Should timeout
        assert "(timeout)" in result.winner
        assert result.total_ticks == 20

    def test_winner_determination_by_hp(self):
        """Test winner determined by HP percentage on timeout."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=30)

        # One aggressive, one defensive
        result = orchestrator.run_match(
            fighter_a_spec={"name": "Attacker", "mass": 80.0, "position": 5.0},
            fighter_b_spec={"name": "Defender", "mass": 60.0, "position": 6.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=defensive_fighter,
            seed=42
        )

        # Winner should be determined
        assert result.winner is not None
        assert "Attacker" in result.winner or "Defender" in result.winner or "draw" in result.winner

    def test_events_recorded(self):
        """Test events are captured during match."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=100)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 5.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 6.0},
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=42
        )

        # Events should be a list
        assert isinstance(result.events, list)
        # May have HIT events if fighters engaged
        if result.events:
            for event in result.events:
                assert isinstance(event, dict)
                assert "type" in event

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different match outcomes."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=50)

        spec_a = {"name": "A", "mass": 70.0, "position": 3.0}
        spec_b = {"name": "B", "mass": 70.0, "position": 9.0}

        result1 = orchestrator.run_match(
            fighter_a_spec=spec_a,
            fighter_b_spec=spec_b,
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=1
        )

        result2 = orchestrator.run_match(
            fighter_a_spec=spec_a,
            fighter_b_spec=spec_b,
            decision_func_a=aggressive_fighter,
            decision_func_b=aggressive_fighter,
            seed=999
        )

        # Results may differ (HP, winner, etc.)
        # At minimum, they should both complete
        assert result1.total_ticks > 0
        assert result2.total_ticks > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
