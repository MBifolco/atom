"""
Atom Combat - Match Orchestrator

Coordinates the match tick loop between fighters and arena.
"""

from typing import Callable, Dict, Any, List
from dataclasses import dataclass

from ..arena import Arena1D, FighterState, WorldConfig
from ..protocol.combat_protocol import generate_snapshot, ProtocolValidator


@dataclass
class MatchResult:
    """Results from a completed match."""
    winner: str  # Fighter name or "draw" or "timeout"
    total_ticks: int
    final_hp_a: float
    final_hp_b: float
    telemetry: Dict[str, Any]  # Full tick-by-tick telemetry
    events: List[Dict]  # All collision/special events


class MatchOrchestrator:
    """
    Orchestrates a complete match between two fighters.

    Responsibilities:
    - Run tick loop
    - Generate snapshots for fighters
    - Call fighter decision functions
    - Validate and pass actions to arena
    - Record telemetry
    - Handle timeouts and finish conditions
    """

    def __init__(self, config: WorldConfig, max_ticks: int = 1000, record_telemetry: bool = True):
        """
        Initialize match orchestrator.

        Args:
            config: WorldConfig instance
            max_ticks: Maximum ticks before timeout
            record_telemetry: Whether to record full telemetry
        """
        self.config = config
        self.max_ticks = max_ticks
        self.record_telemetry = record_telemetry

        # Create protocol validator
        valid_stances = list(config.stances.keys())
        self.validator = ProtocolValidator(config.max_acceleration, valid_stances)

    def run_match(
        self,
        fighter_a_spec: Dict[str, Any],
        fighter_b_spec: Dict[str, Any],
        decision_func_a: Callable,
        decision_func_b: Callable,
        seed: int = 42
    ) -> MatchResult:
        """
        Run a complete match between two fighters.

        Args:
            fighter_a_spec: {"name": str, "mass": float, "position": float}
            fighter_b_spec: {"name": str, "mass": float, "position": float}
            decision_func_a: Decision function for fighter A (snapshot -> action_dict)
            decision_func_b: Decision function for fighter B (snapshot -> action_dict)
            seed: Random seed for reproducibility

        Returns:
            MatchResult with winner, stats, and telemetry
        """
        # Create fighters
        fighter_a = FighterState.create(
            fighter_a_spec["name"],
            fighter_a_spec["mass"],
            fighter_a_spec["position"],
            self.config
        )
        fighter_b = FighterState.create(
            fighter_b_spec["name"],
            fighter_b_spec["mass"],
            fighter_b_spec["position"],
            self.config
        )

        # Create arena
        arena = Arena1D(fighter_a, fighter_b, self.config, seed=seed)

        # Initialize telemetry
        telemetry = {
            "ticks": [],
            "fighter_a_name": fighter_a.name,
            "fighter_b_name": fighter_b.name,
            "config": self.config.to_dict() if self.record_telemetry else {}
        }
        all_events = []

        # Run tick loop
        for tick in range(self.max_ticks):
            # Generate snapshots
            snapshot_a = generate_snapshot(fighter_a, fighter_b, tick, self.config.arena_width)
            snapshot_b = generate_snapshot(fighter_b, fighter_a, tick, self.config.arena_width)

            # Get actions from decision functions
            try:
                action_dict_a = decision_func_a(snapshot_a)
                action_dict_b = decision_func_b(snapshot_b)
            except Exception as e:
                # Fighter crashed - forfeit
                return MatchResult(
                    winner=fighter_b.name if "a" in str(e).lower() else fighter_a.name,
                    total_ticks=tick,
                    final_hp_a=fighter_a.hp,
                    final_hp_b=fighter_b.hp,
                    telemetry=telemetry,
                    events=all_events
                )

            # Validate and clamp actions
            from ..protocol.combat_protocol import Action
            action_a = Action.from_dict(action_dict_a)
            action_b = Action.from_dict(action_dict_b)

            action_a = self.validator.clamp_action(action_a)
            action_b = self.validator.clamp_action(action_b)

            # Execute tick
            events = arena.step(action_a.to_dict(), action_b.to_dict())
            all_events.extend(events)

            # Record telemetry
            if self.record_telemetry:
                telemetry["ticks"].append({
                    "tick": tick,
                    "fighter_a": fighter_a.to_dict(),
                    "fighter_b": fighter_b.to_dict(),
                    "action_a": action_a.to_dict(),
                    "action_b": action_b.to_dict(),
                    "events": events
                })

            # Check for match end
            if arena.is_finished():
                winner = arena.get_winner()
                return MatchResult(
                    winner=winner,
                    total_ticks=tick + 1,
                    final_hp_a=fighter_a.hp,
                    final_hp_b=fighter_b.hp,
                    telemetry=telemetry,
                    events=all_events
                )

        # Timeout - determine winner by HP percentage (not absolute HP)
        hp_pct_a = fighter_a.hp / fighter_a.max_hp
        hp_pct_b = fighter_b.hp / fighter_b.max_hp

        if hp_pct_a > hp_pct_b:
            winner = fighter_a.name
        elif hp_pct_b > hp_pct_a:
            winner = fighter_b.name
        else:
            winner = "draw"

        return MatchResult(
            winner=f"{winner} (timeout)",
            total_ticks=self.max_ticks,
            final_hp_a=fighter_a.hp,
            final_hp_b=fighter_b.hp,
            telemetry=telemetry,
            events=all_events
        )
