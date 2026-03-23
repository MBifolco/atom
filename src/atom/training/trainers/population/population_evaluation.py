"""
Population evaluation helpers.

This module encapsulates evaluation match execution and ELO updates so
PopulationTrainer can remain focused on orchestration.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from src.atom.runtime.arena import WorldConfig
from .population_protocols import EloTrackerEvaluationProtocol, PopulationFighterProtocol


# ---------------------------------------------------------------------------
# Data structures for enriched evaluation results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerFighterEvaluationStats:
    """Aggregated per-fighter stats from an evaluation round-robin.

    Accumulated across all matches where this fighter participates (as either
    the controlled agent or the opponent).
    """

    name: str
    total_stance_ticks: Dict[str, int]   # {neutral: N, extended: N, defending: N}
    total_ticks: int                      # sum of episode lengths
    match_count: int                      # number of evaluation episodes
    total_damage_dealt: float
    total_damage_taken: float


@dataclass(frozen=True)
class EvaluationRunResult:
    """Result of a full evaluation round-robin."""

    matches_run: int
    per_fighter_stats: Dict[str, PerFighterEvaluationStats]


# ---------------------------------------------------------------------------
# Stance counting wrapper for opponent decision functions
# ---------------------------------------------------------------------------

_STANCE_NAMES = ["neutral", "extended", "defending"]


class StanceCountingWrapper:
    """Wraps an opponent decision function to count stance choices per tick.

    This allows collecting behavioral data for the opponent side of a match
    without running mirrored matches.
    """

    def __init__(self, decide_func: Callable):
        self.decide = decide_func
        self.stance_counts: Dict[str, int] = {"neutral": 0, "extended": 0, "defending": 0}

    def __call__(self, snapshot: Any) -> dict:
        action = self.decide(snapshot)
        stance = action.get("stance", "neutral")

        # Handle int / np.integer stances
        if isinstance(stance, (int, np.integer)):
            idx = max(0, min(int(stance), 2))
            stance = _STANCE_NAMES[idx]

        # Default unrecognized stances to neutral
        if stance not in self.stance_counts:
            stance = "neutral"

        self.stance_counts[stance] += 1
        return action


# ---------------------------------------------------------------------------
# Evaluation context and service
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationContext:
    """Runtime context required for population evaluation."""

    config: WorldConfig
    max_ticks: int
    generation: int
    verbose: bool
    logger: logging.Logger


class PopulationEvaluationService:
    """Runs evaluation matches and applies ELO updates."""

    def __init__(self, context: EvaluationContext):
        self.context = context

    def run(
        self,
        population: List[PopulationFighterProtocol],
        elo_tracker: EloTrackerEvaluationProtocol,
        decision_func_factory: Callable[[PopulationFighterProtocol], Callable],
        env_factory: Callable[..., Any],
        num_matches_per_pair: int = 3,
    ) -> EvaluationRunResult:
        """
        Run evaluation matches for all unique pairs.

        Returns:
            EvaluationRunResult with match count and per-fighter stats.
        """
        self.context.logger.info(f"Starting evaluation matches with {len(population)} fighters")

        if self.context.verbose:
            print("\n" + "=" * 60)
            print("EVALUATION MATCHES")
            print("=" * 60)
            print(f"Running matches between {len(population)} fighters")

        pairs: List[Tuple[PopulationFighterProtocol, PopulationFighterProtocol]] = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                pairs.append((population[i], population[j]))

        self.context.logger.info(f"Created {len(pairs)} unique matchups")

        if len(pairs) == 0:
            self.context.logger.error("No pairs created for evaluation! Population may be corrupted.")
            if self.context.verbose:
                print("ERROR: No evaluation pairs created!")
            return EvaluationRunResult(matches_run=0, per_fighter_stats={})

        random.shuffle(pairs)
        matches_run = 0

        # Per-fighter accumulators
        stance_accum: Dict[str, Dict[str, int]] = defaultdict(lambda: {"neutral": 0, "extended": 0, "defending": 0})
        ticks_accum: Dict[str, int] = defaultdict(int)
        match_count_accum: Dict[str, int] = defaultdict(int)
        damage_dealt_accum: Dict[str, float] = defaultdict(float)
        damage_taken_accum: Dict[str, float] = defaultdict(float)

        for fighter_a, fighter_b in pairs:
            wins_a = 0
            wins_b = 0
            total_damage_a = 0
            total_damage_b = 0

            for _ in range(num_matches_per_pair):
                # Wrap opponent to count stance choices
                opponent_wrapper = StanceCountingWrapper(decision_func_factory(fighter_b))

                env = env_factory(
                    opponent_decision_func=opponent_wrapper,
                    config=self.context.config,
                    max_ticks=self.context.max_ticks,
                    fighter_mass=fighter_a.mass,
                    opponent_mass=fighter_b.mass,
                )

                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = fighter_a.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                outcome = info.get("won")
                if outcome is True:
                    wins_a += 1
                elif outcome is False:
                    wins_b += 1
                else:
                    fighter_hp = float(info.get("fighter_hp", 0.0))
                    opponent_hp = float(info.get("opponent_hp", 0.0))
                    if fighter_hp > opponent_hp:
                        wins_a += 1
                    elif opponent_hp > fighter_hp:
                        wins_b += 1

                a_dealt = float(info.get("episode_damage_dealt", 0))
                a_taken = float(info.get("episode_damage_taken", 0))
                total_damage_a += a_dealt
                total_damage_b += a_taken

                fight_ticks = int(info.get("tick", self.context.max_ticks))

                # Fighter A stats (from env info)
                a_stance = info.get("stance_distribution") or {}
                for stance_name in ("neutral", "extended", "defending"):
                    stance_accum[fighter_a.name][stance_name] += a_stance.get(stance_name, 0)
                ticks_accum[fighter_a.name] += fight_ticks
                match_count_accum[fighter_a.name] += 1
                damage_dealt_accum[fighter_a.name] += a_dealt
                damage_taken_accum[fighter_a.name] += a_taken

                # Fighter B stats (derived from A's data + stance wrapper)
                for stance_name in ("neutral", "extended", "defending"):
                    stance_accum[fighter_b.name][stance_name] += opponent_wrapper.stance_counts.get(stance_name, 0)
                ticks_accum[fighter_b.name] += fight_ticks
                match_count_accum[fighter_b.name] += 1
                damage_dealt_accum[fighter_b.name] += a_taken  # B dealt = A taken
                damage_taken_accum[fighter_b.name] += a_dealt  # B taken = A dealt

                env.close()

            if wins_a > wins_b:
                result = "a_wins"
            elif wins_b > wins_a:
                result = "b_wins"
            else:
                result = "draw"

            new_elo_a, new_elo_b = elo_tracker.update_ratings(
                fighter_a.name,
                fighter_b.name,
                result,
                total_damage_a / num_matches_per_pair,
                total_damage_b / num_matches_per_pair,
                {"generation": self.context.generation},
            )

            if self.context.verbose:
                result_str = (
                    f"{fighter_a.name} wins"
                    if result == "a_wins"
                    else (f"{fighter_b.name} wins" if result == "b_wins" else "Draw")
                )
                print(f"  {fighter_a.name} ({new_elo_a:.0f}) vs {fighter_b.name} ({new_elo_b:.0f}): {result_str}")

            matches_run += num_matches_per_pair

        # Build per-fighter stats
        per_fighter_stats: Dict[str, PerFighterEvaluationStats] = {}
        all_names = set(stance_accum.keys()) | set(ticks_accum.keys())
        for name in all_names:
            per_fighter_stats[name] = PerFighterEvaluationStats(
                name=name,
                total_stance_ticks=dict(stance_accum[name]),
                total_ticks=ticks_accum[name],
                match_count=match_count_accum[name],
                total_damage_dealt=damage_dealt_accum[name],
                total_damage_taken=damage_taken_accum[name],
            )

        return EvaluationRunResult(
            matches_run=matches_run,
            per_fighter_stats=per_fighter_stats,
        )
