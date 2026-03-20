"""
Population evaluation helpers.

This module encapsulates evaluation match execution and ELO updates so
PopulationTrainer can remain focused on orchestration.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from src.atom.runtime.arena import WorldConfig
from .population_protocols import EloTrackerEvaluationProtocol, PopulationFighterProtocol


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
    ) -> int:
        """
        Run evaluation matches for all unique pairs.

        Returns:
            Number of matches run.
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
            return 0

        random.shuffle(pairs)
        matches_run = 0

        for fighter_a, fighter_b in pairs:
            wins_a = 0
            wins_b = 0
            total_damage_a = 0
            total_damage_b = 0

            for _ in range(num_matches_per_pair):
                env = env_factory(
                    opponent_decision_func=decision_func_factory(fighter_b),
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

                total_damage_a += info.get("episode_damage_dealt", 0)
                total_damage_b += info.get("episode_damage_taken", 0)
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

        return matches_run
