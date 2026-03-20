"""
Population training loop helpers.

This module encapsulates generation-level loop orchestration utilities used by
PopulationTrainer.train().
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PopulationTrainingLoopContext:
    """Static configuration used by generation-level training loop helpers."""

    population_size: int
    generations: int
    episodes_per_generation: int
    evolution_frequency: int
    keep_top: float
    mutation_rate: float
    replay_recording_frequency: int
    replay_matches_per_pair: int
    verbose: bool
    logger: logging.Logger


class PopulationTrainingLoopHelper:
    """Utilities for trainer generation-loop orchestration and reporting."""

    def __init__(self, context: PopulationTrainingLoopContext):
        self.context = context

    def print_start_banner(self, base_model_path: Optional[str]) -> None:
        """Print and log training start header."""
        if self.context.verbose:
            print("\n" + "=" * 80)
            print("STARTING POPULATION TRAINING")
            print("=" * 80)
            print(f"Population Size: {self.context.population_size}")
            print(f"Generations: {self.context.generations}")
            print(f"Episodes per Generation: {self.context.episodes_per_generation}")
            print(f"Evolution Frequency: Every {self.context.evolution_frequency} generations")
            if base_model_path:
                print(f"Base Model: {base_model_path}")
            print("=" * 80)

    def log_generation_header(self, current_generation: int) -> None:
        """Emit generation header to stdout and log file."""
        if self.context.verbose:
            print(f"\n{'=' * 80}")
            print(f"GENERATION {current_generation}/{self.context.generations}")
            print(f"{'=' * 80}")
        self.context.logger.info(f"{'=' * 80}")
        self.context.logger.info(f"GENERATION {current_generation}/{self.context.generations}")
        self.context.logger.info(f"{'=' * 80}")

    @staticmethod
    def build_fighter_opponent_pairs(
        population: List[Any],
        pairs: List[Tuple[Any, Any]],
    ) -> List[Tuple[Any, List[Any]]]:
        """Build per-fighter opponent batches from matchmaking pairs."""
        fighter_opponent_pairs = []
        for fighter in population:
            opponents = [
                opp
                for pair in pairs
                for opp in [pair[0], pair[1]]
                if pair[0] == fighter or pair[1] == fighter
                if opp != fighter
            ]

            if not opponents:
                opponents = [candidate for candidate in population if candidate != fighter]
                opponents = random.sample(opponents, min(3, len(opponents)))

            fighter_opponent_pairs.append((fighter, opponents))

        return fighter_opponent_pairs

    def log_generation_training_start(self, population_size: int) -> None:
        """Emit generation training-start progress lines."""
        if self.context.verbose:
            print(f"\n🚀 Training {population_size} fighters in parallel...")
        self.context.logger.info(f"Training {population_size} fighters in parallel")

    def log_generation_training_summary(self, results: List[dict]) -> None:
        """Emit generation training summary from parallel training results."""
        if self.context.verbose and results:
            mean_reward_overall = np.mean([r["mean_reward"] for r in results])
            total_episodes = sum(r["episodes"] for r in results)
            print(
                f"  ✓ Completed {total_episodes} episodes total, "
                f"mean reward: {mean_reward_overall:.1f}"
            )
        if results:
            mean_reward_overall = np.mean([r["mean_reward"] for r in results])
            total_episodes = sum(r["episodes"] for r in results)
            self.context.logger.info(
                f"Generation training complete: {total_episodes} episodes total, "
                f"mean reward: {mean_reward_overall:.1f}"
            )

    def maybe_record_replays(
        self,
        replay_recorder: Any,
        population: List[Any],
        generation_zero_based: int,
        trainer_generation: int,
    ) -> None:
        """Record generation replays when replay recording is enabled."""
        if replay_recorder and (generation_zero_based + 1) % self.context.replay_recording_frequency == 0:
            try:
                replay_recorder.record_population_generation(
                    generation=trainer_generation + 1,
                    fighters=population,
                    num_matches_per_pair=self.context.replay_matches_per_pair,
                )
            except Exception as e:
                self.context.logger.warning(
                    f"Failed to record replays for generation {trainer_generation + 1}: {e}"
                )

    def maybe_show_leaderboard(self, elo_tracker: Any, population: List[Any]) -> None:
        """Show active leaderboard in verbose mode."""
        if self.context.verbose:
            active_names = [fighter.name for fighter in population]
            elo_tracker.print_leaderboard(active_only=active_names)

    def should_evolve(self, generation_zero_based: int) -> bool:
        """Return whether evolution should run this generation."""
        return (
            (generation_zero_based + 1) % self.context.evolution_frequency == 0
            and generation_zero_based < self.context.generations - 1
        )

    def print_and_log_final_report(
        self,
        generation: int,
        total_matches: int,
        elo_tracker: Any,
        replay_recorder: Any,
    ) -> None:
        """Emit final completion report and persist replay index."""
        if self.context.verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"Total Generations: {generation}")
            print(f"Total Matches: {total_matches}")
            print("\nFinal Rankings (All Time):")
            elo_tracker.print_leaderboard()

            metrics = elo_tracker.get_diversity_metrics()
            print("\nPopulation Diversity:")
            print(f"  ELO Spread: {metrics['elo_range']:.0f}")
            print(f"  ELO Std Dev: {metrics['elo_std']:.1f}")
            if "win_rate_std" in metrics:
                print(f"  Win Rate Variance: {metrics['win_rate_std']:.3f}")

        self.context.logger.info("=" * 80)
        self.context.logger.info("TRAINING COMPLETE")
        self.context.logger.info("=" * 80)
        self.context.logger.info(f"Total Generations: {generation}")
        self.context.logger.info(f"Total Matches: {total_matches}")

        metrics = elo_tracker.get_diversity_metrics()
        self.context.logger.info("Population Diversity:")
        self.context.logger.info(f"  ELO Spread: {metrics['elo_range']:.0f}")
        self.context.logger.info(f"  ELO Std Dev: {metrics['elo_std']:.1f}")
        if "win_rate_std" in metrics:
            self.context.logger.info(f"  Win Rate Variance: {metrics['win_rate_std']:.3f}")

        if replay_recorder:
            replay_recorder.save_replay_index()

        self.context.logger.info("Training complete")
        self.context.logger.info(f"Final generation: {generation}")
