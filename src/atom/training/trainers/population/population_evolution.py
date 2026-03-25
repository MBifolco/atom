"""
Population evolution helpers.

This module encapsulates selection, cloning, and mutation mechanics for
population training so PopulationTrainer can coordinate at a higher level.
"""

from __future__ import annotations

import logging
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.atom.runtime.arena import WorldConfig
from src.atom.training.gym_env import AtomCombatEnv
from .population_protocols import EloTrackerPopulationProtocol, PopulationFighterProtocol


@dataclass(frozen=True)
class EvolutionContext:
    """Runtime context required by the population evolution lifecycle."""

    config: WorldConfig
    max_ticks: int
    mass_range: Tuple[float, float]
    generation: int
    algorithm: str
    verbose: bool
    logger: logging.Logger


@dataclass(frozen=True)
class EvolutionSelection:
    """Selected survivors and fighters marked for replacement."""

    survivors: List[PopulationFighterProtocol]
    to_replace: List[PopulationFighterProtocol]


@dataclass(frozen=True)
class LineageEvent:
    """Structured record for one parent-child replacement during evolution."""

    generation: int
    child_name: str
    child_generation: int
    parent_name: str
    replaced_fighter_name: str
    parent_elo_at_mutation: float | None
    child_mass: float
    replaced_generation: int


class PopulationEvolver:
    """Selection + mutation lifecycle for population-based training."""

    def __init__(self, context: EvolutionContext):
        self.context = context

    @staticmethod
    def rank_weights(n: int) -> list[int]:
        """Compute rank-based weights: [n, n-1, ..., 1].

        Top-ranked survivor gets weight n, bottom gets weight 1.
        """
        return list(range(n, 0, -1))

    @staticmethod
    def select_parent(
        survivors: list,
        anchor_scores: dict[str, float] | None = None,
    ) -> Any:
        """Select a parent using rank-weighted probability.

        Survivors are assumed to be sorted best-first (index 0 = top rank).
        Top-ranked fighters get proportionally more offspring, but every
        survivor has a nonzero chance — balancing exploitation and exploration.

        When anchor_scores is provided, weights are blended with anchor
        retention so fighters that forget curriculum skills breed less.
        """
        if anchor_scores is not None:
            n = len(survivors)
            weights = []
            for i, fighter in enumerate(survivors):
                rank_pct = (n - i) / n  # 1.0 for top, ~0 for bottom
                a_score = anchor_scores.get(fighter.name, 0.0)
                weights.append(0.8 * rank_pct + 0.2 * a_score)
        else:
            weights = PopulationEvolver.rank_weights(len(survivors))
        return random.choices(survivors, weights=weights, k=1)[0]

    def evolve(
        self,
        population: List[PopulationFighterProtocol],
        elo_tracker: EloTrackerPopulationProtocol,
        keep_top: float,
        mutation_rate: float,
        create_fighter_name: Callable[[int, int], str],
        fighter_factory: Callable[..., PopulationFighterProtocol],
        anchor_scores: dict[str, float] | None = None,
    ) -> List[LineageEvent]:
        """
        Evolve the population by replacing lower-ranked fighters with mutated children.

        Args:
            anchor_scores: Optional {fighter_name: anchor_score (0-1)} from
                curriculum anchor evaluation. When provided, fighters are ranked
                by a composite of ELO percentile (80%) and anchor score (20%)
                instead of raw ELO alone.
        """
        if self.context.verbose:
            print("\n" + "=" * 60)
            print("POPULATION EVOLUTION")
            print("=" * 60)

        selection = self._select_survivors(
            population, elo_tracker, keep_top=keep_top, anchor_scores=anchor_scores,
        )

        if self.context.verbose:
            print(f"  Keeping top {len(selection.survivors)} fighters")
            print(f"  Replacing {len(selection.to_replace)} fighters")

        rankings = elo_tracker.get_rankings()
        lineage_events: list[LineageEvent] = []
        for old_fighter in selection.to_replace:
            parent = self.select_parent(selection.survivors, anchor_scores=anchor_scores)
            population_index = population.index(old_fighter)
            new_name = create_fighter_name(population_index, self.context.generation)
            new_mass = self._sample_child_mass(parent.mass)
            parent_stats = next((stats for stats in rankings if stats.name == parent.name), None)
            new_model = self._clone_and_mutate_model(
                parent=parent,
                new_mass=new_mass,
                mutation_rate=mutation_rate,
            )

            new_fighter = fighter_factory(
                name=new_name,
                model=new_model,
                generation=self.context.generation,
                lineage=f"{parent.name}→{new_name}",
                mass=float(new_mass),
            )

            population[population_index] = new_fighter
            elo_tracker.remove_fighter(old_fighter.name)
            elo_tracker.add_fighter(new_name)
            lineage_events.append(
                LineageEvent(
                    generation=self.context.generation,
                    child_name=new_name,
                    child_generation=self.context.generation,
                    parent_name=parent.name,
                    replaced_fighter_name=old_fighter.name,
                    parent_elo_at_mutation=float(parent_stats.elo) if parent_stats is not None else None,
                    child_mass=float(new_mass),
                    replaced_generation=int(getattr(old_fighter, "generation", 0)),
                )
            )

            if self.context.verbose:
                print(f"    Replaced {old_fighter.name} with {new_name} (child of {parent.name})")

        self.context.logger.info(f"Evolved to generation {self.context.generation}")
        return lineage_events

    def _select_survivors(
        self,
        population: List[PopulationFighterProtocol],
        elo_tracker: EloTrackerPopulationProtocol,
        keep_top: float,
        anchor_scores: dict[str, float] | None = None,
    ) -> EvolutionSelection:
        """Select survivors based on composite score (ELO percentile + anchor retention).

        When anchor_scores is None, falls back to pure ELO ranking.
        """
        rankings = elo_tracker.get_rankings()
        keep_count = max(2, int(len(population) * keep_top))

        population_names = {fighter.name for fighter in population}
        population_rankings = [
            (i, stats)
            for i, stats in enumerate(rankings)
            if stats.name in population_names
        ]

        if anchor_scores is not None:
            # Composite ranking: 80% ELO percentile + 20% anchor score
            n = len(population_rankings)
            elo_percentile = {}
            for rank_idx, (_, stats) in enumerate(population_rankings):
                elo_percentile[stats.name] = (n - rank_idx) / max(1, n)

            def composite_key(fighter):
                elo_pct = elo_percentile.get(fighter.name, 0.0)
                a_score = anchor_scores.get(fighter.name, 0.0)
                return -(0.8 * elo_pct + 0.2 * a_score)  # negative for ascending sort

            population_sorted = sorted(population, key=composite_key)
        else:
            population_sorted = sorted(
                population,
                key=lambda fighter: next(
                    (i for i, stats in population_rankings if stats.name == fighter.name),
                    999,
                ),
            )

        survivors = population_sorted[:keep_count]
        to_replace = population_sorted[keep_count:]
        return EvolutionSelection(survivors=survivors, to_replace=to_replace)

    def _sample_child_mass(self, parent_mass: float) -> float:
        """Sample a child mass near the parent and clamp to configured mass range."""
        mass_variation = np.random.uniform(-5, 5)
        return float(np.clip(parent_mass + mass_variation, *self.context.mass_range))

    def _create_loading_env(self, fighter_mass: float) -> DummyVecEnv:
        """Create minimal env required for model cloning/loading."""
        return DummyVecEnv([lambda fighter_mass=fighter_mass: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            config=self.context.config,
            max_ticks=self.context.max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=70.0,
        ))])

    def _load_parent_model(self, parent: PopulationFighterProtocol, env: DummyVecEnv) -> Any:
        """Load parent model from checkpoint or temporary save path."""
        model_cls = PPO if self.context.algorithm == "ppo" else SAC

        if parent.last_checkpoint:
            return model_cls.load(parent.last_checkpoint, env=env)

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            parent_path = temp_dir / f"{parent.name}_temp.zip"
            parent.model.save(parent_path)
            return model_cls.load(parent_path, env=env)

    def _apply_mutation(self, model: Any, mutation_rate: float) -> None:
        """Apply lightweight optimizer + parameter mutations."""
        import torch

        model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))
        with torch.no_grad():
            for param in model.policy.parameters():
                noise_scale = mutation_rate * 0.1
                noise = torch.randn_like(param) * noise_scale
                param.data.add_(noise)

    def _clone_and_mutate_model(
        self,
        parent: PopulationFighterProtocol,
        new_mass: float,
        mutation_rate: float,
    ) -> Any:
        """Clone parent model into a new environment and apply mutation."""
        env = self._create_loading_env(new_mass)
        model = self._load_parent_model(parent=parent, env=env)
        self._apply_mutation(model, mutation_rate=mutation_rate)
        return model
