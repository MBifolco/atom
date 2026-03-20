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

from ....arena import WorldConfig
from ...gym_env import AtomCombatEnv
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


class PopulationEvolver:
    """Selection + mutation lifecycle for population-based training."""

    def __init__(self, context: EvolutionContext):
        self.context = context

    def evolve(
        self,
        population: List[PopulationFighterProtocol],
        elo_tracker: EloTrackerPopulationProtocol,
        keep_top: float,
        mutation_rate: float,
        create_fighter_name: Callable[[int, int], str],
        fighter_factory: Callable[..., PopulationFighterProtocol],
    ) -> None:
        """
        Evolve the population by replacing lower-ranked fighters with mutated children.
        """
        if self.context.verbose:
            print("\n" + "=" * 60)
            print("POPULATION EVOLUTION")
            print("=" * 60)

        selection = self._select_survivors(population, elo_tracker, keep_top=keep_top)

        if self.context.verbose:
            print(f"  Keeping top {len(selection.survivors)} fighters")
            print(f"  Replacing {len(selection.to_replace)} fighters")

        for old_fighter in selection.to_replace:
            parent = random.choice(selection.survivors)
            population_index = population.index(old_fighter)
            new_name = create_fighter_name(population_index, self.context.generation)
            new_mass = self._sample_child_mass(parent.mass)
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

            if self.context.verbose:
                print(f"    Replaced {old_fighter.name} with {new_name} (child of {parent.name})")

        self.context.logger.info(f"Evolved to generation {self.context.generation}")

    def _select_survivors(
        self,
        population: List[PopulationFighterProtocol],
        elo_tracker: EloTrackerPopulationProtocol,
        keep_top: float,
    ) -> EvolutionSelection:
        """Select survivors based on global ELO ranking restricted to current population."""
        rankings = elo_tracker.get_rankings()
        keep_count = max(2, int(len(population) * keep_top))

        population_names = {fighter.name for fighter in population}
        population_rankings = [
            (i, stats)
            for i, stats in enumerate(rankings)
            if stats.name in population_names
        ]

        population_by_rank = sorted(
            population,
            key=lambda fighter: next(
                (i for i, stats in population_rankings if stats.name == fighter.name),
                999,
            ),
        )

        survivors = population_by_rank[:keep_count]
        to_replace = population_by_rank[keep_count:]
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
