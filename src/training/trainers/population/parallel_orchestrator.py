"""
Parallel orchestration helpers for population fighter training.

This module isolates process-pool task orchestration from PopulationTrainer so
trainer logic can stay focused on population lifecycle decisions.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ....arena import WorldConfig
from ...gym_env import AtomCombatEnv
from .population_protocols import PopulationFighterProtocol


TrainingTask = Tuple[
    str,  # fighter_name
    float,  # fighter_mass
    str,  # model_path
    List[Tuple[str, float, str]],  # opponent_data
    int,  # n_envs_per_fighter
    int,  # episodes
    int,  # max_ticks
    str,  # algorithm
    Optional[dict],  # config_dict
    str,  # logs_dir
    bool,  # use_vmap
    int,  # n_vmap_envs
]


@dataclass(frozen=True)
class ParallelTrainingContext:
    """Runtime context required for parallel fighter orchestration."""

    models_dir: Path
    logs_dir: Path
    config: WorldConfig
    max_ticks: int
    algorithm: str
    n_envs_per_fighter: int
    n_parallel_fighters: int
    use_vmap: bool
    n_vmap_envs: int
    generation: int
    verbose: bool
    logger: logging.Logger


class TrainingWorker:
    """Pluggable worker strategy used by process-pool tasks."""

    def __init__(self, train_fn: Callable[..., Dict[str, Any]]):
        self._train_fn = train_fn

    def run(self, task: TrainingTask) -> Dict[str, Any]:
        return self._train_fn(*task)


class ModelArtifactStore:
    """Handles temp model save/load lifecycle for parallel training."""

    def __init__(self, context: ParallelTrainingContext):
        self.context = context
        self.temp_model_paths: Dict[str, Path] = {}

    def save_fighter_model(self, fighter: PopulationFighterProtocol) -> Path:
        """Save fighter model to a temp artifact path."""
        path = self.context.models_dir / f"temp_{fighter.name}_{self.context.generation}.zip"
        fighter.model.save(path)
        self.temp_model_paths[fighter.name] = path
        return path

    def ensure_opponent_model(
        self,
        opponent: PopulationFighterProtocol,
    ) -> Path:
        """Save opponent model once and reuse path across fighter tasks."""
        existing = self.temp_model_paths.get(opponent.name)
        if existing is not None:
            return existing

        path = self.context.models_dir / f"temp_{opponent.name}_{self.context.generation}.zip"
        opponent.model.save(path)
        self.temp_model_paths[opponent.name] = path
        return path

    def _create_loading_env(self, fighter_mass: float) -> DummyVecEnv:
        """Create minimal env required by SB3 load path."""
        return DummyVecEnv([lambda fighter_mass=fighter_mass: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            config=self.context.config,
            max_ticks=self.context.max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=70.0,
        ))])

    def reload_updated_models(
        self,
        fighter_opponent_pairs: List[Tuple[PopulationFighterProtocol, List[PopulationFighterProtocol]]],
        episodes_per_fighter: int,
    ) -> None:
        """Reload trained models back into fighter instances."""
        for fighter, _ in fighter_opponent_pairs:
            temp_path = self.temp_model_paths.get(fighter.name)
            if temp_path is None or not temp_path.exists():
                continue

            env = self._create_loading_env(fighter.mass)
            try:
                if self.context.algorithm == "ppo":
                    fighter.model = PPO.load(temp_path, env=env)
                else:
                    fighter.model = SAC.load(temp_path, env=env)
            finally:
                env.close()

            fighter.training_episodes += episodes_per_fighter

    def cleanup(self) -> None:
        """Best-effort cleanup of temp model artifacts."""
        for temp_path in self.temp_model_paths.values():
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


class ParallelTrainingOrchestrator:
    """Coordinates task preparation, process execution, and result handling."""

    def __init__(self, context: ParallelTrainingContext):
        self.context = context

    def run(
        self,
        fighter_opponent_pairs: List[Tuple[PopulationFighterProtocol, List[PopulationFighterProtocol]]],
        episodes_per_fighter: int,
        worker: TrainingWorker,
        executor_factory: Callable[..., Any],
    ) -> List[Dict[str, Any]]:
        """Run parallel fighter training end-to-end."""
        if not fighter_opponent_pairs:
            return []

        artifacts = ModelArtifactStore(self.context)
        training_tasks = self._build_training_tasks(
            fighter_opponent_pairs=fighter_opponent_pairs,
            episodes_per_fighter=episodes_per_fighter,
            artifacts=artifacts,
        )

        self._print_and_log_training_start(training_tasks, episodes_per_fighter)
        training_start_time = time.time()

        results: List[Dict[str, Any]] = []
        cleanup_temp_paths_before_reload = True
        executor = None
        try:
            mp_context = mp.get_context("spawn")
            with executor_factory(
                max_workers=self.context.n_parallel_fighters,
                mp_context=mp_context,
            ) as executor:
                results = self._execute_training_tasks(
                    training_tasks=training_tasks,
                    training_start_time=training_start_time,
                    worker=worker,
                    executor=executor,
                )
                cleanup_temp_paths_before_reload = False
        except KeyboardInterrupt:
            if self.context.verbose:
                print("\n\n⚠️  Training interrupted by user (Ctrl+C). Cleaning up worker processes...")
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if self.context.verbose:
                print("   Worker processes terminated.")
            raise
        finally:
            if cleanup_temp_paths_before_reload:
                artifacts.cleanup()

        artifacts.reload_updated_models(
            fighter_opponent_pairs=fighter_opponent_pairs,
            episodes_per_fighter=episodes_per_fighter,
        )

        total_training_time = time.time() - training_start_time
        self._print_and_log_summary(results, total_training_time)

        artifacts.cleanup()
        return results

    def _build_training_tasks(
        self,
        fighter_opponent_pairs: List[Tuple[PopulationFighterProtocol, List[PopulationFighterProtocol]]],
        episodes_per_fighter: int,
        artifacts: ModelArtifactStore,
    ) -> List[TrainingTask]:
        """Build serializable training tasks for process pool workers."""
        training_tasks: List[TrainingTask] = []
        for fighter, opponents in fighter_opponent_pairs:
            temp_model_path = artifacts.save_fighter_model(fighter)

            opponent_data: List[Tuple[str, float, str]] = []
            for opp in opponents[:self.context.n_envs_per_fighter]:
                opp_path = artifacts.ensure_opponent_model(opp)
                opponent_data.append((opp.name, opp.mass, str(opp_path)))

            config_dict = None  # Use WorldConfig defaults in subprocess for stability
            task: TrainingTask = (
                fighter.name,
                fighter.mass,
                str(temp_model_path),
                opponent_data,
                self.context.n_envs_per_fighter,
                episodes_per_fighter,
                self.context.max_ticks,
                self.context.algorithm,
                config_dict,
                str(self.context.logs_dir),
                self.context.use_vmap,
                self.context.n_vmap_envs,
            )
            training_tasks.append(task)

        return training_tasks

    def _print_and_log_training_start(
        self,
        training_tasks: List[TrainingTask],
        episodes_per_fighter: int,
    ) -> None:
        """Emit kickoff summary for parallel training."""
        if self.context.verbose:
            print(
                f"  Training {len(training_tasks)} fighters total, "
                f"{self.context.n_parallel_fighters} at a time..."
            )
            print(f"  Episodes per fighter: {episodes_per_fighter}")
            if len(training_tasks) > self.context.n_parallel_fighters:
                batches = (
                    len(training_tasks) + self.context.n_parallel_fighters - 1
                ) // self.context.n_parallel_fighters
                print(f"  Will train in {batches} batches of {self.context.n_parallel_fighters}")
            print("  Note: PPO alternates between episode collection (physics) and NN training")
            print()

        self.context.logger.info(
            f"Training {len(training_tasks)} fighters, "
            f"{self.context.n_parallel_fighters} at a time"
        )
        self.context.logger.info(f"Episodes per fighter: {episodes_per_fighter}")
        if len(training_tasks) > self.context.n_parallel_fighters:
            batches = (
                len(training_tasks) + self.context.n_parallel_fighters - 1
            ) // self.context.n_parallel_fighters
            self.context.logger.info(f"Training in {batches} batches of {self.context.n_parallel_fighters}")

    def _execute_training_tasks(
        self,
        training_tasks: List[TrainingTask],
        training_start_time: float,
        worker: TrainingWorker,
        executor: Any,
    ) -> List[Dict[str, Any]]:
        """Schedule and collect parallel training futures with bounded concurrency."""
        results: List[Dict[str, Any]] = []
        future_to_fighter: Dict[Any, str] = {}
        future_to_start_time: Dict[Any, float] = {}
        task_queue = list(training_tasks)
        active_futures = set()

        for _ in range(min(self.context.n_parallel_fighters, len(task_queue))):
            if task_queue:
                task = task_queue.pop(0)
                future = executor.submit(worker.run, task)
                fighter_name = task[0]
                future_to_fighter[future] = fighter_name
                future_to_start_time[future] = time.time()
                active_futures.add(future)
                if self.context.verbose:
                    print(f"    ⏳ Starting: {fighter_name}")
                self.context.logger.info(f"Starting training: {fighter_name}")

        completed_count = 0
        total_fighters = len(training_tasks)

        if self.context.verbose:
            print()
            print(f"  ⏱️  Training in progress (started {total_fighters} fighters)...")
            print()

        while active_futures:
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)

            for future in done:
                fighter_name = future_to_fighter[future]
                completed_count += 1
                elapsed = time.time() - future_to_start_time[future]
                active_futures.remove(future)

                try:
                    result = future.result(timeout=300)
                    results.append(result)
                    if self.context.verbose:
                        print(
                            f"    ✅ [{completed_count}/{total_fighters}] {fighter_name}: "
                            f"{result['episodes']} episodes, mean reward: {result['mean_reward']:.1f}, "
                            f"time: {elapsed:.1f}s"
                        )
                    self.context.logger.info(
                        f"Completed training [{completed_count}/{total_fighters}] {fighter_name}: "
                        f"{result['episodes']} episodes, mean reward: {result['mean_reward']:.1f}, "
                        f"time: {elapsed:.1f}s"
                    )

                    if completed_count % 2 == 0 and completed_count < total_fighters:
                        remaining = total_fighters - completed_count
                        overall_elapsed = time.time() - training_start_time
                        avg_time = overall_elapsed / completed_count
                        eta = avg_time * remaining
                        print(f"       💭 {remaining} fighters still training... (ETA: {int(eta)}s)")
                        print()

                except TimeoutError:
                    self.context.logger.error(
                        f"Fighter {fighter_name} training timed out after 300s"
                    )
                    if self.context.verbose:
                        print(
                            f"    ❌ [{completed_count}/{total_fighters}] "
                            f"{fighter_name}: Training timed out"
                        )
                except Exception as e:
                    import traceback

                    error_msg = str(e) if str(e) else repr(e)
                    tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    self.context.logger.error(
                        f"Fighter {fighter_name} training failed: {error_msg}\n{tb_str}"
                    )
                    if self.context.verbose:
                        if not error_msg:
                            print(
                                f"    ❌ [{completed_count}/{total_fighters}] {fighter_name}: "
                                "Subprocess crashed (likely GPU OOM or segfault)"
                            )
                        else:
                            print(
                                f"    ❌ [{completed_count}/{total_fighters}] "
                                f"{fighter_name}: Training failed - {error_msg}"
                            )

                if task_queue:
                    task = task_queue.pop(0)
                    next_future = executor.submit(worker.run, task)
                    new_fighter_name = task[0]
                    future_to_fighter[next_future] = new_fighter_name
                    future_to_start_time[next_future] = time.time()
                    active_futures.add(next_future)
                    if self.context.verbose:
                        print(f"    ⏳ Starting: {new_fighter_name}")
                    self.context.logger.info(f"Starting training: {new_fighter_name}")

        return results

    def _print_and_log_summary(self, results: List[Dict[str, Any]], total_training_time: float) -> None:
        """Emit training summary to stdout and structured log."""
        if self.context.verbose and results:
            print()
            print("  📊 Training Summary:")
            print(f"     Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
            successful = [r for r in results if "mean_reward" in r]
            if successful:
                mean_rewards = [r["mean_reward"] for r in successful]
                print(f"     Average reward: {sum(mean_rewards)/len(mean_rewards):.1f}")
                print(f"     Best reward: {max(mean_rewards):.1f}")
                print(f"     Worst reward: {min(mean_rewards):.1f}")
            print(f"     Success rate: {len(successful)}/{len(results)}")
            if successful:
                total_episodes = sum(r.get("episodes", 0) for r in successful)
                print(f"     Total episodes: {total_episodes}")
                print(f"     Throughput: {total_episodes/total_training_time:.1f} episodes/sec")

        if results:
            self.context.logger.info("Training Summary:")
            self.context.logger.info(
                f"  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)"
            )
            successful = [r for r in results if "mean_reward" in r]
            if successful:
                mean_rewards = [r["mean_reward"] for r in successful]
                self.context.logger.info(f"  Average reward: {sum(mean_rewards)/len(mean_rewards):.1f}")
                self.context.logger.info(f"  Best reward: {max(mean_rewards):.1f}")
                self.context.logger.info(f"  Worst reward: {min(mean_rewards):.1f}")
            self.context.logger.info(f"  Success rate: {len(successful)}/{len(results)}")
            if successful:
                total_episodes = sum(r.get("episodes", 0) for r in successful)
                self.context.logger.info(f"  Total episodes: {total_episodes}")
                self.context.logger.info(f"  Throughput: {total_episodes/total_training_time:.1f} episodes/sec")
