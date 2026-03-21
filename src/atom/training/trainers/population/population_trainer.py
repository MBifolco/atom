"""
Population-Based Training for Atom Combat

Trains a population of fighters simultaneously, allowing them to learn
from each other and evolve diverse strategies.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
import logging
from datetime import datetime
import random
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Clean relative imports within src package structure
from src.atom.runtime.arena import WorldConfig
from src.atom.training.gym_env import AtomCombatEnv
from src.atom.training.signal_engine import build_observation_from_snapshot
from src.atom.training.utils.observability import append_jsonl, ensure_analysis_dir
from src.atom.training.utils.runtime_platform import configure_runtime_gpu_env
from .elo_tracker import EloTracker
from .population_evaluation import EvaluationContext, PopulationEvaluationService
from .population_evolution import EvolutionContext, PopulationEvolver
from .population_persistence import PopulationPersistenceContext, PopulationPersistenceService
from .population_training_loop import PopulationTrainingLoopContext, PopulationTrainingLoopHelper
from .parallel_orchestrator import (
    ParallelTrainingContext,
    ParallelTrainingOrchestrator,
    TrainingWorker,
)


# ==============================================================================
# HELPER FUNCTIONS FOR PARALLEL TRAINING (must be at module level for pickle)
# ==============================================================================

def _configure_process_threading() -> None:
    """
    Configure thread limits for subprocess to prevent CPU oversubscription.

    Must be called BEFORE importing TensorFlow/PyTorch.
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'


def _reconstruct_config(config_dict: Optional[dict]) -> WorldConfig:
    """Reconstruct WorldConfig from dictionary."""
    from src.atom.runtime.arena import WorldConfig

    if config_dict is None:
        return WorldConfig()
    return WorldConfig(**config_dict)


def _compute_training_summary(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics from training results.

    Args:
        results: List of training result dictionaries with 'mean_reward' and 'episodes' keys

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {"successful": 0, "total": 0}

    successful = [r for r in results if 'mean_reward' in r]
    summary = {
        "successful": len(successful),
        "total": len(results),
        "success_rate": len(successful) / len(results) if results else 0
    }

    if successful:
        mean_rewards = [r['mean_reward'] for r in successful]
        total_episodes = sum(r.get('episodes', 0) for r in successful)
        summary.update({
            "mean_rewards": mean_rewards,
            "average_reward": sum(mean_rewards) / len(mean_rewards),
            "best_reward": max(mean_rewards),
            "worst_reward": min(mean_rewards),
            "total_episodes": total_episodes
        })

    return summary


def _compute_training_progress(
    current_timestep: int,
    total_timesteps: int,
    elapsed_time: float
) -> Dict[str, float]:
    """
    Compute training progress metrics.

    Args:
        current_timestep: Current training timestep
        total_timesteps: Target total timesteps
        elapsed_time: Elapsed time in seconds

    Returns:
        Dictionary with progress metrics
    """
    progress_pct = min(100, int(current_timestep * 100 / total_timesteps)) if total_timesteps > 0 else 0
    timesteps_per_sec = current_timestep / elapsed_time if elapsed_time > 0 else 0
    remaining_timesteps = total_timesteps - current_timestep
    eta_sec = remaining_timesteps / timesteps_per_sec if timesteps_per_sec > 0 else 0

    return {
        "progress_pct": progress_pct,
        "timesteps_per_sec": timesteps_per_sec,
        "eta_sec": eta_sec
    }


def _apply_weight_mutation(model, mutation_rate: float) -> None:
    """
    Apply Gaussian noise mutation to model weights.

    Args:
        model: A Stable-Baselines3 model (PPO or SAC)
        mutation_rate: Mutation intensity (0.1 = 10% noise)
    """
    import torch
    with torch.no_grad():
        for param in model.policy.parameters():
            noise_scale = mutation_rate * 0.1
            noise = torch.randn_like(param) * noise_scale
            param.data.add_(noise)


def _calculate_win_rate(wins: int, losses: int, draws: int) -> float:
    """
    Calculate win rate from match statistics.

    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws

    Returns:
        Win rate as float (0.0 to 1.0)
    """
    total_matches = wins + losses + draws
    if total_matches == 0:
        return 0.0
    return (wins + 0.5 * draws) / total_matches


def _select_parent_weighted(survivors: List, elo_tracker) -> Any:
    """
    Select a parent fighter weighted by ELO rating.

    Args:
        survivors: List of surviving fighters
        elo_tracker: ELO tracker instance

    Returns:
        Selected parent fighter
    """
    import random
    # Simple random selection for now (could be weighted by ELO later)
    return random.choice(survivors)


def _format_training_banner(
    population_size: int,
    generations: int,
    episodes_per_generation: int,
    evolution_frequency: int,
    base_model_path: Optional[str] = None
) -> str:
    """
    Format the training banner shown at start.

    Args:
        population_size: Size of the population
        generations: Number of generations to train
        episodes_per_generation: Episodes per generation
        evolution_frequency: How often to evolve
        base_model_path: Optional path to base model

    Returns:
        Formatted banner string
    """
    lines = [
        "",
        "=" * 80,
        "STARTING POPULATION TRAINING",
        "=" * 80,
        f"Population Size: {population_size}",
        f"Generations: {generations}",
        f"Episodes per Generation: {episodes_per_generation}",
        f"Evolution Frequency: Every {evolution_frequency} generations",
    ]
    if base_model_path:
        lines.append(f"Base Model: {base_model_path}")
    lines.append("=" * 80)
    return "\n".join(lines)


def _format_generation_header(current_gen: int, total_gens: int) -> str:
    """
    Format the generation header.

    Args:
        current_gen: Current generation number (1-indexed)
        total_gens: Total number of generations

    Returns:
        Formatted header string
    """
    return f"\n{'='*80}\nGENERATION {current_gen}/{total_gens}\n{'='*80}"


def _format_final_report(
    total_generations: int,
    total_matches: int,
    diversity_metrics: Dict[str, float]
) -> str:
    """
    Format the final training report.

    Args:
        total_generations: Total generations completed
        total_matches: Total matches run
        diversity_metrics: Dictionary with elo_range, elo_std, win_rate_std

    Returns:
        Formatted report string
    """
    lines = [
        "",
        "=" * 80,
        "TRAINING COMPLETE",
        "=" * 80,
        f"Total Generations: {total_generations}",
        f"Total Matches: {total_matches}",
        "",
        "Population Diversity:",
        f"  ELO Spread: {diversity_metrics.get('elo_range', 0):.0f}",
        f"  ELO Std Dev: {diversity_metrics.get('elo_std', 0):.1f}",
    ]
    if 'win_rate_std' in diversity_metrics:
        lines.append(f"  Win Rate Variance: {diversity_metrics['win_rate_std']:.3f}")
    return "\n".join(lines)


def _select_opponents_for_fighter(
    fighter,
    pairs: List[Tuple],
    all_fighters: List,
    max_opponents: int = 3
) -> List:
    """
    Select opponents for a fighter from matchmaking pairs.

    Args:
        fighter: The fighter to find opponents for
        pairs: List of matchmaking pairs (fighter1, fighter2)
        all_fighters: All fighters in population
        max_opponents: Maximum number of opponents to return

    Returns:
        List of opponent fighters
    """
    import random

    opponents = []
    for pair in pairs:
        if pair[0] == fighter:
            opponents.append(pair[1])
        elif pair[1] == fighter:
            opponents.append(pair[0])

    # If no opponents from pairs, use random selection
    if not opponents:
        available = [f for f in all_fighters if f != fighter]
        opponents = random.sample(available, min(max_opponents, len(available)))

    return opponents


def _calculate_batch_eta(
    completed: int,
    total: int,
    elapsed_time: float
) -> float:
    """
    Calculate estimated time remaining for batch processing.

    Args:
        completed: Number of completed items
        total: Total number of items
        elapsed_time: Time elapsed so far in seconds

    Returns:
        Estimated seconds remaining
    """
    if completed == 0:
        return 0.0
    avg_time = elapsed_time / completed
    remaining = total - completed
    return avg_time * remaining


def _format_training_result_line(
    completed: int,
    total: int,
    fighter_name: str,
    episodes: int,
    mean_reward: float,
    elapsed: float
) -> str:
    """
    Format a single training result line.

    Args:
        completed: Completion number
        total: Total count
        fighter_name: Name of the fighter
        episodes: Number of episodes completed
        mean_reward: Mean reward achieved
        elapsed: Time elapsed in seconds

    Returns:
        Formatted result string
    """
    return (f"[{completed}/{total}] {fighter_name}: "
            f"{episodes} episodes, mean reward: {mean_reward:.1f}, "
            f"time: {elapsed:.1f}s")


def _load_opponent_models_for_training(
    opponent_data: List[Tuple[str, float, str]],
    algorithm: str
) -> List[Tuple[str, float, Any]]:
    """
    Load opponent models from paths for training.

    Args:
        opponent_data: List of (name, mass, model_path) tuples
        algorithm: "ppo" or "sac"

    Returns:
        List of (name, mass, model) tuples
    """
    from stable_baselines3 import PPO, SAC

    opponent_models = []
    for opp_name, opp_mass, opp_path in opponent_data:
        if algorithm == "ppo":
            opp_model = PPO.load(opp_path, device="cpu")
        else:
            opp_model = SAC.load(opp_path, device="cpu")
        opponent_models.append((opp_name, opp_mass, opp_model))

    return opponent_models


def _create_opponent_decide_func(model):
    """
    Create a decide function wrapper for a trained PPO model.

    Args:
        model: Trained SB3 PPO model

    Returns:
        Decide function compatible with AtomCombatEnv
    """
    def decide(snapshot):
        obs = build_observation_from_snapshot(snapshot, recent_damage=0.0)

        action, _ = model.predict(obs, deterministic=False)

        acceleration = float(action[0]) * 4.5
        stance_idx = int(np.clip(action[1], 0, 2))
        stances = ["neutral", "extended", "defending"]
        stance = stances[min(stance_idx, 2)]

        return {"acceleration": acceleration, "stance": stance}

    return decide


def _create_vmap_training_environment(
    fighter_mass: float,
    opponent_models: List[Tuple],
    config: WorldConfig,
    n_vmap_envs: int,
    max_ticks: int
):
    """
    Create JAX vmap vectorized environment for GPU training.

    Args:
        fighter_mass: Mass of the learning fighter
        opponent_models: List of (name, mass, model) tuples
        config: WorldConfig instance
        n_vmap_envs: Number of parallel vmap environments
        max_ticks: Maximum ticks per episode

    Returns:
        VmapEnvAdapter wrapped environment
    """
    import os
    from src.atom.training.trainers.curriculum_trainer import VmapEnvAdapter
    from src.atom.training.vmap_env_wrapper import VmapEnvWrapper

    # Configure JAX memory to prevent OOM
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # Extract just the models for vmap
    opp_models_only = [opp_model for _, _, opp_model in opponent_models]

    vmap_env = VmapEnvWrapper(
        n_envs=n_vmap_envs,
        opponent_models=opp_models_only,
        config=config,
        max_ticks=max_ticks,
        fighter_mass=fighter_mass,
        opponent_mass=fighter_mass,  # Assume same mass for simplicity
        seed=42
    )

    return VmapEnvAdapter(vmap_env)


def _create_cpu_training_environment(
    fighter_name: str,
    fighter_mass: float,
    opponent_models: List[Tuple],
    config: WorldConfig,
    n_envs: int,
    max_ticks: int,
    logs_dir: str
):
    """
    Create CPU-based parallel environments using DummyVecEnv.

    Args:
        fighter_name: Name of the fighter (for logging)
        fighter_mass: Mass of the learning fighter
        opponent_models: List of (name, mass, model) tuples
        config: WorldConfig instance
        n_envs: Number of parallel environments
        max_ticks: Maximum ticks per episode
        logs_dir: Directory for monitor logs

    Returns:
        DummyVecEnv with Monitor-wrapped environments
    """
    from pathlib import Path
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from src.atom.training.gym_env import AtomCombatEnv

    env_fns = []
    for i in range(n_envs):
        opp_name, opp_mass, opp_model = opponent_models[i % len(opponent_models)]

        # Use closure with default arguments to capture values
        def make_env(opp_m=opp_model, f_mass=fighter_mass, o_mass=opp_mass, env_idx=i):
            return Monitor(
                AtomCombatEnv(
                    opponent_decision_func=_create_opponent_decide_func(opp_m),
                    config=config,
                    max_ticks=max_ticks,
                    fighter_mass=f_mass,
                    opponent_mass=o_mass
                ),
                str(Path(logs_dir) / f"{fighter_name}_env_{env_idx}")
            )

        env_fns.append(make_env)

    return DummyVecEnv(env_fns)


# ==============================================================================
# TOP-LEVEL TRAINING FUNCTION (must be at module level for pickle)
# ==============================================================================

def _train_single_fighter_parallel(
    fighter_name: str,
    fighter_mass: float,
    model_path: str,
    opponent_data: List[Tuple[str, float, str]],  # List of (name, mass, model_path)
    n_envs: int,
    episodes: int,
    max_ticks: int,
    algorithm: str,
    config_dict: dict,
    logs_dir: str,
    use_vmap: bool = True,  # GPU mode by default
    n_vmap_envs: int = 250
) -> Dict:
    """
    Train a single fighter in a separate process.

    This function must be at module level (not a method) to be picklable.
    All arguments must be serializable (no model objects, only paths).

    Args:
        fighter_name: Name of the fighter
        fighter_mass: Mass of the fighter
        model_path: Path to the fighter's model file
        opponent_data: List of (opponent_name, opponent_mass, opponent_model_path)
        n_envs: Number of parallel environments (CPU mode)
        episodes: Number of episodes to train
        max_ticks: Max ticks per episode
        algorithm: "ppo" or "sac"
        config_dict: WorldConfig as dictionary
        logs_dir: Directory for logs
        use_vmap: Use JAX vmap for GPU acceleration
        n_vmap_envs: Number of vmap environments (GPU mode)

    Returns:
        Dictionary with training statistics
    """
    # Configure threading for subprocess
    _configure_process_threading()

    # Configure GPU memory/runtime for subprocess (if using GPU/vmap)
    if use_vmap:
        configure_runtime_gpu_env(enable_gpu=True, memory_fraction=0.75)

        try:
            import torch
            if torch.cuda.is_available():
                # Limit each process to only a fraction of memory.
                memory_fraction = 0.75
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        except Exception:
            pass  # Silently continue if torch not available or no GPU

        try:
            import jax
            # Clear any existing JAX cache from parent process.
            jax.clear_caches()
        except Exception:
            pass

    # Required imports (after threading config)
    import numpy as np
    from pathlib import Path
    from stable_baselines3 import PPO, SAC
    from src.atom.runtime.arena import WorldConfig

    # Reconstruct config and load opponents
    config = _reconstruct_config(config_dict)
    opponent_models = _load_opponent_models_for_training(opponent_data, algorithm)

    # Create training environments (vmap for GPU or DummyVecEnv for CPU)
    if use_vmap:
        vec_env = _create_vmap_training_environment(
            fighter_mass, opponent_models, config, n_vmap_envs, max_ticks
        )
    else:
        vec_env = _create_cpu_training_environment(
            fighter_name, fighter_mass, opponent_models, config, n_envs, max_ticks, logs_dir
        )

    # Load fighter model
    if algorithm == "ppo":
        fighter_model = PPO.load(model_path, env=vec_env)
    else:
        fighter_model = SAC.load(model_path, env=vec_env)

    # Disable tensorboard logging for population training to avoid confusion
    # (models loaded from curriculum keep their old tensorboard paths)
    fighter_model.tensorboard_log = None
    # Also disable verbose output to prevent rollout stats in stdout
    fighter_model.verbose = 0

    # Track statistics
    episode_count = 0
    recent_rewards = []
    last_report_timestep = 0

    import time

    start_time = time.time()
    last_update_time = start_time

    # Calculate timesteps
    avg_ticks_per_episode = 100
    total_timesteps = episodes * avg_ticks_per_episode

    # Get logger for this subprocess early so callback can use it
    import logging
    import os
    log_dir = Path(logs_dir)  # Use logs_dir which is passed as parameter
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"population_training_{os.getpid()}.log"
    subprocess_logger = logging.getLogger(f"population_fighter_{fighter_name}")
    if not subprocess_logger.handlers:
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        subprocess_logger.addHandler(handler)
        subprocess_logger.setLevel(logging.INFO)

    # Enhanced callback with progress reporting
    class ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            nonlocal episode_count, recent_rewards, last_report_timestep, last_update_time

            for info in self.locals.get("infos", []):
                if "episode" in info:
                    episode_count += 1
                    reward = info["episode"]["r"]
                    recent_rewards.append(reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)

            # Report based on timesteps, not episodes (since PPO trains on timesteps)
            current_timestep = self.num_timesteps
            now = time.time()

            # Report every 2500 timesteps or every 10 seconds
            if current_timestep - last_report_timestep >= 2500 or now - last_update_time >= 10:
                elapsed = now - start_time
                avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                progress_pct = min(100, int(current_timestep * 100 / total_timesteps))

                timesteps_per_sec = current_timestep / elapsed if elapsed > 0 else 0
                eta_sec = (total_timesteps - current_timestep) / timesteps_per_sec if timesteps_per_sec > 0 else 0

                progress_msg = (f"[{fighter_name}] Progress: {progress_pct}% "
                               f"({current_timestep}/{total_timesteps} steps, {episode_count} eps) | "
                               f"Reward: {avg_reward:.1f} | "
                               f"ETA: {int(eta_sec)}s")
                print(f"      {progress_msg}", flush=True)
                subprocess_logger.info(progress_msg)

                last_report_timestep = current_timestep
                last_update_time = now

            return True

    callback = ProgressCallback()

    print(f"    [{fighter_name}] 🎮 Starting training (target: ~{episodes} episodes, {total_timesteps} timesteps)...", flush=True)
    subprocess_logger.info(f"[{fighter_name}] Starting training (target: ~{episodes} episodes, {total_timesteps} timesteps)")

    # Train - this includes both rollout collection AND neural network updates
    # PPO does rollouts in batches, then trains NN multiple epochs on that data
    fighter_model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=True  # Reset to show only current session's timesteps
    )

    training_time = time.time() - start_time
    completion_msg = (f"[{fighter_name}] Training complete in {training_time:.1f}s "
                      f"({episode_count} episodes, {episode_count/training_time:.1f} eps/sec)")
    print(f"    [{fighter_name}] ✅ Training complete in {training_time:.1f}s "
          f"({episode_count} episodes, {episode_count/training_time:.1f} eps/sec)", flush=True)
    subprocess_logger.info(completion_msg)

    # Save updated model
    fighter_model.save(model_path)

    vec_env.close()

    # Clean up GPU memory if using vmap
    if use_vmap:
        # Delete model to free GPU memory
        del fighter_model

        # Clear GPU caches
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        try:
            import jax
            jax.clear_caches()
        except:
            pass

        # Force garbage collection
        import gc
        gc.collect()

    # Return statistics
    return {
        "fighter": fighter_name,
        "episodes": episode_count,
        "mean_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
        "opponent_names": [name for name, _, _ in opponent_data]
    }


@dataclass
class PopulationFighter:
    """Represents a fighter in the population."""
    name: str
    model: PPO  # or SAC
    generation: int = 0
    lineage: str = "founder"
    mass: float = 70.0
    training_episodes: int = 0
    last_checkpoint: Optional[str] = None


class PopulationCallback(BaseCallback):
    """Callback for tracking population training progress."""

    def __init__(self, fighter_name: str, elo_tracker: EloTracker, verbose: int = 0):
        super().__init__(verbose)
        self.fighter_name = fighter_name
        self.elo_tracker = elo_tracker
        self.episode_count = 0
        self.recent_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                self.recent_rewards.append(reward)

                # Keep only last 100 episodes
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)

        return True


class PopulationTrainer:
    """
    Manages population-based training for Atom Combat fighters.

    Key features:
    - Trains multiple fighters simultaneously
    - Fighters learn by competing against each other
    - Tracks performance with ELO ratings
    - Implements evolution/selection mechanics
    - Supports parallel training across CPU cores
    """

    def __init__(self,
                 population_size: int = 8,
                 config: WorldConfig = None,
                 algorithm: str = "ppo",
                 output_dir: str = "outputs/population",
                 n_envs_per_fighter: int = 2,
                 max_ticks: int = 250,
                 mass_range: Tuple[float, float] = (70.0, 70.0),
                 verbose: bool = True,
                 export_threshold: float = 0.5,
                 n_parallel_fighters: int = None,
                 use_vmap: bool = True,  # GPU mode by default (77x speedup)
                 n_vmap_envs: int = 250,
                 record_replays: bool = False,
                 replay_recording_frequency: int = 5,
                 replay_matches_per_pair: int = 2,
                 seed: int = 1337):
        """
        Initialize the population trainer.

        Args:
            population_size: Number of fighters in the population
            config: World configuration
            algorithm: "ppo" (only PPO supported)
            output_dir: Directory for saving models and logs
            n_envs_per_fighter: Parallel environments per fighter (CPU mode)
            max_ticks: Maximum ticks per episode
            mass_range: Range of masses for fighter variety
            verbose: Whether to print progress
            export_threshold: Minimum win rate to export fighters to AIs directory
            n_parallel_fighters: Number of fighters to train in parallel (default: cpu_count - 1)
            use_vmap: Use JAX vmap for GPU-accelerated training (77x speedup)
            n_vmap_envs: Number of vmap environments for GPU mode (default: 250)
            record_replays: Whether to record fight replays for montage
            replay_recording_frequency: Record replays every N generations
            replay_matches_per_pair: Number of evaluation matches per fighter pair for replay recording
            seed: Training seed used for deterministic population initialization
        """
        self.population_size = population_size
        self.config = config or WorldConfig()
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.n_envs_per_fighter = n_envs_per_fighter
        self.max_ticks = max_ticks
        self.mass_range = mass_range
        self.verbose = verbose
        self.export_threshold = export_threshold
        self.use_vmap = use_vmap
        self.n_vmap_envs = n_vmap_envs
        self.record_replays = record_replays
        self.replay_recording_frequency = replay_recording_frequency
        self.replay_matches_per_pair = replay_matches_per_pair
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Parallel training configuration
        if n_parallel_fighters is None:
            if use_vmap:
                # GPU mode: Use sequential training to avoid GPU OOM
                # Multiple processes sharing GPU memory is problematic with ROCm
                n_parallel_fighters = 1  # Sequential GPU training
                if verbose:
                    print(f"⚠️  GPU mode: Training fighters sequentially (1 at a time)")
                    print(f"   (prevents GPU out-of-memory errors with ROCm)")
                    print(f"   Each fighter uses JAX vmap with {n_vmap_envs} parallel environments")
                    print(f"   💡 For parallel training, use CPU mode: --no-vmap")
            else:
                # CPU mode: Use more parallelism
                n_parallel_fighters = max(1, multiprocessing.cpu_count() - 1)
        elif use_vmap and verbose:
            if n_parallel_fighters > 1:
                print(f"⚠️  GPU mode: Using {n_parallel_fighters} parallel fighters")
                print(f"   WARNING: This may cause GPU out-of-memory errors!")
                print(f"   Recommended: Use --n-parallel-fighters 1 for GPU mode")
            else:
                print(f"✅ GPU mode: Sequential training (1 fighter at a time)")

        self.n_parallel_fighters = n_parallel_fighters

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.analysis_dir = ensure_analysis_dir(self.output_dir)

        # Initialize population
        self.population: List[PopulationFighter] = []
        self.elo_tracker = EloTracker()

        # Training state
        self.generation = 0
        self.total_matches = 0

        # Replay recorder (if enabled)
        self.replay_recorder = None
        if self.record_replays:
            from ...replay_recorder import ReplayRecorder
            self.replay_recorder = ReplayRecorder(
                output_dir=str(self.output_dir),
                config=self.config,
                max_ticks=max_ticks,
                verbose=verbose
            )

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for population training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"population_training_{timestamp}.log"

        self.logger = logging.getLogger('population_trainer')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.propagate = False

        self.logger.info("="*80)
        self.logger.info("POPULATION TRAINING INITIALIZED")
        self.logger.info(f"Population Size: {self.population_size}")
        self.logger.info(f"Algorithm: {self.algorithm}")
        self.logger.info(f"Mass Range: {self.mass_range}")
        self.logger.info(f"Parallel Fighters: {self.n_parallel_fighters}")
        if self.use_vmap:
            self.logger.info(f"GPU Acceleration: ENABLED (vmap with {self.n_vmap_envs} envs)")
        else:
            self.logger.info(f"GPU Acceleration: DISABLED (CPU with {self.n_envs_per_fighter} envs)")
        self.logger.info("="*80)

    def _create_fighter_name(self, index: int, generation: int = 0) -> str:
        """Generate a deterministic fighter name with optional funkybob support."""
        import random

        # Reproducible seed based on index and generation.
        seed = index + (generation * 1000)
        rng = random.Random(seed)

        try:
            import funkybob
        except ImportError:
            # Fallback path for environments where funkybob is unavailable (e.g., some Colab runtimes).
            adjectives = [
                "Swift", "Iron", "Clever", "Bold", "Silent", "Fierce",
                "Rapid", "Stone", "Noble", "Rogue", "Brisk", "Prime",
            ]
            animals = [
                "Falcon", "Viper", "Wolf", "Tiger", "Eagle", "Panther",
                "Raven", "Cobra", "Jaguar", "Lynx", "Hawk", "Shark",
            ]
            base_name = f"{rng.choice(adjectives)}_{rng.choice(animals)}"
        else:
            # funkybob uses global random state; isolate and restore it.
            previous_state = random.getstate()
            random.seed(seed)
            try:
                name_generator = funkybob.RandomNameGenerator(members=2, separator='_')
                base_name = next(iter(name_generator))
            finally:
                random.setstate(previous_state)

        # Add generation suffix if not first generation
        if generation > 0:
            return f"{base_name}_G{generation}"
        return base_name

    def initialize_population(self, base_model_path: Optional[str] = None, variation_factor: float = 0.1):
        """
        Initialize the population with new fighters.

        Args:
            base_model_path: Optional path to a pre-trained model to use as base
            variation_factor: How much to vary models if using base model (0-1)
        """
        if self.verbose:
            print(f"\nInitializing population of {self.population_size} fighters...")
            if base_model_path:
                print(f"  Using base model: {base_model_path}")
                print(f"  Variation factor: {variation_factor}")

        for i in range(self.population_size):
            name = self._create_fighter_name(i, self.generation)
            mass = np.random.uniform(*self.mass_range)

            # Create a simple environment for model initialization
            env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
                opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
                config=self.config,
                max_ticks=self.max_ticks,
                fighter_mass=mass,
                opponent_mass=70.0
            ))])

            # Create or load model
            if base_model_path and Path(base_model_path).exists():
                # Load from base model
                if self.algorithm == "ppo":
                    model = PPO.load(base_model_path, env=env)
                else:  # SAC
                    model = SAC.load(base_model_path, env=env)

                # Add variation for diversity (except first fighter)
                if i > 0 and variation_factor > 0:
                    self._add_variation_to_model(model, variation_factor)

            else:
                # Create new model from scratch
                if self.algorithm == "ppo":
                    model = PPO(
                        "MlpPolicy",
                        env,
                        verbose=0,
                        learning_rate=1e-4,
                        n_steps=2048 // self.n_envs_per_fighter,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.01  # Encourage exploration
                    )
                else:  # SAC
                    model = SAC(
                        "MlpPolicy",
                        env,
                        verbose=0,
                        learning_rate=1e-4,  # Lower for fine-tuning and stability
                        buffer_size=50000,
                        learning_starts=100,
                        batch_size=256,
                        tau=0.005,
                        gamma=0.99
                    )

            fighter = PopulationFighter(
                name=name,
                model=model,
                generation=self.generation,
                mass=mass
            )

            self.population.append(fighter)
            self.elo_tracker.add_fighter(name)

            if self.verbose:
                print(f"  Created {name} (mass: {mass:.1f}kg)")

        self.logger.info(f"Initialized {len(self.population)} fighters")

    def _add_variation_to_model(self, model, variation_factor: float):
        """
        Add random variation to model parameters to increase diversity.

        Args:
            model: The model to vary
            variation_factor: How much variation to add (0-1)
        """
        import torch

        # Get policy parameters
        policy_params = model.policy.state_dict()

        # Add noise to weights and biases
        for key, value in policy_params.items():
            if 'weight' in key or 'bias' in key:
                # Add gaussian noise proportional to parameter magnitude
                noise = torch.randn_like(value) * variation_factor * 0.1
                value.data += value.data * noise

        # Update the model with varied parameters
        model.policy.load_state_dict(policy_params)

    def _get_fighter_decision_func(self, fighter: PopulationFighter) -> Callable:
        """Create a decision function for a trained fighter."""
        def decide(snapshot):
            obs = build_observation_from_snapshot(snapshot, recent_damage=0.0)

            # Get action from model
            action, _ = fighter.model.predict(obs, deterministic=False)

            # Convert continuous action to game action
            acceleration = float(action[0]) * 4.5  # Scale from [-1, 1] to [-4.5, 4.5]
            stance_idx = int(action[1])
            stances = ["neutral", "extended", "defending"]  # Only 3 stances now
            stance = stances[min(stance_idx, 2)]  # Clamp to 0-2

            return {"acceleration": acceleration, "stance": stance}

        return decide

    def create_matchmaking_pairs(self) -> List[Tuple[PopulationFighter, PopulationFighter]]:
        """
        Create pairs of fighters for training matches.

        Uses multiple strategies:
        1. Similar skill matches (based on ELO)
        2. Random matches for diversity
        3. Top vs bottom for teaching
        """
        pairs = []
        available_fighters = self.population.copy()
        random.shuffle(available_fighters)

        # Strategy 1: Similar skill pairs (50% of matches)
        active_names = [f.name for f in self.population]
        balanced_matches = self.elo_tracker.suggest_balanced_matches(
            len(available_fighters) // 4,
            active_fighters=active_names
        )
        for name_a, name_b in balanced_matches:
            fighter_a = next((f for f in self.population if f.name == name_a), None)
            fighter_b = next((f for f in self.population if f.name == name_b), None)
            if fighter_a and fighter_b:
                pairs.append((fighter_a, fighter_b))

        # Strategy 2: Random pairs for the rest
        used_fighters = set()
        for pair in pairs:
            used_fighters.add(pair[0].name)
            used_fighters.add(pair[1].name)

        remaining = [f for f in available_fighters if f.name not in used_fighters]
        random.shuffle(remaining)

        for i in range(0, len(remaining) - 1, 2):
            pairs.append((remaining[i], remaining[i+1]))

        return pairs

    def train_fighters_parallel(self,
                                fighter_opponent_pairs: List[Tuple[PopulationFighter, List[PopulationFighter]]],
                                episodes_per_fighter: int) -> List[Dict]:
        """
        Train multiple fighters in parallel using process pool.

        Args:
            fighter_opponent_pairs: List of (fighter, opponents) tuples
            episodes_per_fighter: Number of episodes each fighter should train

        Returns:
            List of training statistics dictionaries
        """
        orchestrator = ParallelTrainingOrchestrator(self._build_parallel_training_context())
        worker = TrainingWorker(_train_single_fighter_parallel)
        return orchestrator.run(
            fighter_opponent_pairs=fighter_opponent_pairs,
            episodes_per_fighter=episodes_per_fighter,
            worker=worker,
            executor_factory=ProcessPoolExecutor,
        )

    def _build_parallel_training_context(self) -> ParallelTrainingContext:
        """Build immutable context passed to parallel orchestration helpers."""
        return ParallelTrainingContext(
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
            config=self.config,
            max_ticks=self.max_ticks,
            algorithm=self.algorithm,
            n_envs_per_fighter=self.n_envs_per_fighter,
            n_parallel_fighters=self.n_parallel_fighters,
            use_vmap=self.use_vmap,
            n_vmap_envs=self.n_vmap_envs,
            generation=self.generation,
            verbose=self.verbose,
            logger=self.logger,
        )

    def train_fighter_batch(self,
                           fighter: PopulationFighter,
                           opponents: List[PopulationFighter],
                           episodes: int = 100) -> Dict:
        """
        Train a single fighter against a batch of opponents (sequential fallback).

        Returns training statistics.
        """
        # Create environments with different opponents
        env_fns = []
        opponent_names = []

        for i in range(self.n_envs_per_fighter):
            opponent = opponents[i % len(opponents)]
            opponent_names.append(opponent.name)

            env_fn = lambda opp=opponent, fighter_mass=fighter.mass, opp_mass=opponent.mass: Monitor(
                AtomCombatEnv(
                    opponent_decision_func=self._get_fighter_decision_func(opp),
                    config=self.config,
                    max_ticks=self.max_ticks,
                    fighter_mass=fighter_mass,
                    opponent_mass=opp_mass
                )
            )
            env_fns.append(env_fn)

        # Create vectorized environment
        # Note: Using DummyVecEnv to avoid pickle issues with SubprocVecEnv
        # The fighter models contain weakrefs that can't be pickled for multiprocessing
        vec_env = DummyVecEnv(env_fns)

        # If the model has a different number of envs, we need to reload it with the new env
        # (set_env requires matching number of environments)
        if vec_env.num_envs != fighter.model.n_envs:
            # Save current model temporarily
            temp_path = self.models_dir / f"temp_{fighter.name}.zip"
            temp_path.parent.mkdir(exist_ok=True)
            fighter.model.save(temp_path)

            # Reload with new environment
            if self.algorithm == "ppo":
                fighter.model = PPO.load(temp_path, env=vec_env)
            else:
                fighter.model = SAC.load(temp_path, env=vec_env)

            # Clean up temp file
            temp_path.unlink()
        else:
            # Same number of envs, can use set_env
            fighter.model.set_env(vec_env)

        # Create callback
        callback = PopulationCallback(fighter.name, self.elo_tracker)

        # Calculate timesteps
        avg_ticks_per_episode = 100
        total_timesteps = episodes * avg_ticks_per_episode

        # Train
        fighter.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=False  # Continue from previous training
        )

        fighter.training_episodes += episodes

        # Return statistics
        stats = {
            "fighter": fighter.name,
            "episodes": callback.episode_count,
            "mean_reward": np.mean(callback.recent_rewards) if callback.recent_rewards else 0,
            "opponents": opponent_names
        }

        vec_env.close()

        return stats

    def run_evaluation_matches(self, num_matches_per_pair: int = 3) -> None:
        """
        Run evaluation matches between all fighters to update ELO ratings.
        """
        service = PopulationEvaluationService(self._build_evaluation_context())
        matches_run = service.run(
            population=self.population,
            elo_tracker=self.elo_tracker,
            decision_func_factory=self._get_fighter_decision_func,
            env_factory=AtomCombatEnv,
            num_matches_per_pair=num_matches_per_pair,
        )
        self.total_matches += matches_run

    def _build_evaluation_context(self) -> EvaluationContext:
        """Build immutable context for evaluation helpers."""
        return EvaluationContext(
            config=self.config,
            max_ticks=self.max_ticks,
            generation=self.generation,
            verbose=self.verbose,
            logger=self.logger,
        )

    def evolve_population(self, keep_top: float = 0.5, mutation_rate: float = 0.1) -> None:
        """
        Evolve the population by replacing weak fighters with mutations of strong ones.

        Args:
            keep_top: Fraction of population to keep
            mutation_rate: How much to vary the cloned models
        """
        evolver = PopulationEvolver(self._build_evolution_context())
        evolver.evolve(
            population=self.population,
            elo_tracker=self.elo_tracker,
            keep_top=keep_top,
            mutation_rate=mutation_rate,
            create_fighter_name=self._create_fighter_name,
            fighter_factory=self._create_population_fighter,
        )

    def _build_evolution_context(self) -> EvolutionContext:
        """Build immutable context passed to population evolution helpers."""
        return EvolutionContext(
            config=self.config,
            max_ticks=self.max_ticks,
            mass_range=self.mass_range,
            generation=self.generation,
            algorithm=self.algorithm,
            verbose=self.verbose,
            logger=self.logger,
        )

    def _create_population_fighter(
        self,
        name: str,
        model: Any,
        generation: int,
        lineage: str,
        mass: float,
    ) -> PopulationFighter:
        """Factory wrapper for creating population fighters during evolution."""
        return PopulationFighter(
            name=name,
            model=model,
            generation=generation,
            lineage=lineage,
            mass=mass,
        )

    def save_population(self, min_win_rate: float = None) -> None:
        """
        Save all fighters in the population.

        Also exports qualifying fighters to fighters/AIs/ directory in atom_fight.py format.

        Args:
            min_win_rate: Minimum win rate to export to AIs directory (default: use export_threshold)
        """
        if min_win_rate is None:
            min_win_rate = self.export_threshold
        persistence = self._build_population_persistence_service()
        generation_dir = persistence.generation_dir()
        persistence.save_generation_models(self.population, generation_dir)
        persistence.write_rankings_file(self.elo_tracker.get_rankings(), generation_dir)

        # Export qualifying fighters to AIs directory
        self._export_qualifying_fighters(min_win_rate)

        if self.verbose:
            print(f"\nSaved generation {self.generation} to {generation_dir}")

    def _build_population_persistence_context(self) -> PopulationPersistenceContext:
        """Build immutable context for persistence/export helpers."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        return PopulationPersistenceContext(
            models_dir=self.models_dir,
            project_root=project_root,
            algorithm=self.algorithm,
            population_size=self.population_size,
            generation=self.generation,
            verbose=self.verbose,
            logger=self.logger,
        )

    def _build_population_persistence_service(self) -> PopulationPersistenceService:
        """Create a persistence/export helper service for current trainer state."""
        return PopulationPersistenceService(self._build_population_persistence_context())

    def _export_qualifying_fighters(self, min_win_rate: float = 0.5) -> None:
        """
        Export fighters that meet win rate threshold to fighters/AIs/ directory.

        Each fighter gets its own folder with:
        - {fighter_name}.onnx - ONNX model for inference
        - {fighter_name}.py - Python wrapper with decide() function
        - README.md - Fighter stats and metadata

        Args:
            min_win_rate: Minimum win rate to qualify for export
        """
        # Calculate win rates for all fighters
        rankings = self.elo_tracker.get_rankings()
        persistence = self._build_population_persistence_service()
        ais_dir = persistence.resolve_ais_dir()

        exported_count = 0

        for fighter in self.population:
            # Get fighter stats
            stats = next((s for s in rankings if s.name == fighter.name), None)
            if not stats:
                continue

            # Calculate win rate
            win_rate = persistence.compute_win_rate(stats)

            # Check if fighter qualifies
            if win_rate >= min_win_rate:
                try:
                    self._export_fighter_to_ais(fighter, stats, win_rate, ais_dir)
                    exported_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to export {fighter.name}: {e}")
                    self._record_export_failure(
                        fighter_name=fighter.name,
                        export_target=str(ais_dir / fighter.name),
                        exception=e,
                        training_artifacts_saved=True,
                        win_rate=win_rate,
                        elo=float(stats.elo),
                    )
                    if self.verbose:
                        print(f"  ⚠️  Failed to export {fighter.name}: {e}")

        if self.verbose and exported_count > 0:
            print(f"\n✅ Exported {exported_count} qualifying fighters to {ais_dir}")

    def _export_fighter_to_ais(self,
                                fighter: PopulationFighter,
                                stats,
                                win_rate: float,
                                ais_dir: Path) -> None:
        """
        Export a single fighter to the AIs directory.

        Creates a folder with ONNX model, Python wrapper, and README.
        """
        persistence = self._build_population_persistence_service()
        fighter_dir = persistence.export_fighter_bundle(
            fighter=fighter,
            stats=stats,
            win_rate=win_rate,
            ais_dir=ais_dir,
        )

        if self.verbose:
            print(f"  📦 Exported {fighter.name} (WR: {win_rate:.1%}, ELO: {stats.elo:.0f}) to {fighter_dir}")

    def _export_model_to_onnx(self, model, output_path: Path) -> None:
        """Export a Stable-Baselines3 model to ONNX format."""
        persistence = self._build_population_persistence_service()
        persistence.export_model_to_onnx(model, output_path)

    def _create_fighter_wrapper(self, fighter: PopulationFighter, output_path: Path, onnx_filename: str) -> None:
        """Create a Python wrapper file with decide() function for atom_fight.py."""
        persistence = self._build_population_persistence_service()
        persistence.create_fighter_wrapper(fighter, output_path, onnx_filename)

    def _create_fighter_readme(self,
                               fighter: PopulationFighter,
                               stats,
                               win_rate: float,
                               output_path: Path) -> None:
        """Create a README.md file with fighter stats and usage info."""
        persistence = self._build_population_persistence_service()
        persistence.create_fighter_readme(fighter, stats, win_rate, output_path)

    def _active_rankings_for_names(self, fighter_names: list[str]) -> list[Any]:
        """Return Elo rankings restricted to the provided fighter names."""
        active_names = set(fighter_names)
        return [stats for stats in self.elo_tracker.get_rankings() if stats.name in active_names]

    def _top_active_stats(self, fighter_names: list[str]):
        """Return the current top-ranked active fighter stats, if any."""
        rankings = self._active_rankings_for_names(fighter_names)
        return rankings[0] if rankings else None

    def _append_generation_summary(
        self,
        *,
        generation: int,
        pre_population_names: list[str],
        post_population_names: list[str],
        champion_before,
        champion_after,
        results: list[dict],
        episodes_per_fighter: int,
        training_seconds: float,
        evaluation_seconds: float,
        saving_seconds: float,
    ) -> None:
        """Persist a structured generation summary for later comparison."""
        total_episodes = sum(int(result.get("episodes", 0)) for result in results)
        mean_reward_overall = (
            float(np.mean([result["mean_reward"] for result in results]))
            if results
            else None
        )
        survivor_names = sorted(set(pre_population_names) & set(post_population_names))
        child_names = sorted(set(post_population_names) - set(pre_population_names))
        generation_record = {
            "timestamp": datetime.now().isoformat(),
            "generation": int(generation),
            "population_size": len(post_population_names),
            "episodes_per_fighter_target": int(episodes_per_fighter),
            "total_episodes_completed": int(total_episodes),
            "training_wall_clock_seconds": float(training_seconds),
            "evaluation_wall_clock_seconds": float(evaluation_seconds),
            "saving_wall_clock_seconds": float(saving_seconds),
            "total_generation_wall_clock_seconds": float(training_seconds + evaluation_seconds + saving_seconds),
            "active_population_before_evolution": list(pre_population_names),
            "active_population_after_evolution": list(post_population_names),
            "survivor_names": survivor_names,
            "survivor_count_carried_forward": len(survivor_names),
            "child_names": child_names,
            "child_count_introduced": len(child_names),
            "champion_before_training": getattr(champion_before, "name", None),
            "champion_before_training_elo": float(champion_before.elo) if champion_before is not None else None,
            "champion_after_evaluation": getattr(champion_after, "name", None),
            "champion_after_evaluation_elo": float(champion_after.elo) if champion_after is not None else None,
            "champion_turnover": (
                champion_before is not None
                and champion_after is not None
                and champion_before.name != champion_after.name
            ),
            "total_matches_so_far": int(self.total_matches),
            "mean_reward_overall": mean_reward_overall,
            "fighter_results": [
                {
                    "fighter": result.get("fighter"),
                    "episodes": int(result.get("episodes", 0)),
                    "mean_reward": float(result.get("mean_reward", 0.0)),
                    "opponent_names": list(result.get("opponent_names", [])),
                }
                for result in results
            ],
            "diversity_metrics": self.elo_tracker.get_diversity_metrics(),
        }
        append_jsonl(self.analysis_dir / "generation_summary.jsonl", generation_record)

    def _record_export_failure(
        self,
        *,
        fighter_name: str,
        export_target: str,
        exception: Exception,
        training_artifacts_saved: bool,
        win_rate: float | None = None,
        elo: float | None = None,
    ) -> None:
        """Persist structured export failure details without interrupting training."""
        append_jsonl(
            self.analysis_dir / "export_failures.jsonl",
            {
                "timestamp": datetime.now().isoformat(),
                "generation": int(self.generation),
                "fighter_name": fighter_name,
                "export_target": export_target,
                "exception_type": exception.__class__.__name__,
                "exception_message": str(exception),
                "training_artifacts_saved": bool(training_artifacts_saved),
                "win_rate": float(win_rate) if win_rate is not None else None,
                "elo": float(elo) if elo is not None else None,
            },
        )

    def _build_training_loop_context(
        self,
        generations: int,
        episodes_per_generation: int,
        evolution_frequency: int,
        keep_top: float,
        mutation_rate: float,
    ) -> PopulationTrainingLoopContext:
        """Build immutable context used by train-loop helper utilities."""
        return PopulationTrainingLoopContext(
            population_size=self.population_size,
            generations=generations,
            episodes_per_generation=episodes_per_generation,
            evolution_frequency=evolution_frequency,
            keep_top=keep_top,
            mutation_rate=mutation_rate,
            replay_recording_frequency=self.replay_recording_frequency,
            replay_matches_per_pair=self.replay_matches_per_pair,
            verbose=self.verbose,
            logger=self.logger,
        )

    def train(self,
             generations: int = 10,
             episodes_per_generation: int = 500,
             evolution_frequency: int = 2,
             base_model_path: Optional[str] = None,
             keep_top: float = 0.5,
             mutation_rate: float = 0.1) -> None:
        """
        Run the full population training loop.

        Args:
            generations: Number of generations to evolve
            episodes_per_generation: Training episodes per generation
            evolution_frequency: Evolve every N generations
            base_model_path: Optional path to pre-trained model for initialization
            keep_top: Fraction of population to keep during evolution
            mutation_rate: Strength of mutations (0.1 = 10% noise added to weights)
        """
        loop_helper = PopulationTrainingLoopHelper(
            self._build_training_loop_context(
                generations=generations,
                episodes_per_generation=episodes_per_generation,
                evolution_frequency=evolution_frequency,
                keep_top=keep_top,
                mutation_rate=mutation_rate,
            )
        )
        loop_helper.print_start_banner(base_model_path=base_model_path)

        # Initialize population if empty
        if not self.population:
            self.initialize_population(base_model_path=base_model_path)

        # Training loop
        for gen in range(generations):
            loop_helper.log_generation_header(current_generation=self.generation + 1)
            pre_population_names = [fighter.name for fighter in self.population]
            champion_before = self._top_active_stats(pre_population_names)

            # Create matchmaking pairs
            pairs = self.create_matchmaking_pairs()

            # Prepare fighter-opponent pairs for parallel training
            fighter_opponent_pairs = loop_helper.build_fighter_opponent_pairs(self.population, pairs)
            episodes_per_fighter = episodes_per_generation // len(self.population)

            # Train all fighters in parallel
            loop_helper.log_generation_training_start(population_size=len(self.population))
            training_started_at = time.time()
            results = self.train_fighters_parallel(
                fighter_opponent_pairs,
                episodes_per_fighter=episodes_per_fighter
            )
            training_seconds = time.time() - training_started_at

            loop_helper.log_generation_training_summary(results)

            # Evaluation matches
            evaluation_started_at = time.time()
            self.run_evaluation_matches(num_matches_per_pair=3)
            evaluation_seconds = time.time() - evaluation_started_at
            champion_after = self._top_active_stats(pre_population_names)

            # Record replays if enabled (based on frequency)
            loop_helper.maybe_record_replays(
                replay_recorder=self.replay_recorder,
                population=self.population,
                generation_zero_based=gen,
                trainer_generation=self.generation,
            )

            # Show leaderboard (only active fighters)
            loop_helper.maybe_show_leaderboard(self.elo_tracker, self.population)

            # Evolution
            if loop_helper.should_evolve(gen):
                self.evolve_population(keep_top=keep_top, mutation_rate=mutation_rate)

            post_population_names = [fighter.name for fighter in self.population]

            # Save checkpoint
            save_started_at = time.time()
            self.save_population()
            saving_seconds = time.time() - save_started_at
            self._append_generation_summary(
                generation=self.generation,
                pre_population_names=pre_population_names,
                post_population_names=post_population_names,
                champion_before=champion_before,
                champion_after=champion_after,
                results=results,
                episodes_per_fighter=episodes_per_fighter,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                saving_seconds=saving_seconds,
            )

            self.generation += 1

        # Final report
        loop_helper.print_and_log_final_report(
            generation=self.generation,
            total_matches=self.total_matches,
            elo_tracker=self.elo_tracker,
            replay_recorder=self.replay_recorder,
        )
