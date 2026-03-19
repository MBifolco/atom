"""
Population-Based Training for Atom Combat

Trains a population of fighters simultaneously, allowing them to learn
from each other and evolve diverse strategies.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
import logging
from datetime import datetime
import time
import random
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Clean relative imports within src package structure
from ....arena import WorldConfig
from ...gym_env import AtomCombatEnv
from .elo_tracker import EloTracker


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
    from src.arena import WorldConfig

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
        algorithm: "ppo" (only PPO supported)

    Returns:
        List of (name, mass, model) tuples
    """
    from stable_baselines3 import PPO

    opponent_models = []
    for opp_name, opp_mass, opp_path in opponent_data:
        opp_model = PPO.load(opp_path, device="cpu")
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
    import numpy as np

    def decide(snapshot):
        # Build 13-dimensional observation to match current gym_env
        position = snapshot["you"]["position"]
        arena_width = snapshot["arena"]["width"]

        # Calculate wall distances
        wall_dist_left = position
        wall_dist_right = arena_width - position

        # Get opponent stance as integer
        opp_stance = snapshot["opponent"]["stance"]
        # Handle JAX arrays - convert to Python type
        if hasattr(opp_stance, '__array__'):
            opp_stance_val = int(opp_stance)
            # JAX arrays contain stance as integer already (0, 1, 2)
            opp_stance_int = opp_stance_val
        else:
            # String stance from Python arena
            stance_map = {"neutral": 0, "extended": 1, "defending": 2, "retracted": 0}  # retracted fallback to neutral
            opp_stance_int = stance_map.get(opp_stance, 0)

        # Estimate recent damage (not available in snapshot, use 0)
        recent_damage = 0.0

        obs = np.array([
            position,  # 0: position
            snapshot["you"]["velocity"],  # 1: velocity
            snapshot["you"]["hp"] / snapshot["you"]["max_hp"],  # 2: hp_norm
            snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"],  # 3: stamina_norm
            snapshot["opponent"]["distance"],  # 4: distance
            snapshot["opponent"]["velocity"],  # 5: rel_velocity (from opponent's perspective)
            snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"],  # 6: opp_hp_norm
            snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"],  # 7: opp_stamina_norm
            arena_width,  # 8: arena_width
            wall_dist_left,  # 9: wall_dist_left
            wall_dist_right,  # 10: wall_dist_right
            opp_stance_int,  # 11: opp_stance_int
            recent_damage  # 12: recent_damage_dealt
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=False)

        acceleration = float(action[0]) * 4.5
        stance_idx = int(action[1])
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
    from src.training.trainers.curriculum_trainer import VmapEnvAdapter
    from src.training.vmap_env_wrapper import VmapEnvWrapper

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
    from src.training.gym_env import AtomCombatEnv

    env_fns = []
    for i in range(n_envs):
        opp_name, opp_mass, opp_model = opponent_models[i % len(opponent_models)]

        # Use closure with default arguments to capture values
        def make_env(opp_m=opp_model, f_mass=fighter_mass, o_mass=opp_mass):
            return Monitor(
                AtomCombatEnv(
                    opponent_decision_func=_create_opponent_decide_func(opp_m),
                    config=config,
                    max_ticks=max_ticks,
                    fighter_mass=f_mass,
                    opponent_mass=o_mass
                ),
                str(Path(logs_dir) / f"{fighter_name}_env_{i}")
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
        algorithm: "ppo" (only PPO supported)
        config_dict: WorldConfig as dictionary
        logs_dir: Directory for logs
        use_vmap: Use JAX vmap for GPU acceleration
        n_vmap_envs: Number of vmap environments (GPU mode)

    Returns:
        Dictionary with training statistics
    """
    # Configure threading for subprocess
    _configure_process_threading()

    # Configure GPU memory for subprocess (if using GPU)
    if use_vmap:
        import os

        # Fix for AMD GPU (RX 6650 XT and similar RDNA2 cards)
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # RDNA2 architecture
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'  # Disable triton

        try:
            import torch
            # Limit each process to use only a fraction of GPU memory
            # Since we default to sequential (1 process), give it most of the memory
            # Leave 20% for system overhead
            memory_fraction = 0.75  # Conservative allocation
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        except Exception as e:
            pass  # Silently continue if torch not available or no GPU

        # Configure ROCm/HIP
        os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use only GPU 0
        os.environ['ROCR_VISIBLE_DEVICES'] = '0'
        os.environ['GPU_DEVICE_ORDINAL'] = '0'

        # Configure JAX memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Match torch fraction
        os.environ['JAX_PLATFORMS'] = 'rocm'  # Force ROCm platform

        try:
            import jax
            # Clear any existing JAX cache from parent process
            jax.clear_caches()
        except:
            pass

    # Required imports (after threading config)
    import numpy as np
    from pathlib import Path
    from stable_baselines3 import PPO
    from src.arena import WorldConfig

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

    import sys
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
                 replay_matches_per_pair: int = 2):
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
        """Generate a unique name for a fighter using funkybob."""
        import random
        import funkybob

        # Set random seed for reproducible names based on index and generation
        seed = index + (generation * 1000)
        random.seed(seed)

        # Create name generator with 2 members and underscore separator
        # (e.g., "Happy_Panda", "Swift_Eagle")
        name_generator = funkybob.RandomNameGenerator(members=2, separator='_')
        name_iter = iter(name_generator)

        # Generate a name
        base_name = next(name_iter)

        # Reset random state to avoid affecting other random operations
        random.seed()

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
            # Convert snapshot to enhanced observation (13 values)
            you_hp_norm = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
            you_stamina_norm = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
            opp_hp_norm = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
            opp_stamina_norm = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

            # Arena width
            arena_width = snapshot["arena"]["width"]

            # Wall distances
            wall_dist_left = snapshot["you"]["position"]
            wall_dist_right = arena_width - snapshot["you"]["position"]

            # Opponent stance as integer (0=neutral, 1=extended, 2=defending)
            opp_stance_hint = snapshot["opponent"].get("stance_hint", snapshot["opponent"].get("stance", "neutral"))
            # Handle JAX arrays - convert to Python type
            if hasattr(opp_stance_hint, '__array__'):
                opp_stance_int = int(opp_stance_hint)
            else:
                # String stance from Python arena
                stance_map = {"neutral": 0, "extended": 1, "defending": 2}
                opp_stance_int = stance_map.get(opp_stance_hint, 0)

            # Recent damage (placeholder - would need tracking)
            recent_damage = 0.0

            obs = np.array([
                snapshot["you"]["position"],        # 0: position
                snapshot["you"]["velocity"],        # 1: velocity
                you_hp_norm,                        # 2: hp_norm
                you_stamina_norm,                   # 3: stamina_norm
                snapshot["opponent"]["distance"],   # 4: distance
                snapshot["opponent"]["velocity"],   # 5: rel_velocity
                opp_hp_norm,                        # 6: opp_hp_norm
                opp_stamina_norm,                   # 7: opp_stamina_norm
                arena_width,                        # 8: arena_width
                wall_dist_left,                     # 9: wall_dist_left
                wall_dist_right,                    # 10: wall_dist_right
                opp_stance_int,                     # 11: opp_stance
                recent_damage                       # 12: recent_damage
            ], dtype=np.float32)

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
        if not fighter_opponent_pairs:
            return []

        # Prepare serializable data for each fighter
        training_tasks = []
        temp_model_paths = {}

        for fighter, opponents in fighter_opponent_pairs:
            # Save fighter model to temp file
            temp_model_path = self.models_dir / f"temp_{fighter.name}_{self.generation}.zip"
            fighter.model.save(temp_model_path)
            temp_model_paths[fighter.name] = temp_model_path

            # Save opponent models to temp files
            opponent_data = []
            for opp in opponents[:self.n_envs_per_fighter]:  # Only need as many as envs
                if opp.name not in temp_model_paths:
                    opp_path = self.models_dir / f"temp_{opp.name}_{self.generation}.zip"
                    opp.model.save(opp_path)
                    temp_model_paths[opp.name] = opp_path
                else:
                    opp_path = temp_model_paths[opp.name]

                opponent_data.append((opp.name, opp.mass, str(opp_path)))

            # Convert config to dictionary (using dataclasses.asdict-like approach)
            # For simplicity, just pass None and use default config in subprocess
            config_dict = None  # Will use WorldConfig() default in subprocess

            task = (
                fighter.name,
                fighter.mass,
                str(temp_model_path),
                opponent_data,
                self.n_envs_per_fighter,
                episodes_per_fighter,
                self.max_ticks,
                self.algorithm,
                config_dict,
                str(self.logs_dir),
                self.use_vmap,
                self.n_vmap_envs
            )
            training_tasks.append(task)

        # Run training in parallel
        results = []

        if self.verbose:
            print(f"  Training {len(training_tasks)} fighters total, {self.n_parallel_fighters} at a time...")
            print(f"  Episodes per fighter: {episodes_per_fighter}")
            if len(training_tasks) > self.n_parallel_fighters:
                batches = (len(training_tasks) + self.n_parallel_fighters - 1) // self.n_parallel_fighters
                print(f"  Will train in {batches} batches of {self.n_parallel_fighters}")
            print(f"  Note: PPO alternates between episode collection (physics) and NN training")
            print()

        # Log to file as well
        self.logger.info(f"Training {len(training_tasks)} fighters, {self.n_parallel_fighters} at a time")
        self.logger.info(f"Episodes per fighter: {episodes_per_fighter}")
        if len(training_tasks) > self.n_parallel_fighters:
            batches = (len(training_tasks) + self.n_parallel_fighters - 1) // self.n_parallel_fighters
            self.logger.info(f"Training in {batches} batches of {self.n_parallel_fighters}")

        # Set multiprocessing start method to 'spawn' to avoid TensorFlow/PyTorch issues
        import multiprocessing as mp
        mp_context = mp.get_context('spawn')

        import time
        training_start_time = time.time()

        executor = None
        try:
            with ProcessPoolExecutor(max_workers=self.n_parallel_fighters, mp_context=mp_context) as executor:
                # Process tasks in batches to avoid memory issues
                future_to_fighter = {}
                future_to_start_time = {}
                task_queue = list(training_tasks)  # Create a queue of tasks
                active_futures = set()

                # Submit initial batch (up to n_parallel_fighters tasks)
                for _ in range(min(self.n_parallel_fighters, len(task_queue))):
                    if task_queue:
                        task = task_queue.pop(0)
                        future = executor.submit(_train_single_fighter_parallel, *task)
                        fighter_name = task[0]
                        future_to_fighter[future] = fighter_name
                        future_to_start_time[future] = time.time()
                        active_futures.add(future)
                        if self.verbose:
                            print(f"    ⏳ Starting: {fighter_name}")
                        self.logger.info(f"Starting training: {fighter_name}")

                # Collect results as they complete
                completed_count = 0
                total_fighters = len(training_tasks)

                if self.verbose:
                    print()
                    print(f"  ⏱️  Training in progress (started {total_fighters} fighters)...")
                    print()

                # Process futures as they complete, including newly submitted ones
                while active_futures:
                    # Wait for the next future to complete
                    done, pending = wait(
                        active_futures,
                        return_when=FIRST_COMPLETED
                    )

                    for future in done:
                        fighter_name = future_to_fighter[future]
                        completed_count += 1
                        elapsed = time.time() - future_to_start_time[future]
                        active_futures.remove(future)

                        try:
                            result = future.result(timeout=300)  # 5 minute timeout per fighter
                            results.append(result)
                            if self.verbose:
                                print(f"    ✅ [{completed_count}/{total_fighters}] {fighter_name}: "
                                      f"{result['episodes']} episodes, mean reward: {result['mean_reward']:.1f}, "
                                      f"time: {elapsed:.1f}s")
                            self.logger.info(f"Completed training [{completed_count}/{total_fighters}] {fighter_name}: "
                                            f"{result['episodes']} episodes, mean reward: {result['mean_reward']:.1f}, "
                                            f"time: {elapsed:.1f}s")

                            # Show remaining fighters every 2 completions
                            if completed_count % 2 == 0 and completed_count < total_fighters:
                                remaining = total_fighters - completed_count
                                overall_elapsed = time.time() - training_start_time
                                avg_time = overall_elapsed / completed_count
                                eta = avg_time * remaining
                                print(f"       💭 {remaining} fighters still training... (ETA: {int(eta)}s)")
                                print()

                        except TimeoutError:
                            self.logger.error(f"Fighter {fighter_name} training timed out after 300s")
                            if self.verbose:
                                print(f"    ❌ [{completed_count}/{total_fighters}] {fighter_name}: Training timed out")
                        except Exception as e:
                            import traceback
                            error_msg = str(e) if str(e) else repr(e)
                            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                            self.logger.error(f"Fighter {fighter_name} training failed: {error_msg}\n{tb_str}")
                            if self.verbose:
                                if not error_msg or error_msg == "":
                                    print(f"    ❌ [{completed_count}/{total_fighters}] {fighter_name}: Subprocess crashed (likely GPU OOM or segfault)")
                                else:
                                    print(f"    ❌ [{completed_count}/{total_fighters}] {fighter_name}: Training failed - {error_msg}")

                        # Submit next task if available
                        if task_queue:
                            task = task_queue.pop(0)
                            future = executor.submit(_train_single_fighter_parallel, *task)
                            new_fighter_name = task[0]
                            future_to_fighter[future] = new_fighter_name
                            future_to_start_time[future] = time.time()
                            active_futures.add(future)
                            if self.verbose:
                                print(f"    ⏳ Starting: {new_fighter_name}")
                            self.logger.info(f"Starting training: {new_fighter_name}")

        except KeyboardInterrupt:
            if self.verbose:
                print("\n\n⚠️  Training interrupted by user (Ctrl+C). Cleaning up worker processes...")
            # Shutdown executor and cancel pending futures
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if self.verbose:
                print("   Worker processes terminated.")
            # Re-raise to allow higher-level cleanup
            raise

        finally:
            # Always clean up temp files, even on interrupt
            for temp_path in temp_model_paths.values():
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass  # Best effort cleanup

        # Reload updated models back into fighters
        for fighter, _ in fighter_opponent_pairs:
            temp_path = temp_model_paths[fighter.name]
            if temp_path.exists():
                # Create a simple env for loading
                env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
                    opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
                    config=self.config,
                    max_ticks=self.max_ticks,
                    fighter_mass=fighter.mass,
                    opponent_mass=70.0
                ))])

                if self.algorithm == "ppo":
                    fighter.model = PPO.load(temp_path, env=env)
                else:
                    fighter.model = SAC.load(temp_path, env=env)

                fighter.training_episodes += episodes_per_fighter

        # Print training summary
        total_training_time = time.time() - training_start_time
        if self.verbose and results:
            print()
            print("  📊 Training Summary:")
            print(f"     Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
            successful = [r for r in results if 'mean_reward' in r]
            if successful:
                mean_rewards = [r['mean_reward'] for r in successful]
                print(f"     Average reward: {sum(mean_rewards)/len(mean_rewards):.1f}")
                print(f"     Best reward: {max(mean_rewards):.1f}")
                print(f"     Worst reward: {min(mean_rewards):.1f}")
            print(f"     Success rate: {len(successful)}/{len(results)}")
            if successful:
                total_episodes = sum(r.get('episodes', 0) for r in successful)
                print(f"     Total episodes: {total_episodes}")
                print(f"     Throughput: {total_episodes/total_training_time:.1f} episodes/sec")

        # Log summary to file
        if results:
            self.logger.info("Training Summary:")
            self.logger.info(f"  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
            successful = [r for r in results if 'mean_reward' in r]
            if successful:
                mean_rewards = [r['mean_reward'] for r in successful]
                self.logger.info(f"  Average reward: {sum(mean_rewards)/len(mean_rewards):.1f}")
                self.logger.info(f"  Best reward: {max(mean_rewards):.1f}")
                self.logger.info(f"  Worst reward: {min(mean_rewards):.1f}")
            self.logger.info(f"  Success rate: {len(successful)}/{len(results)}")
            if successful:
                total_episodes = sum(r.get('episodes', 0) for r in successful)
                self.logger.info(f"  Total episodes: {total_episodes}")
                self.logger.info(f"  Throughput: {total_episodes/total_training_time:.1f} episodes/sec")

        # Note: Temp files are cleaned up in finally block

        return results

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
        self.logger.info(f"Starting evaluation matches with {len(self.population)} fighters")

        if self.verbose:
            print("\n" + "="*60)
            print("EVALUATION MATCHES")
            print("="*60)
            print(f"Running matches between {len(self.population)} fighters")

        # Get all unique pairs
        pairs = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                pairs.append((self.population[i], self.population[j]))

        self.logger.info(f"Created {len(pairs)} unique matchups")

        if len(pairs) == 0:
            self.logger.error("No pairs created for evaluation! Population may be corrupted.")
            if self.verbose:
                print("ERROR: No evaluation pairs created!")
            return

        random.shuffle(pairs)

        # Run matches
        for fighter_a, fighter_b in pairs:
            wins_a = 0
            wins_b = 0
            total_damage_a = 0
            total_damage_b = 0

            for _ in range(num_matches_per_pair):
                # Create environment for the match
                env = AtomCombatEnv(
                    opponent_decision_func=self._get_fighter_decision_func(fighter_b),
                    config=self.config,
                    max_ticks=self.max_ticks,
                    fighter_mass=fighter_a.mass,
                    opponent_mass=fighter_b.mass
                )

                # Run match
                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = fighter_a.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                # Record results
                if info.get("won"):
                    wins_a += 1
                elif info.get("opponent_hp", 0) <= 0:
                    wins_b += 1

                total_damage_a += info.get("episode_damage_dealt", 0)
                total_damage_b += info.get("episode_damage_taken", 0)

                env.close()

            # Determine overall result
            if wins_a > wins_b:
                result = "a_wins"
            elif wins_b > wins_a:
                result = "b_wins"
            else:
                result = "draw"

            # Update ELO ratings
            new_elo_a, new_elo_b = self.elo_tracker.update_ratings(
                fighter_a.name,
                fighter_b.name,
                result,
                total_damage_a / num_matches_per_pair,
                total_damage_b / num_matches_per_pair,
                {"generation": self.generation}
            )

            if self.verbose:
                result_str = f"{fighter_a.name} wins" if result == "a_wins" else \
                           (f"{fighter_b.name} wins" if result == "b_wins" else "Draw")
                print(f"  {fighter_a.name} ({new_elo_a:.0f}) vs {fighter_b.name} ({new_elo_b:.0f}): {result_str}")

            self.total_matches += num_matches_per_pair

    def evolve_population(self, keep_top: float = 0.5, mutation_rate: float = 0.1) -> None:
        """
        Evolve the population by replacing weak fighters with mutations of strong ones.

        Args:
            keep_top: Fraction of population to keep
            mutation_rate: How much to vary the cloned models
        """
        if self.verbose:
            print("\n" + "="*60)
            print("POPULATION EVOLUTION")
            print("="*60)

        rankings = self.elo_tracker.get_rankings()
        keep_count = max(2, int(len(self.population) * keep_top))

        # Get rankings for ONLY the current population fighters
        population_names = {f.name for f in self.population}
        population_rankings = [(i, stats) for i, stats in enumerate(rankings)
                              if stats.name in population_names]

        # Sort current population by their global ELO rank
        population_by_rank = sorted(self.population,
                                   key=lambda f: next((i for i, s in population_rankings if s.name == f.name), 999))

        # Identify survivors and those to replace based on relative ranking in current population
        survivors = population_by_rank[:keep_count]
        to_replace = population_by_rank[keep_count:]

        if self.verbose:
            print(f"  Keeping top {len(survivors)} fighters")
            print(f"  Replacing {len(to_replace)} fighters")

        # Replace weak fighters with mutations of strong ones
        for i, old_fighter in enumerate(to_replace):
            # Select a parent from survivors (weighted by ELO)
            parent = random.choice(survivors)

            # Create new fighter name
            new_name = self._create_fighter_name(
                self.population.index(old_fighter),
                self.generation
            )

            # Vary the mass slightly
            mass_variation = np.random.uniform(-5, 5)
            new_mass = np.clip(parent.mass + mass_variation, *self.mass_range)

            # Clone the parent's model
            env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
                opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
                config=self.config,
                max_ticks=self.max_ticks,
                fighter_mass=new_mass,
                opponent_mass=70.0
            ))])

            if self.algorithm == "ppo":
                # Load parent model - save temporarily if no checkpoint exists
                if parent.last_checkpoint:
                    parent_path = parent.last_checkpoint
                else:
                    # Save temporarily for loading
                    import tempfile
                    temp_dir = Path(tempfile.mkdtemp())
                    parent_path = temp_dir / f"{parent.name}_temp.zip"
                    parent.model.save(parent_path)

                new_model = PPO.load(parent_path, env=env)

                # Clean up temp file if we created one
                if not parent.last_checkpoint:
                    parent_path.unlink()
                    parent_path.parent.rmdir()

                # Apply mutation by slightly changing learning rate
                new_model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))

                # IMPORTANT: Also mutate the actual neural network weights!
                import torch
                with torch.no_grad():
                    for param in new_model.policy.parameters():
                        # Add Gaussian noise to weights
                        # Scale noise by mutation_rate and parameter std
                        noise_scale = mutation_rate * 0.1  # 0.1 is base noise level
                        noise = torch.randn_like(param) * noise_scale
                        param.data.add_(noise)

            else:  # SAC
                # Load parent model - save temporarily if no checkpoint exists
                if parent.last_checkpoint:
                    parent_path = parent.last_checkpoint
                else:
                    # Save temporarily for loading
                    import tempfile
                    temp_dir = Path(tempfile.mkdtemp())
                    parent_path = temp_dir / f"{parent.name}_temp.zip"
                    parent.model.save(parent_path)

                new_model = SAC.load(parent_path, env=env)

                # Clean up temp file if we created one
                if not parent.last_checkpoint:
                    parent_path.unlink()
                    parent_path.parent.rmdir()

                new_model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))

                # IMPORTANT: Also mutate the actual neural network weights!
                import torch
                with torch.no_grad():
                    for param in new_model.policy.parameters():
                        # Add Gaussian noise to weights
                        noise_scale = mutation_rate * 0.1  # 0.1 is base noise level
                        noise = torch.randn_like(param) * noise_scale
                        param.data.add_(noise)

            # Create new fighter
            new_fighter = PopulationFighter(
                name=new_name,
                model=new_model,
                generation=self.generation,
                lineage=f"{parent.name}→{new_name}",
                mass=new_mass
            )

            # Replace in population
            idx = self.population.index(old_fighter)
            self.population[idx] = new_fighter

            # Remove old fighter from ELO tracker before adding new one
            self.elo_tracker.remove_fighter(old_fighter.name)
            # Add new fighter to ELO tracker (starts at default ELO)
            self.elo_tracker.add_fighter(new_name)

            if self.verbose:
                print(f"    Replaced {old_fighter.name} with {new_name} (child of {parent.name})")

        self.logger.info(f"Evolved to generation {self.generation}")

    def save_population(self, min_win_rate: float = None) -> None:
        """
        Save all fighters in the population.

        Also exports qualifying fighters to fighters/AIs/ directory in atom_fight.py format.

        Args:
            min_win_rate: Minimum win rate to export to AIs directory (default: use export_threshold)
        """
        if min_win_rate is None:
            min_win_rate = self.export_threshold
        generation_dir = self.models_dir / f"generation_{self.generation}"
        generation_dir.mkdir(exist_ok=True)

        for fighter in self.population:
            model_path = generation_dir / f"{fighter.name}.zip"
            fighter.model.save(model_path)
            fighter.last_checkpoint = str(model_path)

        # Save ELO rankings
        rankings_file = generation_dir / "rankings.txt"
        with open(rankings_file, 'w') as f:
            f.write(f"Generation {self.generation} Rankings\n")
            f.write("="*60 + "\n")
            for i, stats in enumerate(self.elo_tracker.get_rankings(), 1):
                f.write(f"{i}. {stats.name}: ELO={stats.elo:.0f}, "
                       f"Record={stats.wins}-{stats.losses}-{stats.draws}\n")

        # Export qualifying fighters to AIs directory
        self._export_qualifying_fighters(min_win_rate)

        if self.verbose:
            print(f"\nSaved generation {self.generation} to {generation_dir}")

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

        # Determine project root (where atom_fight.py is)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        ais_dir = project_root / "fighters" / "AIs"
        ais_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0

        for fighter in self.population:
            # Get fighter stats
            stats = next((s for s in rankings if s.name == fighter.name), None)
            if not stats:
                continue

            # Calculate win rate
            total_matches = stats.wins + stats.losses + stats.draws
            if total_matches == 0:
                win_rate = 0.0
            else:
                win_rate = (stats.wins + 0.5 * stats.draws) / total_matches

            # Check if fighter qualifies
            if win_rate >= min_win_rate:
                try:
                    self._export_fighter_to_ais(fighter, stats, win_rate, ais_dir)
                    exported_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to export {fighter.name}: {e}")
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
        # Create fighter directory
        fighter_dir = ais_dir / fighter.name
        fighter_dir.mkdir(parents=True, exist_ok=True)

        # Export model to ONNX
        onnx_path = fighter_dir / f"{fighter.name}.onnx"
        self._export_model_to_onnx(fighter.model, onnx_path)

        # Create Python wrapper
        py_path = fighter_dir / f"{fighter.name}.py"
        self._create_fighter_wrapper(fighter, py_path, onnx_path.name)

        # Create README
        readme_path = fighter_dir / "README.md"
        self._create_fighter_readme(fighter, stats, win_rate, readme_path)

        if self.verbose:
            print(f"  📦 Exported {fighter.name} (WR: {win_rate:.1%}, ELO: {stats.elo:.0f}) to {fighter_dir}")

    def _export_model_to_onnx(self, model, output_path: Path) -> None:
        """Export a Stable-Baselines3 model to ONNX format."""
        import torch

        # Get the policy network
        policy = model.policy

        # Create dummy input (observation space)
        dummy_input = torch.zeros(1, 9, dtype=torch.float32)

        # Export to ONNX
        # Note: Using opset 17 instead of 12 because PyTorch 2.x requires opset 17+
        # Modern browsers and onnxruntime support opset 17
        torch.onnx.export(
            policy,
            dummy_input,
            str(output_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=17
        )

    def _create_fighter_wrapper(self, fighter: PopulationFighter, output_path: Path, onnx_filename: str) -> None:
        """Create a Python wrapper file with decide() function for atom_fight.py."""
        template = f'''"""
{fighter.name} - Trained AI Fighter

Generation: {fighter.generation}
Lineage: {fighter.lineage}
Mass: {fighter.mass:.1f}kg
Training Episodes: {fighter.training_episodes}

Auto-generated wrapper for trained ONNX model.
Compatible with atom_fight.py
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# ONNX model path (relative to this file)
ONNX_MODEL = "{onnx_filename}"

# Global session (loaded once)
_session = None
_stance_names = ["neutral", "extended", "retracted", "defending"]


def _load_session():
    """Load ONNX session (lazy loading)."""
    global _session
    if _session is None:
        model_path = Path(__file__).parent / ONNX_MODEL
        _session = ort.InferenceSession(str(model_path))
    return _session


def decide(snapshot):
    """
    Decision function for trained fighter.

    Args:
        snapshot: Combat snapshot from the arena
            - you: dict with position, velocity, hp, max_hp, stamina, max_stamina
            - opponent: dict with distance, velocity, hp, max_hp, stamina, max_stamina
            - arena: dict with width

    Returns:
        dict with:
            - acceleration: float (-4.5 to +4.5)
            - stance: str ("neutral", "extended", "retracted", "defending")
    """
    session = _load_session()

    # Convert snapshot to observation
    you = snapshot["you"]
    opponent = snapshot["opponent"]
    arena = snapshot["arena"]

    you_hp_norm = you["hp"] / you["max_hp"]
    you_stamina_norm = you["stamina"] / you["max_stamina"]
    opp_hp_norm = opponent["hp"] / opponent["max_hp"]
    opp_stamina_norm = opponent["stamina"] / opponent["max_stamina"]

    obs = np.array([
        you["position"],
        you["velocity"],
        you_hp_norm,
        you_stamina_norm,
        opponent["distance"],
        opponent["velocity"],
        opp_hp_norm,
        opp_stamina_norm,
        arena["width"]
    ], dtype=np.float32).reshape(1, -1)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {{input_name: obs}})

    # Parse action
    # Action space is Box: [acceleration_normalized, stance_selector]
    action = outputs[0][0]
    acceleration_normalized = np.clip(action[0], -1.0, 1.0)
    stance_idx = int(np.clip(action[1], 0, 3))

    # Scale acceleration (max_acceleration = 4.5)
    acceleration = float(acceleration_normalized * 4.5)
    stance = _stance_names[stance_idx]

    return {{"acceleration": acceleration, "stance": stance}}
'''

        with open(output_path, 'w') as f:
            f.write(template)

    def _create_fighter_readme(self,
                               fighter: PopulationFighter,
                               stats,
                               win_rate: float,
                               output_path: Path) -> None:
        """Create a README.md file with fighter stats and usage info."""
        total_matches = stats.wins + stats.losses + stats.draws

        readme = f'''# {fighter.name}

Trained AI Fighter from Population-Based Training

## Stats

- **Generation**: {fighter.generation}
- **Lineage**: {fighter.lineage}
- **Mass**: {fighter.mass:.1f}kg
- **Training Episodes**: {fighter.training_episodes}

### Performance Metrics

- **ELO Rating**: {stats.elo:.0f}
- **Win Rate**: {win_rate:.1%}
- **Record**: {stats.wins}W - {stats.losses}L - {stats.draws}D
- **Total Matches**: {total_matches}

## Usage

This fighter is compatible with `atom_fight.py`:

```bash
# Fight against another AI
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/rusher.py

# Watch the fight in terminal
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/tank.py --watch

# Generate HTML replay
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/balanced.py --html replay.html

# Custom mass (if different from trained mass)
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/rusher.py --mass-a {fighter.mass:.0f}
```

## Files

- `{fighter.name}.py` - Python wrapper with decide() function
- `{fighter.name}.onnx` - ONNX model (neural network weights)
- `README.md` - This file

## Requirements

```bash
pip install onnxruntime numpy
```

## Strategy

This fighter learned its strategy through population-based training, competing against
other evolving AI fighters. Its behavior emerged from reinforcement learning rather than
being hand-coded.

**Training Algorithm**: {self.algorithm.upper()}
**Population Size**: {self.population_size}
**Generation**: {self.generation}

## Notes

- The fighter was trained at {fighter.mass:.1f}kg mass. Performance may vary with different masses.
- Win rate of {win_rate:.1%} was achieved against the training population.
- The ONNX model requires `onnxruntime` to run inference.
'''

        with open(output_path, 'w') as f:
            f.write(readme)

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
        if self.verbose:
            print("\n" + "="*80)
            print("STARTING POPULATION TRAINING")
            print("="*80)
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {generations}")
            print(f"Episodes per Generation: {episodes_per_generation}")
            print(f"Evolution Frequency: Every {evolution_frequency} generations")
            if base_model_path:
                print(f"Base Model: {base_model_path}")
            print("="*80)

        # Initialize population if empty
        if not self.population:
            self.initialize_population(base_model_path=base_model_path)

        # Training loop
        for gen in range(generations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"GENERATION {self.generation + 1}/{generations}")
                print(f"{'='*80}")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"GENERATION {self.generation + 1}/{generations}")
            self.logger.info(f"{'='*80}")

            # Create matchmaking pairs
            pairs = self.create_matchmaking_pairs()

            # Prepare fighter-opponent pairs for parallel training
            fighter_opponent_pairs = []
            for fighter in self.population:
                # Select opponents for this fighter
                opponents = [
                    opp for pair in pairs
                    for opp in [pair[0], pair[1]]
                    if pair[0] == fighter or pair[1] == fighter
                    if opp != fighter
                ]

                # If no opponents from pairs, use random selection
                if not opponents:
                    opponents = [f for f in self.population if f != fighter]
                    opponents = random.sample(opponents, min(3, len(opponents)))

                fighter_opponent_pairs.append((fighter, opponents))

            # Train all fighters in parallel
            if self.verbose:
                print(f"\n🚀 Training {len(self.population)} fighters in parallel...")
            self.logger.info(f"Training {len(self.population)} fighters in parallel")

            results = self.train_fighters_parallel(
                fighter_opponent_pairs,
                episodes_per_fighter=episodes_per_generation // len(self.population)
            )

            if self.verbose and results:
                mean_reward_overall = np.mean([r['mean_reward'] for r in results])
                total_episodes = sum(r['episodes'] for r in results)
                print(f"  ✓ Completed {total_episodes} episodes total, "
                      f"mean reward: {mean_reward_overall:.1f}")
            if results:
                mean_reward_overall = np.mean([r['mean_reward'] for r in results])
                total_episodes = sum(r['episodes'] for r in results)
                self.logger.info(f"Generation training complete: {total_episodes} episodes total, "
                                f"mean reward: {mean_reward_overall:.1f}")

            # Evaluation matches
            self.run_evaluation_matches(num_matches_per_pair=3)

            # Record replays if enabled (based on frequency)
            if self.replay_recorder and (gen + 1) % self.replay_recording_frequency == 0:
                try:
                    self.replay_recorder.record_population_generation(
                        generation=self.generation + 1,
                        fighters=self.population,
                        num_matches_per_pair=self.replay_matches_per_pair
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to record replays for generation {self.generation + 1}: {e}")

            # Show leaderboard (only active fighters)
            if self.verbose:
                active_names = [f.name for f in self.population]
                self.elo_tracker.print_leaderboard(active_only=active_names)

            # Evolution
            if (gen + 1) % evolution_frequency == 0 and gen < generations - 1:
                self.evolve_population(keep_top=keep_top, mutation_rate=mutation_rate)

            # Save checkpoint
            self.save_population()

            self.generation += 1

        # Final report
        if self.verbose:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Total Generations: {self.generation}")
            print(f"Total Matches: {self.total_matches}")
            print("\nFinal Rankings (All Time):")
            self.elo_tracker.print_leaderboard()  # Show all fighters for historical record

            # Show diversity metrics
            metrics = self.elo_tracker.get_diversity_metrics()
            print(f"\nPopulation Diversity:")
            print(f"  ELO Spread: {metrics['elo_range']:.0f}")
            print(f"  ELO Std Dev: {metrics['elo_std']:.1f}")
            if 'win_rate_std' in metrics:
                print(f"  Win Rate Variance: {metrics['win_rate_std']:.3f}")

        # Log final report to file
        self.logger.info("="*80)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"Total Generations: {self.generation}")
        self.logger.info(f"Total Matches: {self.total_matches}")

        # Log diversity metrics
        metrics = self.elo_tracker.get_diversity_metrics()
        self.logger.info("Population Diversity:")
        self.logger.info(f"  ELO Spread: {metrics['elo_range']:.0f}")
        self.logger.info(f"  ELO Std Dev: {metrics['elo_std']:.1f}")
        if 'win_rate_std' in metrics:
            self.logger.info(f"  Win Rate Variance: {metrics['win_rate_std']:.3f}")

        # Save replay index if recording was enabled
        if self.replay_recorder:
            self.replay_recorder.save_replay_index()

        self.logger.info("Training complete")
        self.logger.info(f"Final generation: {self.generation}")