"""
Population-Based Training for Atom Combat

Trains a population of fighters simultaneously, allowing them to learn
from each other and evolve diverse strategies.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import logging
from datetime import datetime
import time
import random
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Clean relative imports within src package structure
from ....arena import WorldConfig
from ...gym_env import AtomCombatEnv
from .elo_tracker import EloTracker


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
    use_vmap: bool = False,
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
    # IMPORTANT: Set thread limits BEFORE importing TensorFlow/PyTorch
    # This prevents each process from using all CPU cores for BLAS/MKL operations
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'

    import numpy as np
    from pathlib import Path
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from src.arena import WorldConfig
    from src.training.gym_env import AtomCombatEnv

    # Reconstruct WorldConfig
    if config_dict is None:
        config = WorldConfig()  # Use default
    else:
        config = WorldConfig(**config_dict)

    # Load opponent models on CPU to save GPU memory
    # Opponent inference is fast on CPU (~1-2ms), GPU is reserved for physics
    opponent_models = []
    for opp_name, opp_mass, opp_path in opponent_data:
        if algorithm == "ppo":
            opp_model = PPO.load(opp_path, device="cpu")
        else:
            opp_model = SAC.load(opp_path, device="cpu")
        opponent_models.append((opp_name, opp_mass, opp_model))

    # Create environments based on mode
    if use_vmap:
        # GPU mode: Configure JAX memory to prevent OOM
        # Note: This is set PER SUBPROCESS, so divide by expected parallel processes
        import os
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        # Reduce memory fraction for parallel training - each subprocess gets a fraction
        # With 4 parallel fighters: 0.2/fighter = 80% total GPU usage (safe)
        # With 2 parallel fighters: 0.35/fighter = 70% total GPU usage
        # Set memory fraction conservatively (each subprocess doesn't know total count)
        # For safety, assume up to 4 parallel processes
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'  # 20% per process × 4 = 80% total
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

        # GPU mode: Use VmapEnvWrapper with opponent models
        from src.training.trainers.curriculum_trainer import VmapEnvAdapter
        from src.training.vmap_env_wrapper import VmapEnvWrapper

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

        vec_env = VmapEnvAdapter(vmap_env)
    else:
        # CPU mode: Use DummyVecEnv with decision functions
        def create_opponent_decide_func(model):
            def decide(snapshot):
                obs = np.array([
                    snapshot["you"]["position"],
                    snapshot["you"]["velocity"],
                    snapshot["you"]["hp"] / snapshot["you"]["max_hp"],
                    snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"],
                    snapshot["opponent"]["distance"],
                    snapshot["opponent"]["velocity"],
                    snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"],
                    snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"],
                    snapshot["arena"]["width"]
                ], dtype=np.float32)

                action, _ = model.predict(obs, deterministic=False)

                acceleration = float(action[0]) * 4.5
                stance_idx = int(action[1])
                stances = ["neutral", "extended", "retracted", "defending"]
                stance = stances[min(stance_idx, 3)]

                return {"acceleration": acceleration, "stance": stance}

            return decide

        env_fns = []
        for i in range(n_envs):
            opp_name, opp_mass, opp_model = opponent_models[i % len(opponent_models)]

            # Use closure with default arguments to capture values
            def make_env(opp_m=opp_model, f_mass=fighter_mass, o_mass=opp_mass):
                return Monitor(
                    AtomCombatEnv(
                        opponent_decision_func=create_opponent_decide_func(opp_m),
                        config=config,
                        max_ticks=max_ticks,
                        fighter_mass=f_mass,
                        opponent_mass=o_mass
                    ),
                    str(Path(logs_dir) / f"{fighter_name}_env_{i}")
                )

            env_fns.append(make_env)

        vec_env = DummyVecEnv(env_fns)

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
                 use_vmap: bool = False,
                 n_vmap_envs: int = 250):
        """
        Initialize the population trainer.

        Args:
            population_size: Number of fighters in the population
            config: World configuration
            algorithm: "ppo" or "sac"
            output_dir: Directory for saving models and logs
            n_envs_per_fighter: Parallel environments per fighter (CPU mode)
            max_ticks: Maximum ticks per episode
            mass_range: Range of masses for fighter variety
            verbose: Whether to print progress
            export_threshold: Minimum win rate to export fighters to AIs directory
            n_parallel_fighters: Number of fighters to train in parallel (default: cpu_count - 1)
            use_vmap: Use JAX vmap for GPU-accelerated training (77x speedup)
            n_vmap_envs: Number of vmap environments for GPU mode (default: 250)
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

        # Parallel training configuration
        if n_parallel_fighters is None:
            if use_vmap:
                # GPU mode: Limit parallelism to avoid GPU OOM
                # Each process uses GPU, so use fewer parallel processes
                n_parallel_fighters = 2  # Conservative: 2 parallel GPU processes
                if verbose:
                    print(f"⚠️  GPU mode: Automatically limiting parallel fighters to {n_parallel_fighters}")
                    print(f"   (prevents GPU out-of-memory errors)")
                    print(f"   💡 Tip: If GPU memory usage is low (<50%), try --n-parallel-fighters 4")
            else:
                # CPU mode: Use more parallelism
                n_parallel_fighters = max(1, multiprocessing.cpu_count() - 1)
        elif use_vmap and verbose:
            print(f"🚀 GPU mode: Using {n_parallel_fighters} parallel fighters")

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
            # Convert snapshot to observation
            obs = np.array([
                snapshot["you"]["position"],
                snapshot["you"]["velocity"],
                snapshot["you"]["hp"] / snapshot["you"]["max_hp"],
                snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"],
                snapshot["opponent"]["distance"],
                snapshot["opponent"]["velocity"],
                snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"],
                snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"],
                snapshot["arena"]["width"]
            ], dtype=np.float32)

            # Get action from model
            action, _ = fighter.model.predict(obs, deterministic=False)

            # Convert continuous action to game action
            acceleration = float(action[0]) * 4.5  # Scale from [-1, 1] to [-4.5, 4.5]
            stance_idx = int(action[1])
            stances = ["neutral", "extended", "retracted", "defending"]
            stance = stances[min(stance_idx, 3)]

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
                # Submit all tasks
                future_to_fighter = {}
                future_to_start_time = {}
                for task in training_tasks:
                    future = executor.submit(_train_single_fighter_parallel, *task)
                    fighter_name = task[0]
                    future_to_fighter[future] = fighter_name
                    future_to_start_time[future] = time.time()
                    if self.verbose:
                        print(f"    ⏳ Starting: {fighter_name}")
                    self.logger.info(f"Starting training: {fighter_name}")

                # Collect results as they complete
                completed_count = 0
                total_fighters = len(future_to_fighter)

                if self.verbose:
                    print()
                    print(f"  ⏱️  Training in progress (started {total_fighters} fighters)...")
                    print()

                for future in as_completed(future_to_fighter):
                    fighter_name = future_to_fighter[future]
                    completed_count += 1
                    elapsed = time.time() - future_to_start_time[future]

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
                new_model = PPO.load(
                    parent.last_checkpoint if parent.last_checkpoint else parent.model,
                    env=env
                )
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
                new_model = SAC.load(
                    parent.last_checkpoint if parent.last_checkpoint else parent.model,
                    env=env
                )
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

        self.logger.info("Training complete")
        self.logger.info(f"Final generation: {self.generation}")