"""
Atom Combat - PPO Trainer

Train fighters using Proximal Policy Optimization with mixed opponents.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Callable, Optional
import importlib.util
import logging
from datetime import datetime
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ...gym_env import AtomCombatEnv
from src.arena import WorldConfig


class VerboseLoggingCallback(BaseCallback):
    """Callback to log detailed training information to file."""

    def __init__(self, log_path: str, opponent_names: List[str], verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.opponent_names = opponent_names
        self.file_logger = None  # Use different name than 'logger'
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Set up logging when training starts."""
        # Create logger
        self.file_logger = logging.getLogger('atom_training')
        self.file_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.file_logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.file_logger.addHandler(fh)
        # Prevent propagation to root logger
        self.file_logger.propagate = False

        # Log training start
        self.file_logger.info("=" * 80)
        self.file_logger.info("TRAINING SESSION STARTED")
        self.file_logger.info(f"Opponents: {', '.join(self.opponent_names)}")
        self.file_logger.info(f"Number of environments: {self.training_env.num_envs}")
        self.file_logger.info("=" * 80)

    def _on_step(self) -> bool:
        """Log episode information."""
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            if "episode" in info:
                self.episode_count += 1
                opponent_idx = env_idx % len(self.opponent_names)
                opponent_name = self.opponent_names[opponent_idx]

                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                # Log episode details
                self.file_logger.debug(
                    f"Episode {self.episode_count} | "
                    f"Env {env_idx} | "
                    f"Opponent: {opponent_name} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Length: {episode_length} ticks"
                )

                # Log additional info if available
                if "fighter_hp" in info:
                    self.file_logger.debug(
                        f"  Final HP: Fighter={info.get('fighter_hp', 'N/A'):.1f}, "
                        f"Opponent={info.get('opponent_hp', 'N/A'):.1f}"
                    )
                if "episode_damage_dealt" in info:
                    self.file_logger.debug(
                        f"  Damage: Dealt={info.get('episode_damage_dealt', 0):.1f}, "
                        f"Taken={info.get('episode_damage_taken', 0):.1f}"
                    )
                if "won" in info:
                    self.file_logger.debug(f"  Result: {'WIN' if info['won'] else 'LOSS'}")

                # Log reward breakdown if available
                if "reward_breakdown" in info and info["reward_breakdown"] is not None:
                    breakdown = info["reward_breakdown"]
                    self.file_logger.debug(
                        f"  Reward Breakdown: "
                        f"Terminal={breakdown['terminal']:+.1f}, "
                        f"Damage={breakdown['damage']:+.1f}, "
                        f"Proximity={breakdown['proximity']:+.1f}, "
                        f"Inaction={breakdown['inaction']:+.1f} | "
                        f"Total={breakdown['total']:+.1f}"
                    )

        return True


class ProgressCallback(BaseCallback):
    """Callback to track and display training progress."""

    def __init__(self, check_freq: int = 1000, opponent_names: List[str] = None, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.opponent_names = opponent_names or []
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.episodes_per_opponent = {name: 0 for name in self.opponent_names}

    def _on_step(self) -> bool:
        """Called at each step."""
        # Collect episode info
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                # Track opponent
                if self.opponent_names:
                    opponent_idx = env_idx % len(self.opponent_names)
                    opponent_name = self.opponent_names[opponent_idx]
                    self.episodes_per_opponent[opponent_name] += 1

        # Print progress every check_freq steps
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    improvement = "⬆️"
                else:
                    improvement = ""

                print(f"Step {self.n_calls:,} | Episodes: {len(self.episode_rewards)} | "
                      f"Mean Reward: {mean_reward:.1f} {improvement} | "
                      f"Mean Length: {mean_length:.0f} ticks")

                # Show opponent distribution
                if self.opponent_names and len(self.opponent_names) > 1:
                    total = sum(self.episodes_per_opponent.values())
                    if total > 0:
                        opponent_stats = ", ".join(
                            f"{name}: {count}" for name, count in self.episodes_per_opponent.items()
                        )
                        print(f"  Opponents: {opponent_stats}")

        return True


class EpisodeLimitCallback(BaseCallback):
    """Stop training when target episode count is reached."""

    def __init__(self, target_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Check if we've reached target episodes."""
        # Count completed episodes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1

        # Stop if we've reached target
        if self.episode_count >= self.target_episodes:
            if self.verbose > 0:
                print(f"\n🎯 Reached target: {self.episode_count} episodes")
            return False  # Stop training

        return True  # Continue training


class PlateauStoppingCallback(BaseCallback):
    """Stop training when performance plateaus."""

    def __init__(
        self,
        check_freq: int = 5000,
        patience: int = 5,
        min_delta: float = 1.0,
        verbose: int = 1
    ):
        """
        Args:
            check_freq: How often to check for plateau
            patience: How many checks without improvement before stopping
            min_delta: Minimum change in mean reward to count as improvement
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.checks_without_improvement = 0
        self.episode_rewards = []
        self.episode_wins = []
        self.episode_damages = []

    def _on_step(self) -> bool:
        """Check for plateau with multiple metrics."""
        # Collect episode metrics
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                # Track wins and damage dealt
                if "won" in info:
                    self.episode_wins.append(1 if info["won"] else 0)
                if "episode_damage_dealt" in info:
                    self.episode_damages.append(info["episode_damage_dealt"])

        # Check every check_freq steps
        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 100:
            mean_reward = np.mean(self.episode_rewards[-100:])
            win_rate = np.mean(self.episode_wins[-100:]) if self.episode_wins else 0.0
            mean_damage = np.mean(self.episode_damages[-100:]) if self.episode_damages else 0.0

            # Detect combat avoidance (no damage dealt)
            if mean_damage < 1.0:
                if self.verbose > 0:
                    print(f"  ⚠️  WARNING: Combat avoidance detected! Damage dealt: {mean_damage:.1f}")
                    print(f"      Fighter may be stuck in local minimum (avoiding all combat)")
                # Don't count as improvement if avoiding combat
                self.checks_without_improvement += 1
            # Real improvement requires BOTH better reward AND engagement
            elif mean_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = mean_reward
                self.checks_without_improvement = 0
                if self.verbose > 0:
                    print(f"  📈 Improvement: Reward={mean_reward:.1f}, WinRate={win_rate:.1%}, Damage={mean_damage:.1f}")
            else:
                self.checks_without_improvement += 1
                if self.verbose > 0:
                    print(f"  📊 No improvement ({self.checks_without_improvement}/{self.patience}) - Reward={mean_reward:.1f}, WinRate={win_rate:.1%}")

                # Stop if plateaued
                if self.checks_without_improvement >= self.patience:
                    if self.verbose > 0:
                        print(f"  🛑 Training stopped: Performance plateaued")
                        print(f"      Final stats: Reward={mean_reward:.1f}, WinRate={win_rate:.1%}, Damage={mean_damage:.1f}")
                    return False  # Stop training

        return True  # Continue training


def load_opponent_function(filepath: str) -> Callable:
    """Load opponent decision function from Python file."""
    path = Path(filepath)
    spec = importlib.util.spec_from_file_location("opponent_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def make_env(
    opponent_func: Callable,
    config: WorldConfig,
    max_ticks: int,
    fighter_mass: float,
    opponent_mass: float,
    seed: int
):
    """Create a single environment (for parallel training)."""
    def _init():
        env = AtomCombatEnv(
            opponent_decision_func=opponent_func,
            config=config,
            max_ticks=max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=opponent_mass,
            seed=seed
        )
        env = Monitor(env)  # Wrap in Monitor for episode stats
        return env
    return _init


def train_fighter(
    opponent_files: List[str],
    output_path: str,
    episodes: int = 10000,
    n_envs: int = 10,
    fighter_mass: float = 70.0,
    opponent_mass: float = 75.0,
    max_ticks: int = 400,
    checkpoint_freq: int = 10000,
    patience: int = 5,
    verbose: bool = True,
    tensorboard_log: Optional[str] = None,
    continue_from_model: Optional[str] = None
):
    """
    Train a fighter using PPO with mixed opponents.

    Args:
        opponent_files: List of paths to opponent Python files
        output_path: Where to save the trained model
        episodes: Target number of episodes (approximate)
        n_envs: Number of parallel environments (CPU cores to use)
        fighter_mass: Mass of the fighter being trained
        opponent_mass: Mass of opponents
        max_ticks: Max ticks per episode
        checkpoint_freq: Save checkpoint every N steps
        patience: Patience for plateau detection
        verbose: Show progress
        tensorboard_log: Optional TensorBoard log directory
        continue_from_model: Path to existing model to continue training (prevents forgetting)
    """
    print("=" * 60)
    print("ATOM COMBAT - FIGHTER TRAINING".center(60))
    print("=" * 60)

    # Load opponent decision functions
    print(f"\nLoading {len(opponent_files)} opponent(s)...")
    opponents = []
    opponent_names = []
    for filepath in opponent_files:
        opponent_func = load_opponent_function(filepath)
        opponents.append(opponent_func)
        opponent_name = Path(filepath).stem  # Filename without extension
        opponent_names.append(opponent_name)
        print(f"  ✓ {opponent_name}")

    # Create world config
    config = WorldConfig()

    # Set up training log file in logs/ directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_obj = Path(output_path)

    # Extract model name - for curriculum it's the parent directory name
    # E.g., "outputs/parzival/level1.zip" -> "log_parzival_level_1_timestamp.log"
    model_name = output_path_obj.parent.name
    level_name = output_path_obj.stem

    # Format: log_modelname_level_X_timestamp.log for curriculum
    if model_name not in ["outputs", "training"] and level_name.startswith("level"):
        # Extract level number from "level1" -> "1"
        level_num = level_name.replace("level", "")
        log_name = f"{model_name}_level_{level_num}"
    elif model_name not in ["outputs", "training"]:
        log_name = f"{model_name}_{level_name}"
    else:
        log_name = level_name

    # Logs go in the model's directory (e.g., outputs/parzival_1.0.12/logs/)
    logs_dir = output_path_obj.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / f"{log_name}_{timestamp}.log"

    print(f"\nTraining Configuration:")
    print(f"  Fighter mass: {fighter_mass}kg")
    print(f"  Opponent mass: {opponent_mass}kg")
    print(f"  Target episodes: {episodes:,}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Max ticks/episode: {max_ticks}")
    print(f"  Log file: {log_path.name}")

    # Calculate total timesteps
    # Use generous estimate to ensure we don't run out before hitting episode limit
    # With fast fights (~65 ticks), we need more timesteps than the old estimate
    avg_ticks_per_episode = 100  # Conservative estimate (fights often finish in 60-100 ticks)
    total_timesteps = episodes * avg_ticks_per_episode * 2  # 2x buffer to ensure we hit episode limit first

    print(f"  Timesteps budget: ~{total_timesteps:,} (will stop at {episodes} episodes)")

    # Create parallel environments
    print(f"\nCreating {n_envs} parallel environments...")

    # Assign opponents to environments (cycle through opponents)
    env_fns = []
    env_opponent_map = []
    for i in range(n_envs):
        opponent_idx = i % len(opponents)
        opponent_func = opponents[opponent_idx]
        opponent_name = opponent_names[opponent_idx]
        env_opponent_map.append((i, opponent_name))

        env_fn = make_env(
            opponent_func=opponent_func,
            config=config,
            max_ticks=max_ticks,
            fighter_mass=fighter_mass,
            opponent_mass=opponent_mass,
            seed=42 + i  # Different seed per env
        )
        env_fns.append(env_fn)

    # Show environment-opponent mapping
    print("  Environment → Opponent mapping:")
    for env_id, opp_name in env_opponent_map:
        print(f"    Env {env_id} → {opp_name}")

    # Use SubprocVecEnv for true parallelism across CPU cores
    vec_env = SubprocVecEnv(env_fns)

    print("  ✓ Environments ready")

    # Create or load PPO model
    if continue_from_model:
        print(f"\nLoading existing model from: {continue_from_model}")
        print("  (Continuing training to prevent catastrophic forgetting)")
        model = PPO.load(continue_from_model, env=vec_env)
        # Lower learning rate for continual learning to reduce forgetting
        model.learning_rate = 1e-4
        print(f"  ✓ Model loaded - learning rate lowered to {model.learning_rate} for stability")
    else:
        print("\nInitializing fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            tensorboard_log=tensorboard_log,
            learning_rate=1e-4,  # Lower rate to reduce catastrophic forgetting
            n_steps=2048 // n_envs,  # Adjust for number of envs
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
        )
        print("  ✓ Fresh model created")

    # Create callbacks
    callbacks = []

    # Verbose logging to file
    verbose_log_cb = VerboseLoggingCallback(
        log_path=str(log_path),
        opponent_names=opponent_names,
        verbose=1
    )
    callbacks.append(verbose_log_cb)

    if verbose:
        progress_cb = ProgressCallback(
            check_freq=1000,
            opponent_names=opponent_names
        )
        callbacks.append(progress_cb)

    # Episode limit (primary stopping condition)
    episode_limit_cb = EpisodeLimitCallback(
        target_episodes=episodes,
        verbose=1 if verbose else 0
    )
    callbacks.append(episode_limit_cb)

    # Plateau stopping (secondary - backup if stalling before episode limit)
    plateau_cb = PlateauStoppingCallback(
        check_freq=5000,
        patience=patience,
        min_delta=1.0,
        verbose=1 if verbose else 0
    )
    callbacks.append(plateau_cb)

    # Checkpoint saving in outputs/checkpoints/
    if "outputs" in str(output_path):
        checkpoint_dir = Path(output_path).parent / "checkpoints"
    else:
        checkpoint_dir = Path(output_path).parent / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="fighter_checkpoint"
    )
    callbacks.append(checkpoint_cb)

    # Train
    print("\nStarting training...")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False  # We have custom progress
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    print("-" * 60)

    # Save final model
    print(f"\nSaving trained model to: {output_path}")
    model.save(output_path)

    # Also save as ONNX (we'll add this next)
    print("  ✓ Model saved")

    # Close environments
    vec_env.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE".center(60))
    print("=" * 60)

    return model


def train_curriculum(
    output_base: str,
    episodes_per_level: int = 2000,
    n_envs: int = 10,
    fighter_mass: float = 70.0,
    opponent_mass: float = 70.0,  # Changed to same mass by default
    max_ticks: int = 400,
    graduation_tests: int = 20,
    verbose: bool = True,
    create_wrappers: bool = False
):
    """
    Train a fighter through curriculum learning.

    Automatically progresses through opponents as graduation criteria are met.

    Args:
        output_base: Base name for outputs (e.g., "my_fighter") - should be a Path object
        episodes_per_level: Episodes to train at each level
        n_envs: Parallel environments
        fighter_mass: Fighter mass
        opponent_mass: Opponent mass
        max_ticks: Max ticks per episode
        graduation_tests: Number of test matches to determine graduation
        verbose: Show progress
        create_wrappers: Create .py wrappers at each level
    """
    # Convert to Path if string and create model directory
    if isinstance(output_base, str):
        output_base = Path(output_base)

    # Create a directory for this model to keep files organized
    # E.g., "outputs/parzival" -> "outputs/parzival/" directory
    model_dir = output_base
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Model directory: {model_dir}")

    # Define curriculum progression
    # Full curriculum matching OPPONENT_PROGRESSION.md
    curriculum = [
        {
            "name": "Level 1: Training Dummy",
            "opponent": "../fighters/training_opponents/training_dummy.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.95,  # Should be 100%, use 95% for safety
            "description": "Stationary target - learn basic mechanics"
        },
        {
            "name": "Level 2: Wanderer",
            "opponent": "../fighters/training_opponents/wanderer.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.90,  # 90%+ per progression doc
            "description": "Random movement - learn positioning"
        },
        {
            "name": "Level 3: Bumbler",
            "opponent": "../fighters/training_opponents/bumbler.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.80,  # 80%+ per progression doc
            "description": "Poor fighter - learn timing and stamina management"
        },
        {
            "name": "Level 4: Novice",
            "opponent": "../fighters/training_opponents/novice.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.70,  # 70%+ per progression doc
            "description": "Competent fundamentals - exploit predictability"
        },
        {
            "name": "Level 5: Rusher",
            "opponent": "../fighters/examples/rusher.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.60,  # 60%+ per progression doc
            "description": "Aggressive pressure - counter aggression"
        },
        {
            "name": "Level 6: Tank",
            "opponent": "../fighters/examples/tank.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.55,  # 55%+ per progression doc
            "description": "Defensive counter-puncher - break through defense"
        },
        {
            "name": "Level 7: Balanced",
            "opponent": "../fighters/examples/balanced.py",
            "episodes": episodes_per_level,
            "win_rate_required": 0.50,  # 50%+ per progression doc (competitive)
            "description": "Adaptive tactician - handle adaptive opponent"
        },
    ]

    print("=" * 60)
    print("CURRICULUM LEARNING - PROGRESSIVE TRAINING".center(60))
    print("=" * 60)
    print(f"\n📚 Training through {len(curriculum)} levels")
    print(f"   Episodes per level: {episodes_per_level}")
    print(f"   Graduation tests: {graduation_tests} matches")
    print()

    # Track overall progress
    current_model_path = None

    for level_idx, level in enumerate(curriculum, 1):
        print("\n" + "=" * 60)
        print(f"🎯 {level['name']}".center(60))
        print("=" * 60)
        print(f"Opponent: {Path(level['opponent']).stem}")
        print(f"Goal: {level['description']}")
        print(f"Graduation: {level['win_rate_required']*100:.0f}% win rate")
        print()

        # Output paths for this level (all in model directory)
        level_model_path = model_dir / f"level{level_idx}.zip"
        level_onnx_path = model_dir / f"level{level_idx}.onnx"
        wrapper_path = model_dir / f"level{level_idx}.py"

        # Determine if we should continue from previous level (prevents catastrophic forgetting)
        continue_from = None
        if level_idx > 1:
            # Load previous level's model to continue training
            prev_level_path = model_dir / f"level{level_idx-1}.zip"
            if prev_level_path.exists():
                continue_from = str(prev_level_path)
                print(f"📚 Continuing from Level {level_idx-1} model (prevents forgetting)")
            else:
                print(f"⚠️  Warning: Previous level model not found, starting fresh")

        # Build opponent list: include ALL previous levels + current level
        # This prevents catastrophic forgetting by mixing in previous opponents
        opponent_files = []
        for prev_idx in range(level_idx + 1):
            opponent_files.append(curriculum[prev_idx]['opponent'])

        if len(opponent_files) > 1:
            print(f"📚 Training against {len(opponent_files)} opponents (prevents forgetting)")
            print(f"   Mix: {', '.join(Path(f).stem for f in opponent_files)}")

        # Train at this level
        print(f"Training for {level['episodes']} episodes...")
        model = train_fighter(
            opponent_files=opponent_files,  # All previous + current level opponents
            output_path=str(level_model_path),
            episodes=level['episodes'],
            n_envs=n_envs,
            fighter_mass=fighter_mass,
            opponent_mass=opponent_mass,
            max_ticks=max_ticks,
            checkpoint_freq=10000,
            patience=5,
            verbose=verbose,
            tensorboard_log=None,
            continue_from_model=continue_from
        )

        # Export to ONNX
        print(f"\nExporting to ONNX: {level_onnx_path}")
        from ...onnx_fighter import export_to_onnx, create_fighter_wrapper
        try:
            export_to_onnx(str(level_model_path), str(level_onnx_path))
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")

        # Create wrapper if requested
        if create_wrappers and level_onnx_path.exists():
            print(f"Creating fighter wrapper: {wrapper_path}")
            create_fighter_wrapper(str(level_onnx_path), str(wrapper_path))

        # Graduation test
        print(f"\n" + "=" * 60)
        print(f"🎓 GRADUATION TEST".center(60))
        print("=" * 60)

        # Load the fighter once
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Import our trained fighter
        if create_wrappers and wrapper_path.exists():
            fighter_spec = importlib.util.spec_from_file_location("trained_fighter", str(wrapper_path))
            fighter_mod = importlib.util.module_from_spec(fighter_spec)
            fighter_spec.loader.exec_module(fighter_mod)
            fighter_func = fighter_mod.decide
        else:
            # Use ONNX directly
            from ...onnx_fighter import ONNXFighter
            onnx_fighter = ONNXFighter(str(level_onnx_path))
            fighter_func = onnx_fighter.decide

        # Run test matches
        from src.arena import WorldConfig
        from src.orchestrator import MatchOrchestrator

        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=max_ticks, record_telemetry=False)

        # Test against ALL previous levels + current level to check for catastrophic forgetting
        all_passed = True

        for test_level_idx in range(level_idx):
            test_level = curriculum[test_level_idx]
            required_rate = test_level['win_rate_required']
            print(f"\nTesting against {Path(test_level['opponent']).stem} (need {required_rate*100:.0f}% win rate)...")

            # Import opponent
            opponent_module = importlib.util.spec_from_file_location("opponent", test_level['opponent'])
            opponent_mod = importlib.util.module_from_spec(opponent_module)
            opponent_module.loader.exec_module(opponent_mod)
            opponent_func = opponent_mod.decide

            wins = 0
            losses = 0

            for test_num in range(graduation_tests):
                fighter_spec = {"name": "AI", "mass": fighter_mass, "position": 2.0}
                opponent_spec = {"name": "Opponent", "mass": opponent_mass, "position": 10.0}

                result = orchestrator.run_match(
                    fighter_spec,
                    opponent_spec,
                    fighter_func,
                    opponent_func,
                    seed=1000 + test_level_idx * 100 + test_num
                )

                # Only count actual wins (draws are NOT wins)
                if "AI" in result.winner:
                    wins += 1
                else:
                    losses += 1

                # Show progress after each test
                current_win_rate = wins / (test_num + 1)
                if "AI" in result.winner:
                    result_emoji = "✅"  # Win
                elif result.winner == "draw":
                    result_emoji = "🤝"  # Draw (counted as loss)
                else:
                    result_emoji = "❌"  # Loss
                progress_emoji = "✅" if current_win_rate >= required_rate else "⏳" if current_win_rate >= required_rate * 0.8 else "⚠️"
                print(f"  Test {test_num + 1}/{graduation_tests}: {result_emoji} ({wins}W-{losses}L, {current_win_rate*100:.1f}%, need {required_rate*100:.0f}%) {progress_emoji}")

            win_rate = wins / graduation_tests

            print(f"\n  Final Results: {wins}W-{losses}L ({win_rate*100:.1f}%) - Required: {required_rate*100:.0f}%")

            if win_rate >= required_rate:
                print(f"  ✅ Still beating {Path(test_level['opponent']).stem}")
            else:
                print(f"  ❌ FORGOT how to beat {Path(test_level['opponent']).stem}!")
                print(f"     This is 'catastrophic forgetting' - model degraded on previous opponents")
                all_passed = False

        if all_passed:
            print(f"\n✅ PASSED - Graduated from {level['name']}")
            print(f"   (Retained ability to beat all previous opponents)")
            current_model_path = level_model_path
        else:
            print(f"\n❌ FAILED - Catastrophic forgetting detected")
            print(f"\n⚠️  Stopping curriculum at {level['name']}")
            print(f"   Recommend: Use smaller learning rate or add previous opponents to training mix")
            break

        # Small pause between levels
        time.sleep(2)

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE".center(60))
    print("=" * 60)

    if current_model_path:
        print(f"\n✓ Final model: {current_model_path}")
        return current_model_path
    else:
        print(f"\n⚠️  Training incomplete - stopped at {level['name']}")
        return None
