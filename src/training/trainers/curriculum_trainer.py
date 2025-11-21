"""
Curriculum-Based Training System for Atom Combat

Implements progressive training using test dummies before advancing to
hardcoded fighters and eventually population-based training.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
import logging
from datetime import datetime
import time
import json
from dataclasses import dataclass, field
from enum import Enum

# Use Stable Baselines3 with PyTorch (JAX is in physics engine)
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import test dummy loader
import importlib.util

# Level 3: JAX vmap support
from ..vmap_env_wrapper import VmapEnvWrapper
from ...arena import WorldConfig


class VmapEnvAdapter(VecEnv):
    """Adapter to make VmapEnvWrapper compatible with SBX VecEnv interface."""
    def __init__(self, vmap_env):
        self.vmap_env = vmap_env
        super().__init__(
            num_envs=vmap_env.n_envs,
            observation_space=vmap_env.observation_space,
            action_space=vmap_env.action_space
        )
        self.metadata = {"render_modes": []}
        self._supports_set_opponent = False  # vmap doesn't support dynamic opponent switching

    def reset(self):
        obs, _ = self.vmap_env.reset()
        return obs

    def step_async(self, actions):
        """Store actions for step_wait."""
        self.actions = actions

    def step_wait(self):
        """Execute stored actions and return results."""
        obs, rewards, dones, truncated, infos = self.vmap_env.step(self.actions)
        # SBX expects combined done
        dones = np.logical_or(dones, truncated)
        return obs, rewards, dones, infos

    def close(self):
        pass  # VmapEnvWrapper doesn't need cleanup

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Required for VecEnv interface."""
        return False

    def get_attr(self, attr_name, indices=None):
        """Get attribute from environments."""
        return [getattr(self.vmap_env, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute (not supported)."""
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Override to handle set_opponent calls gracefully."""
        if method_name == "set_opponent":
            # This should no longer happen since we recreate VmapEnvWrapper for new levels
            # If you see this warning, it means the curriculum trainer isn't recreating envs properly
            if not hasattr(self, '_warned_about_opponent_switch'):
                print("⚠️  WARNING: Attempted to call set_opponent() on VmapEnvWrapper.")
                print("    This should not happen - vmap environments should be recreated for new levels.")
                print("    If you see this, there's a bug in curriculum_trainer.py:advance_level()")
                self._warned_about_opponent_switch = True
            return
        # For other methods, try to call them (though vmap doesn't support much)
        return [None] * self.num_envs


class DifficultyLevel(Enum):
    """Training difficulty levels."""
    FUNDAMENTALS = "fundamentals"      # Level 1: Stationary targets
    BASIC_SKILLS = "basic_skills"      # Level 2: Simple movements
    INTERMEDIATE = "intermediate"       # Level 3: Distance/stamina management
    ADVANCED = "advanced"               # Level 4: Behavioral fighters
    EXPERT = "expert"                   # Level 5: Hardcoded fighters
    POPULATION = "population"           # Level 6: Population training


@dataclass
class CurriculumLevel:
    """Represents a training level in the curriculum."""
    name: str
    difficulty: DifficultyLevel
    opponents: List[str]  # Paths to opponent files
    min_episodes: int = 100
    graduation_win_rate: float = 0.7
    graduation_episodes: int = 20  # Must maintain win rate for this many episodes
    description: str = ""


@dataclass
class TrainingProgress:
    """Tracks a fighter's progress through the curriculum."""
    current_level: int = 0
    episodes_at_level: int = 0
    wins_at_level: int = 0
    recent_episodes: List[bool] = field(default_factory=list)  # Win/loss history
    graduated_levels: List[str] = field(default_factory=list)
    total_episodes: int = 0
    total_wins: int = 0
    start_time: float = field(default_factory=time.time)


class CurriculumCallback(BaseCallback):
    """Callback for tracking curriculum training progress."""

    def __init__(self, curriculum_trainer, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_trainer = curriculum_trainer
        self.episode_rewards = []
        self.episode_wins = []
        self.recent_reward_components = []
        self.last_rollout_time = None
        self.last_train_time = None

    def _on_rollout_start(self) -> None:
        """Called before collecting rollouts."""
        import time
        self.last_rollout_time = time.time()

        # Initialize rollout counter if not exists
        if not hasattr(self, 'rollout_count'):
            self.rollout_count = 0
        self.rollout_count += 1

        # For SAC: Only log every 50 rollouts to reduce spam
        # For PPO: Always log (only happens once per iteration)
        should_log = (
            self.curriculum_trainer.algorithm == 'ppo' or
            self.rollout_count % 50 == 1
        )

        if self.verbose and should_log:
            print(f"\n🎮 Collecting rollouts (GPU physics)...", flush=True)

    def _on_rollout_end(self) -> None:
        """Called after rollouts are collected, before training."""
        import time

        # Same logging logic as _on_rollout_start
        should_log = (
            self.curriculum_trainer.algorithm == 'ppo' or
            self.rollout_count % 50 == 1
        )

        if self.last_rollout_time:
            rollout_duration = time.time() - self.last_rollout_time
            if self.verbose and should_log:
                print(f"✅ Rollouts collected in {rollout_duration:.2f}s", flush=True)

        self.last_train_time = time.time()
        if self.verbose and should_log:
            print(f"🧠 Training neural network (CPU)...", flush=True)

    def _on_training_end(self) -> None:
        """Called after training completes."""
        import time

        # Same logging logic as other methods
        should_log = (
            self.curriculum_trainer.algorithm == 'ppo' or
            getattr(self, 'rollout_count', 0) % 50 == 0
        )

        if self.last_train_time:
            train_duration = time.time() - self.last_train_time
            if self.verbose and should_log:
                print(f"✅ Neural network trained in {train_duration:.2f}s\n", flush=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # Track episode completion
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)

                # Check if won (use the "won" key from environment)
                won = info.get("won", False)
                self.episode_wins.append(won)

                # Track reward breakdown if available
                if "reward_breakdown" in info:
                    self.recent_reward_components.append(info["reward_breakdown"])
                    if len(self.recent_reward_components) > 100:
                        self.recent_reward_components.pop(0)

                # Update curriculum progress with additional info
                self.curriculum_trainer.update_progress(won, reward, info)

                # Check for level graduation
                if self.curriculum_trainer.should_graduate():
                    self.curriculum_trainer.advance_level()

                    # Stop training if curriculum is complete
                    if self.curriculum_trainer.progress.current_level >= len(self.curriculum_trainer.curriculum):
                        self.curriculum_trainer.logger.info("Curriculum complete - stopping training early")
                        return False

        return True


class CurriculumTrainer:
    """
    Manages curriculum-based training for Atom Combat fighters.

    Progressively trains fighters through increasingly difficult opponents:
    1. Stationary test dummies (fundamentals)
    2. Movement test dummies (basic skills)
    3. Distance/stamina managers (intermediate)
    4. Behavioral fighters (advanced)
    5. Hardcoded fighters (expert)
    6. Population-based training
    """

    def __init__(self,
                 algorithm: str = "ppo",
                 output_dir: str = "outputs/curriculum",
                 n_envs: int = 4,
                 max_ticks: int = 250,
                 verbose: bool = True,
                 device: str = "auto",
                 use_vmap: bool = False,
                 debug: bool = False,
                 record_replays: bool = False,
                 replay_matches_per_opponent: int = 3):
        """
        Initialize the curriculum trainer.

        Args:
            algorithm: "ppo" (only PPO supported)
            output_dir: Directory for saving models and logs
            n_envs: Number of parallel environments (or vmap batch size)
            max_ticks: Maximum ticks per episode
            verbose: Whether to print progress
            device: Device to use for training ("cpu", "cuda", or "auto")
            use_vmap: Use JAX vmap for parallel environments (Level 3/4 optimization)
            record_replays: Whether to record fight replays for montage
            replay_matches_per_opponent: Number of evaluation matches per opponent for replay recording
        """
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.n_envs = n_envs
        self.max_ticks = max_ticks
        self.verbose = verbose
        self.device = device
        self.use_vmap = use_vmap
        self.debug = debug
        self.record_replays = record_replays
        self.replay_matches_per_opponent = replay_matches_per_opponent

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize curriculum
        self.curriculum = self._build_curriculum()
        self.progress = TrainingProgress()

        # Model and environments
        self.model = None
        self.envs = None

        # Replay recorder (if enabled)
        self.replay_recorder = None
        if self.record_replays:
            from ..replay_recorder import ReplayRecorder
            self.replay_recorder = ReplayRecorder(
                output_dir=str(self.output_dir),
                max_ticks=max_ticks,
                verbose=verbose
            )

        # Setup logging
        self._setup_logging()

    def _build_curriculum(self) -> List[CurriculumLevel]:
        """Build the training curriculum."""
        test_dummy_dir = Path("fighters/test_dummies")
        example_dir = Path("fighters/examples")

        curriculum = []

        # Level 1: Fundamentals (stationary dummies)
        curriculum.append(CurriculumLevel(
            name="Fundamentals",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=[
                str(test_dummy_dir / "atomic/stationary_neutral.py"),
                str(test_dummy_dir / "atomic/stationary_extended.py"),
                str(test_dummy_dir / "atomic/stationary_defending.py"),
            ],
            min_episodes=200,
            graduation_win_rate=0.9,  # Should easily beat stationary targets
            graduation_episodes=50,  # Increased from 10 to prevent lucky streaks
            description="Learn basic attacking and stance usage against stationary targets"
        ))

        # Level 2: Basic Skills (simple movements)
        curriculum.append(CurriculumLevel(
            name="Basic Skills",
            difficulty=DifficultyLevel.BASIC_SKILLS,
            opponents=[
                str(test_dummy_dir / "atomic/approach_slow.py"),
                str(test_dummy_dir / "atomic/flee_always.py"),
                str(test_dummy_dir / "atomic/shuttle_slow.py"),
                str(test_dummy_dir / "atomic/shuttle_medium.py"),
                str(test_dummy_dir / "atomic/circle_left.py"),
                str(test_dummy_dir / "atomic/circle_right.py"),
            ],
            min_episodes=300,
            graduation_win_rate=0.88,  # High standards maintained
            graduation_episodes=50,
            description="Learn pursuit, evasion, and predictive movement"
        ))

        # Level 3: Intermediate (distance/stamina management)
        curriculum.append(CurriculumLevel(
            name="Intermediate",
            difficulty=DifficultyLevel.INTERMEDIATE,
            opponents=[
                str(test_dummy_dir / "atomic/distance_keeper_1m.py"),
                str(test_dummy_dir / "atomic/distance_keeper_3m.py"),
                str(test_dummy_dir / "atomic/distance_keeper_5m.py"),
                str(test_dummy_dir / "atomic/stamina_waster.py"),
                str(test_dummy_dir / "atomic/stamina_cycler.py"),
                str(test_dummy_dir / "atomic/stamina_efficient.py"),
                str(test_dummy_dir / "atomic/charge_on_approach.py"),
                str(test_dummy_dir / "atomic/wall_hugger_left.py"),
                str(test_dummy_dir / "atomic/wall_hugger_right.py"),
            ],
            min_episodes=400,
            graduation_win_rate=0.85,  # Maintained high standards
            graduation_episodes=50,
            description="Learn spacing control, resource management, and wall combat"
        ))

        # Level 4: Advanced (behavioral fighters)
        curriculum.append(CurriculumLevel(
            name="Advanced",
            difficulty=DifficultyLevel.ADVANCED,
            opponents=[
                str(test_dummy_dir / "behavioral/perfect_defender.py"),
                str(test_dummy_dir / "behavioral/burst_attacker.py"),
                str(test_dummy_dir / "behavioral/perfect_kiter.py"),
                str(test_dummy_dir / "behavioral/stamina_optimizer.py"),
                str(test_dummy_dir / "behavioral/wall_fighter.py"),
                str(test_dummy_dir / "behavioral/adaptive_fighter.py"),
            ],
            min_episodes=500,
            graduation_win_rate=0.83,  # Staying near 85% standards
            graduation_episodes=50,
            description="Learn complex strategies and counter-strategies"
        ))

        # Level 5: Expert (hardcoded fighters)
        curriculum.append(CurriculumLevel(
            name="Expert",
            difficulty=DifficultyLevel.EXPERT,
            opponents=[
                str(example_dir / "tank.py"),
                str(example_dir / "rusher.py"),
                str(example_dir / "balanced.py"),
                str(example_dir / "grappler.py"),
                str(example_dir / "zoner.py"),
                str(example_dir / "dodger.py"),
                str(example_dir / "berserker.py"),
            ],
            min_episodes=600,
            graduation_win_rate=0.80,  # Excellence required even at final level
            graduation_episodes=50,
            description="Master combat against diverse expert strategies"
        ))

        return curriculum

    def _setup_logging(self):
        """Setup logging for curriculum training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"curriculum_training_{timestamp}.log"

        self.logger = logging.getLogger('curriculum_trainer')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

        if self.verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info("="*80)
        self.logger.info("CURRICULUM TRAINING INITIALIZED")
        self.logger.info(f"Algorithm: {self.algorithm}")
        self.logger.info(f"Training Backend: SB3 (PyTorch) with JAX physics")
        self.logger.info(f"GPU Acceleration: {'Enabled (vmap)' if self.use_vmap else 'Disabled'}")
        self.logger.info(f"Curriculum Levels: {len(self.curriculum)}")
        self.logger.info("="*80)

    def load_opponent(self, opponent_path: str) -> Callable:
        """Load an opponent decision function from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location("opponent", opponent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.decide
        except Exception as e:
            self.logger.error(f"Failed to load opponent {opponent_path}: {e}")
            # Return a dummy opponent that does nothing
            return lambda s: {"acceleration": 0, "stance": "neutral"}

    def create_env(self, opponent_path: str, env_id: int = 0) -> Any:
        """Create a single environment with the specified opponent."""
        from ..gym_env import AtomCombatEnv

        opponent_func = self.load_opponent(opponent_path)

        return AtomCombatEnv(
            opponent_decision_func=opponent_func,
            max_ticks=self.max_ticks,
            fighter_mass=70.0,
            opponent_mass=70.0
        )

    def create_envs_for_level(self, level: CurriculumLevel) -> Any:
        """Create parallel environments for the current level."""

        if self.use_vmap:
            # Level 3/4: Use JAX vmap for GPU-accelerated parallelization
            # Now supports multiple opponents distributed across environments!
            opponent_paths = level.opponents

            if self.verbose:
                print(f"🚀 Creating {self.n_envs} parallel JAX vmap environments...")
                print(f"   Opponents: {len(opponent_paths)} different types")
                if len(opponent_paths) > 1:
                    envs_per_opponent = self.n_envs // len(opponent_paths)
                    print(f"   Distribution: ~{envs_per_opponent} envs per opponent")
                    for i, path in enumerate(opponent_paths):
                        print(f"     {i+1}. {Path(path).stem}")

            vmap_env = VmapEnvWrapper(
                n_envs=self.n_envs,
                opponent_paths=opponent_paths,  # Pass list of paths instead of single func
                config=WorldConfig(),
                max_ticks=self.max_ticks,
                fighter_mass=70.0,
                opponent_mass=70.0,
                seed=42,
                debug=self.debug
            )

            # Wrap with adapter for SBX compatibility
            adapted_env = VmapEnvAdapter(vmap_env)

            # Wrap with NaN checker to catch numerical instabilities
            return VecCheckNan(adapted_env, raise_exception=True, warn_once=True)

        else:
            # Level 1/2: Use DummyVecEnv with multiple opponents
            # Create environments with different opponents from the level
            env_fns = []
            for i in range(self.n_envs):
                opponent_idx = i % len(level.opponents)
                opponent_path = level.opponents[opponent_idx]

                # Use Monitor with unique level-based filenames to avoid file handle conflicts
                # The allow_early_resets=True is important for level transitions
                level_name = level.name.replace(" ", "_").lower()
                monitor_file = str(self.logs_dir / f"{level_name}_env_{i}")

                env_fn = lambda opp_path=opponent_path, mfile=monitor_file: Monitor(
                    self.create_env(opp_path),
                    mfile,
                    allow_early_resets=True
                )
                env_fns.append(env_fn)

            # ALWAYS use DummyVecEnv for curriculum training
            # SubprocVecEnv has too many pickle/process issues during curriculum progression
            dummy_env = DummyVecEnv(env_fns)

            # Wrap with NaN checker to catch numerical instabilities
            return VecCheckNan(dummy_env, raise_exception=True, warn_once=True)

    def initialize_model(self):
        """Initialize or load the RL model."""
        if self.algorithm == "ppo":
            # IMPORTANT: Force CPU for MlpPolicy as GPU is 1.4x slower for simple MLPs
            # The main speedup comes from JAX physics simulation (77x), not NN training
            # See: https://github.com/DLR-RM/stable-baselines3/issues/1245
            actual_device = "cpu" if self.device == "auto" else self.device

            # Debug: Print device being used
            if self.verbose:
                print(f"Initializing PPO with device: {actual_device} (forced CPU for MlpPolicy)")

            self.model = PPO(
                "MlpPolicy",
                self.envs,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                max_grad_norm=0.5,  # Gradient clipping to prevent NaN
                ent_coef=0.01,  # Entropy coefficient for exploration
                vf_coef=0.5,  # Value function coefficient
                normalize_advantage=True,  # Normalize advantages
                verbose=2 if self.verbose else 0,  # Set to 2 for detailed NN training logs
                tensorboard_log=str(self.logs_dir / "tensorboard"),
                device=actual_device
            )
        elif self.algorithm == "sac":
            # Same CPU optimization for SAC with MlpPolicy
            actual_device = "cpu" if self.device == "auto" else self.device

            # Debug: Print device being used
            if self.verbose:
                print(f"Initializing SAC with device: {actual_device} (forced CPU for MlpPolicy)")

            self.model = SAC(
                "MlpPolicy",
                self.envs,
                learning_rate=3e-4,
                buffer_size=50000,      # Smaller buffer for faster initial learning
                learning_starts=100,    # Start learning much earlier (was 1000)
                batch_size=64,          # Smaller batch size like PPO (was 256)
                tau=0.01,               # Faster target network updates (was 0.005)
                gamma=0.99,
                ent_coef='auto',        # Auto-tune entropy for exploration
                target_update_interval=1,  # Update target network every step
                gradient_steps=1,       # One gradient step per env step
                verbose=1 if self.verbose else 0,
                tensorboard_log=str(self.logs_dir / "tensorboard"),
                device=actual_device
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def update_progress(self, won: bool, reward: float = 0, info: dict = None):
        """Update training progress for the current episode."""
        self.progress.episodes_at_level += 1
        self.progress.total_episodes += 1

        if won:
            self.progress.wins_at_level += 1
            self.progress.total_wins += 1

        # Track recent episodes for graduation check
        self.progress.recent_episodes.append(won)
        if len(self.progress.recent_episodes) > self.get_current_level().graduation_episodes:
            self.progress.recent_episodes.pop(0)

        # Track recent rewards for analysis
        if not hasattr(self.progress, 'recent_rewards'):
            self.progress.recent_rewards = []
            self.progress.recent_reward_breakdowns = []

        self.progress.recent_rewards.append(reward)
        if len(self.progress.recent_rewards) > 100:
            self.progress.recent_rewards.pop(0)

        # Track reward breakdown if available
        if info and "reward_breakdown" in info:
            self.progress.recent_reward_breakdowns.append(info["reward_breakdown"])
            if len(self.progress.recent_reward_breakdowns) > 20:
                self.progress.recent_reward_breakdowns.pop(0)

        # Log progress every 100 episodes
        if self.progress.episodes_at_level % 100 == 0:
            level = self.get_current_level()
            overall_win_rate = self.progress.wins_at_level / max(1, self.progress.episodes_at_level)

            # Recent win rate (if we have enough data)
            if len(self.progress.recent_episodes) >= level.graduation_episodes:
                recent_win_rate = sum(self.progress.recent_episodes) / len(self.progress.recent_episodes)
                mean_reward = np.mean(self.progress.recent_rewards) if self.progress.recent_rewards else 0

                self.logger.info(
                    f"Progress: Episode {self.progress.episodes_at_level} | "
                    f"Overall WR: {overall_win_rate:.1%} | "
                    f"Recent WR: {recent_win_rate:.1%} "
                    f"(need {level.graduation_win_rate:.1%}) | "
                    f"Mean Reward: {mean_reward:.1f}"
                )

                # Log detailed reward breakdown every 100 episodes
                if self.progress.recent_reward_breakdowns:
                    # Average the components
                    avg_breakdown = {}
                    for breakdown in self.progress.recent_reward_breakdowns:
                        for key, value in breakdown.items():
                            if key not in avg_breakdown:
                                avg_breakdown[key] = []
                            avg_breakdown[key].append(value)

                    # Calculate averages
                    reward_summary = []
                    for key, values in avg_breakdown.items():
                        if key != "total":
                            avg_val = np.mean(values)
                            if abs(avg_val) > 0.1:  # Only show significant components
                                reward_summary.append(f"{key}={avg_val:+.1f}")

                    if reward_summary:
                        self.logger.info(f"  Reward Components: {', '.join(reward_summary)}")

                # Log win/loss/draw counts
                recent_wins = sum(self.progress.recent_episodes)
                recent_losses = len(self.progress.recent_episodes) - recent_wins
                self.logger.info(
                    f"  Last {len(self.progress.recent_episodes)} episodes: "
                    f"{recent_wins} wins, {recent_losses} losses"
                )
            else:
                self.logger.info(
                    f"Progress: Episode {self.progress.episodes_at_level} | "
                    f"Overall WR: {overall_win_rate:.1%}"
                )

    def get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level."""
        if self.progress.current_level >= len(self.curriculum):
            return self.curriculum[-1]  # Stay at highest level
        return self.curriculum[self.progress.current_level]

    def should_graduate(self) -> bool:
        """Check if the fighter should graduate to the next level."""
        # Don't graduate if already completed all curriculum levels
        if self.progress.current_level >= len(self.curriculum):
            return False

        level = self.get_current_level()

        # Check minimum episodes
        if self.progress.episodes_at_level < level.min_episodes:
            return False

        # Check recent win rate
        if len(self.progress.recent_episodes) < level.graduation_episodes:
            return False

        recent_win_rate = sum(self.progress.recent_episodes) / len(self.progress.recent_episodes)
        recent_wins = sum(self.progress.recent_episodes)
        overall_win_rate = self.progress.wins_at_level / max(1, self.progress.episodes_at_level)

        # Only check recent win rate for graduation
        # The overall win rate requirement creates an inverted difficulty curve
        # where early levels become harder due to accumulated losses
        recent_passed = recent_win_rate >= level.graduation_win_rate

        # Optional: Add a minimum overall win rate to prevent extreme cases
        # But keep it low and consistent across levels
        min_overall = 0.5  # 50% minimum overall win rate for any level
        overall_passed = overall_win_rate >= min_overall

        # Debug logging for graduation decisions
        will_graduate = recent_passed and overall_passed

        # Only log if graduating OR every 100 episodes when recent passes (to reduce spam)
        should_log = will_graduate or (recent_passed and self.progress.episodes_at_level % 100 == 0)

        if should_log:
            status = "✅ PASSED" if will_graduate else "❌ FAILED (overall too low)"
            self.logger.info(f"🎓 GRADUATION CHECK {status}")
            self.logger.info(f"   Recent wins: {recent_wins}/{len(self.progress.recent_episodes)}")
            self.logger.info(f"   Recent WR: {recent_win_rate:.2%} (need {level.graduation_win_rate:.1%}) {'✓' if recent_passed else '✗'}")
            self.logger.info(f"   Overall WR: {overall_win_rate:.2%} (need {min_overall:.1%}) {'✓' if overall_passed else '✗'}")
            self.logger.info(f"   Episodes at level: {self.progress.episodes_at_level}")

        return will_graduate

    def advance_level(self):
        """Advance to the next curriculum level."""
        current = self.get_current_level()

        self.logger.info("="*60)
        self.logger.info(f"GRADUATED from {current.name}!")
        self.logger.info(f"Episodes: {self.progress.episodes_at_level}")
        self.logger.info(f"Win Rate: {self.progress.wins_at_level / max(1, self.progress.episodes_at_level):.2%}")
        self.logger.info("="*60)

        # Record replays if enabled (BEFORE moving to next level)
        if self.replay_recorder:
            try:
                self.replay_recorder.record_curriculum_stage(
                    stage_name=current.name,
                    level_num=self.progress.current_level + 1,  # 1-indexed
                    model=self.model,
                    opponent_paths=current.opponents,
                    num_matches_per_opponent=self.replay_matches_per_opponent
                )
            except Exception as e:
                self.logger.warning(f"Failed to record replays for {current.name}: {e}")

        # Record graduation
        self.progress.graduated_levels.append(current.name)

        # Move to next level
        self.progress.current_level += 1
        self.progress.episodes_at_level = 0
        self.progress.wins_at_level = 0
        self.progress.recent_episodes = []

        # Log the reset for debugging
        self.logger.info("📊 METRICS RESET FOR NEW LEVEL:")
        self.logger.info(f"   Episodes at level: {self.progress.episodes_at_level}")
        self.logger.info(f"   Wins at level: {self.progress.wins_at_level}")
        self.logger.info(f"   Recent episodes buffer: {len(self.progress.recent_episodes)} episodes")

        # Check if completed curriculum
        if self.progress.current_level >= len(self.curriculum):
            self.logger.info("🎉 CURRICULUM COMPLETED! 🎉")
            self.on_curriculum_complete()
        else:
            # Setup new level
            new_level = self.get_current_level()
            self.logger.info(f"Starting Level {self.progress.current_level + 1}: {new_level.name}")
            self.logger.info(f"Description: {new_level.description}")
            self.logger.info(f"Opponents: {len(new_level.opponents)} different types")
            self.logger.info(f"Graduation Requirements: {new_level.graduation_win_rate:.0%} win rate over {new_level.graduation_episodes} episodes")

            if self.use_vmap:
                # For vmap: Create new VmapEnvWrapper with new opponents
                # We can't dynamically switch opponents in vmap, so recreate the wrapper
                self.logger.info("📦 Recreating vmap environments for new level...")

                # Create new environment
                self.envs = self.create_envs_for_level(new_level)

                # Update model's environment reference
                self.model.set_env(self.envs)

                # Note: Old environment will be garbage collected automatically
                # Don't manually delete attributes as the model may still hold references
            else:
                # For CPU: Switch opponents in existing environments (avoids closing/recreating VecEnv)
                # This prevents Monitor file handle issues during level transitions
                for env_idx in range(self.n_envs):
                    opponent_idx = env_idx % len(new_level.opponents)
                    opponent_path = new_level.opponents[opponent_idx]
                    opponent_func = self.load_opponent(opponent_path)

                    # Use env_method to call set_opponent() on each environment
                    self.envs.env_method('set_opponent', opponent_func, indices=[env_idx])

    def on_curriculum_complete(self):
        """Called when the entire curriculum is completed."""
        elapsed = time.time() - self.progress.start_time

        self.logger.info("\n" + "="*80)
        self.logger.info("CURRICULUM TRAINING COMPLETE!")
        self.logger.info("="*80)
        self.logger.info(f"Total Episodes: {self.progress.total_episodes}")
        self.logger.info(f"Total Wins: {self.progress.total_wins}")
        self.logger.info(f"Overall Win Rate: {self.progress.total_wins / max(1, self.progress.total_episodes):.2%}")
        self.logger.info(f"Training Time: {elapsed/3600:.1f} hours")
        self.logger.info(f"Graduated Levels: {', '.join(self.progress.graduated_levels)}")

        # Save final model
        final_model_path = self.models_dir / "curriculum_graduate.zip"
        self.model.save(final_model_path)
        self.logger.info(f"Final model saved to: {final_model_path}")

        # Save replay index if recording was enabled
        if self.replay_recorder:
            self.replay_recorder.save_replay_index()

        # Save training report
        self.save_training_report()

    def save_training_report(self):
        """Save a detailed training report."""
        report = {
            "algorithm": self.algorithm,
            "total_episodes": self.progress.total_episodes,
            "total_wins": self.progress.total_wins,
            "overall_win_rate": self.progress.total_wins / max(1, self.progress.total_episodes),
            "graduated_levels": self.progress.graduated_levels,
            "training_time_hours": (time.time() - self.progress.start_time) / 3600,
            "curriculum_levels": [
                {
                    "name": level.name,
                    "difficulty": level.difficulty.value,
                    "num_opponents": len(level.opponents),
                    "graduation_win_rate": level.graduation_win_rate,
                    "min_episodes": level.min_episodes
                }
                for level in self.curriculum
            ],
            "timestamp": datetime.now().isoformat()
        }

        report_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Training report saved to: {report_path}")

    def train(self, total_timesteps: int = 1_000_000):
        """
        Train the fighter through the curriculum.

        Args:
            total_timesteps: Total training timesteps across all levels
        """
        self.logger.info("Starting curriculum training...")

        # Start with first level
        level = self.get_current_level()
        self.logger.info(f"Level 1: {level.name}")
        self.logger.info(f"Description: {level.description}")

        # Create environments for first level
        self.envs = self.create_envs_for_level(level)

        # Initialize model
        self.initialize_model()

        # Create callback
        callback = CurriculumCallback(self, verbose=self.verbose)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=self.verbose
        )

        # Close environments
        if self.envs:
            self.envs.close()

        self.logger.info("Training complete!")

        # Check if we graduated at least the first level
        if len(self.progress.graduated_levels) == 0:
            # Didn't graduate even Level 1
            level = self.curriculum[0]

            # Calculate current performance
            if len(self.progress.recent_episodes) > 0:
                current_win_rate = sum(self.progress.recent_episodes) / len(self.progress.recent_episodes)
            else:
                current_win_rate = 0.0

            overall_win_rate = self.progress.wins_at_level / max(1, self.progress.episodes_at_level)

            self.logger.error("="*80)
            self.logger.error("CURRICULUM TRAINING FAILED")
            self.logger.error("="*80)
            self.logger.error(f"Failed to graduate Level 1: {level.name}")
            self.logger.error(f"Episodes completed: {self.progress.episodes_at_level}")
            self.logger.error(f"Overall win rate: {overall_win_rate:.1%}")
            self.logger.error(f"Recent win rate: {current_win_rate:.1%} (need {level.graduation_win_rate:.1%})")
            self.logger.error(f"Required: {level.graduation_win_rate:.1%} win rate over {level.graduation_episodes} episodes")
            self.logger.error("")
            self.logger.error("Suggestions:")
            self.logger.error(f"  - Increase timesteps (current: {total_timesteps:,})")
            self.logger.error(f"  - Try 3-5x more timesteps for Level 1 graduation")
            self.logger.error("="*80)

            raise RuntimeError(
                f"Curriculum training failed to graduate Level 1 after {total_timesteps:,} timesteps. "
                f"Win rate: {overall_win_rate:.1%} (need {level.graduation_win_rate:.1%}). "
                f"Increase --timesteps parameter."
            )

        # Save the model only if graduated
        final_model_path = self.models_dir / "curriculum_graduate.zip"
        self.model.save(final_model_path)
        self.logger.info(f"Model saved to: {final_model_path}")

    def save_checkpoint(self, name: str = None):
        """Save a model checkpoint."""
        if not self.model:
            return

        if name is None:
            name = f"level_{self.progress.current_level}_ep_{self.progress.total_episodes}"

        checkpoint_path = self.models_dir / f"{name}.zip"
        self.model.save(checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        if self.algorithm == "ppo":
            self.model = PPO.load(path, env=self.envs)
        elif self.algorithm == "sac":
            self.model = SAC.load(path, env=self.envs)

        self.logger.info(f"Checkpoint loaded: {path}")


def main():
    """Main entry point for curriculum training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Atom Combat fighters using curriculum learning"
    )

    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/curriculum",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1000,
        help="Maximum ticks per episode"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = CurriculumTrainer(
        algorithm=args.algorithm,
        output_dir=args.output_dir,
        n_envs=args.envs,
        max_ticks=args.max_ticks,
        verbose=True
    )

    # Train
    trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()