"""
Curriculum-Based Training System for Atom Combat

Implements progressive training using test dummies before advancing to
hardcoded fighters and eventually population-based training.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
import logging
from datetime import datetime
import time
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

# Use Stable Baselines3 with PyTorch (JAX is in physics engine)
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import test dummy loader
import importlib.util

# Level 3: JAX vmap support
from .curriculum_components import (
    CallbackStepProcessor,
    CurriculumTrainingError,
    EnvFactory,
    GraduationPolicy,
    LevelRunner,
    LevelTransitionStateMachine,
    ModelFactory,
    ProgressReporter,
    ReplayEvaluationService,
    RecoveryManager,
)
from ..utils.nan_detector import NaNDetector
from ..utils.observability import append_jsonl, ensure_analysis_dir


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
        """Clean up vmap environment resources."""
        if hasattr(self, 'vmap_env') and self.vmap_env is not None:
            if hasattr(self.vmap_env, 'close'):
                self.vmap_env.close()
            del self.vmap_env
            self.vmap_env = None

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
        self.replay_evaluation_service = ReplayEvaluationService(self.curriculum_trainer)
        self.step_processor = CallbackStepProcessor(
            curriculum_trainer=self.curriculum_trainer,
            replay_evaluation_service=self.replay_evaluation_service,
            record_evaluation_replay_fn=lambda episode_num, total_episodes: self._record_evaluation_replay(
                episode_num,
                total_episodes,
            ),
        )

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
        # Check for NaN in observations, actions, and rewards
        if hasattr(self.curriculum_trainer, 'nan_detector'):
            detector = self.curriculum_trainer.nan_detector

            # Check observations
            obs = self.locals.get("obs_tensor", None)
            if obs is not None:
                if detector.check_observations(obs, self.n_calls):
                    self.curriculum_trainer.logger.error(f"NaN detected in observations at step {self.n_calls}!")

            # Check actions
            actions = self.locals.get("actions", None)
            if actions is not None:
                if detector.check_actions(actions, self.n_calls):
                    self.curriculum_trainer.logger.error(f"NaN detected in actions at step {self.n_calls}!")

            # Check rewards
            rewards = self.locals.get("rewards", None)
            if rewards is not None:
                if detector.check_rewards(rewards, self.n_calls):
                    self.curriculum_trainer.logger.error(f"NaN detected in rewards at step {self.n_calls}!")

        return self.step_processor.process_infos(
            infos=self.locals.get("infos", []),
            episode_rewards=self.episode_rewards,
            episode_wins=self.episode_wins,
            recent_reward_components=self.recent_reward_components,
        )

    def _record_evaluation_replay(self, episode_num: int, total_episodes: int):
        """Compatibility wrapper used by tests and callback processing."""
        self.replay_evaluation_service.record_evaluation_replay(
            episode_num=episode_num,
            total_episodes=total_episodes,
            episode_rewards=self.episode_rewards,
            episode_wins=self.episode_wins,
        )


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
                 replay_matches_per_opponent: int = 3,
                 override_episodes_per_level: int = None,
                 checkpoint_interval: int = 100000,
                 seed: int = 1337,
                 enable_level1_sanity_gate: bool = True,
                 level1_sanity_gate_episode_threshold: int = 4000,
                 level1_sanity_gate_min_recent_win_rate: float = 0.15,
                 level1_sanity_gate_min_overall_win_rate: float = 0.12):
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
            override_episodes_per_level: Force graduation after N episodes (for testing). None for normal graduation.
            checkpoint_interval: Save checkpoint bundle every N timesteps.
            seed: Training seed for reproducible curriculum runs.
            enable_level1_sanity_gate: Stop obviously broken Level 1 runs early.
            level1_sanity_gate_episode_threshold: Level 1 episode threshold before sanity check triggers.
            level1_sanity_gate_min_recent_win_rate: Minimum recent Level 1 win rate after threshold.
            level1_sanity_gate_min_overall_win_rate: Minimum overall Level 1 win rate after threshold.
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
        self.override_episodes_per_level = override_episodes_per_level
        self.checkpoint_interval = checkpoint_interval
        self.seed = int(seed)
        self.enable_level1_sanity_gate = enable_level1_sanity_gate
        self.level1_sanity_gate_episode_threshold = level1_sanity_gate_episode_threshold
        self.level1_sanity_gate_min_recent_win_rate = level1_sanity_gate_min_recent_win_rate
        self.level1_sanity_gate_min_overall_win_rate = level1_sanity_gate_min_overall_win_rate
        self.abort_reason = None

        # Validate override
        if self.override_episodes_per_level is not None and self.override_episodes_per_level <= 0:
            raise ValueError("override_episodes_per_level must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.level1_sanity_gate_episode_threshold <= 0:
            raise ValueError("level1_sanity_gate_episode_threshold must be positive")

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.analysis_dir = ensure_analysis_dir(self.output_dir)

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

        # Progressive replay recorder for tracking learning progression
        self.progressive_recorder = None
        if self.record_replays:
            from ..progressive_replay_recorder import ProgressiveReplayRecorder
            self.progressive_recorder = ProgressiveReplayRecorder(
                output_dir=str(self.output_dir),
                max_ticks=max_ticks,
                verbose=verbose
            )

        # Initialize NaN detector for debugging
        self.nan_detector = NaNDetector(
            log_dir=str(self.logs_dir / "nan_debug"),
            verbose=True
        )

        # Setup logging
        self._setup_logging()

        # Phase 2 components: policy/reporting/recovery orchestration
        self.graduation_policy = GraduationPolicy(
            override_episodes_per_level=self.override_episodes_per_level,
            min_overall_win_rate=0.5,
        )
        self.progress_reporter = ProgressReporter(self.logger)
        self.recovery_manager = RecoveryManager(
            models_dir=self.models_dir,
            logs_dir=self.logs_dir,
            logger=self.logger,
            checkpoint_interval=self.checkpoint_interval,
            max_retries=3,
        )
        self.level_runner = LevelRunner(
            logger=self.logger,
            recovery_manager=self.recovery_manager,
        )
        self.env_factory = EnvFactory(
            n_envs=self.n_envs,
            max_ticks=self.max_ticks,
            use_vmap=self.use_vmap,
            debug=self.debug,
            logs_dir=self.logs_dir,
            verbose=self.verbose,
            create_env_fn=self.create_env,
            vmap_adapter_cls=VmapEnvAdapter,
            seed_base=self.seed,
        )
        self.model_factory = ModelFactory(
            logs_dir=self.logs_dir,
            verbose=self.verbose,
            seed=self.seed,
        )
        self.level_transition_state_machine = LevelTransitionStateMachine()
        self._current_level_started_at = time.time()
        self._current_level_start_total_episodes = 0
        self._current_level_start_total_wins = 0
        self._current_level_start_timesteps = 0
        self._current_level_summary_written = False
        self.holdout_matches_per_opponent = 1

        # Log override setting
        if self.override_episodes_per_level is not None and self.verbose:
            self.logger.info(f"⚠️  Graduation override enabled: {self.override_episodes_per_level} episodes per level")

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
                str(test_dummy_dir / "atomic/stamina_efficient.py"),
                str(test_dummy_dir / "atomic/charge_on_approach.py"),
                # Using some Level 2 opponents as substitutes for missing files
                str(test_dummy_dir / "atomic/forward_mover.py"),
                str(test_dummy_dir / "atomic/backward_mover.py"),
                str(test_dummy_dir / "atomic/sideways_mover_smooth.py"),
            ],
            min_episodes=400,
            graduation_win_rate=0.85,  # Maintained high standards
            graduation_episodes=50,
            description="Learn spacing control, resource management, and wall combat"
        ))

        # Level 4: Advanced (stance switchers and complex movement)
        curriculum.append(CurriculumLevel(
            name="Advanced",
            difficulty=DifficultyLevel.ADVANCED,
            opponents=[
                str(test_dummy_dir / "atomic/aggressive_stance_switcher.py"),
                str(test_dummy_dir / "atomic/balanced_stance_switcher.py"),
                str(test_dummy_dir / "atomic/defensive_stance_switcher.py"),
                str(test_dummy_dir / "atomic/forward_charger.py"),
                str(test_dummy_dir / "atomic/oscillator.py"),
                str(test_dummy_dir / "atomic/retreater.py"),
            ],
            min_episodes=500,
            graduation_win_rate=0.83,  # Staying near 85% standards
            graduation_episodes=50,
            description="Learn complex strategies and counter-strategies"
        ))

        # Level 5: Expert (example fighters)
        curriculum.append(CurriculumLevel(
            name="Expert",
            difficulty=DifficultyLevel.EXPERT,
            opponents=[
                str(example_dir / "boxer.py"),
                str(example_dir / "counter_puncher.py"),
                str(example_dir / "out_fighter.py"),
                str(example_dir / "slugger.py"),
                str(example_dir / "swarmer.py"),
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
        self.logger.info(f"Training Seed: {self.seed}")
        if self.enable_level1_sanity_gate:
            self.logger.info(
                "Level 1 sanity gate: enabled "
                f"(episodes>={self.level1_sanity_gate_episode_threshold}, "
                f"recent>={self.level1_sanity_gate_min_recent_win_rate:.0%}, "
                f"overall>={self.level1_sanity_gate_min_overall_win_rate:.0%})"
            )
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
            opponent_mass=70.0,
            seed=self.seed + env_id
        )

    def create_envs_for_level(self, level: CurriculumLevel) -> Any:
        """Create parallel environments for the current level."""
        return self.env_factory.create_envs_for_level(level)

    def initialize_model(self):
        """Initialize or load the RL model."""
        self.model = self.model_factory.create_model(
            algorithm=self.algorithm,
            envs=self.envs,
            device=self.device,
            seed=self.seed,
        )

    def update_progress(self, won: bool, reward: float = 0, info: dict = None):
        """Update training progress for the current episode."""
        self.progress_reporter.update_progress(
            progress=self.progress,
            level=self.get_current_level(),
            won=won,
            reward=reward,
            info=info,
        )

    def get_current_level(self) -> CurriculumLevel:
        """Get the current curriculum level."""
        if self.progress.current_level >= len(self.curriculum):
            return self.curriculum[-1]  # Stay at highest level
        return self.curriculum[self.progress.current_level]

    def _get_holdout_suite(self) -> List[Dict[str, str]]:
        """Return the fixed lightweight curriculum holdout suite."""
        test_dummy_dir = Path("fighters/test_dummies/atomic")
        example_dir = Path("fighters/examples")
        return [
            {"category": "stationary", "label": "stationary_neutral", "opponent_path": str(test_dummy_dir / "stationary_neutral.py")},
            {"category": "stationary", "label": "stationary_defending", "opponent_path": str(test_dummy_dir / "stationary_defending.py")},
            {"category": "movement", "label": "approach_slow", "opponent_path": str(test_dummy_dir / "approach_slow.py")},
            {"category": "movement", "label": "circle_right", "opponent_path": str(test_dummy_dir / "circle_right.py")},
            {"category": "spacing", "label": "distance_keeper_1m", "opponent_path": str(test_dummy_dir / "distance_keeper_1m.py")},
            {"category": "spacing", "label": "charge_on_approach", "opponent_path": str(test_dummy_dir / "charge_on_approach.py")},
            {"category": "advanced", "label": "aggressive_stance_switcher", "opponent_path": str(test_dummy_dir / "aggressive_stance_switcher.py")},
            {"category": "advanced", "label": "retreater", "opponent_path": str(test_dummy_dir / "retreater.py")},
            {"category": "expert", "label": "boxer", "opponent_path": str(example_dir / "boxer.py")},
            {"category": "expert", "label": "slugger", "opponent_path": str(example_dir / "slugger.py")},
        ]

    def _run_holdout_match(self, opponent_path: str, env_id: int = 0) -> dict[str, Any]:
        """Run a single deterministic holdout match for the current model."""
        env = self.create_env(opponent_path, env_id=env_id)
        try:
            obs, _ = env.reset()
            done = False
            info: dict[str, Any] = {}
            total_reward = 0.0
            fight_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                fight_length += 1
                done = bool(terminated or truncated)

            fighter_hp = float(info.get("fighter_hp", 0.0))
            opponent_hp = float(info.get("opponent_hp", 0.0))
            won = info.get("won")
            if won is None:
                won = fighter_hp > opponent_hp

            return {
                "won": bool(won),
                "damage_dealt": float(info.get("episode_damage_dealt", 0.0)),
                "damage_taken": float(info.get("episode_damage_taken", 0.0)),
                "fight_length": int(fight_length),
                "reward": float(total_reward),
            }
        finally:
            env.close()

    def _evaluate_holdout_suite(self, checkpoint_label: str, matches_per_opponent: int | None = None) -> None:
        """Evaluate the current curriculum model on the fixed holdout suite."""
        if self.model is None:
            return

        if matches_per_opponent is None:
            matches_per_opponent = self.holdout_matches_per_opponent

        suite_results: list[dict[str, Any]] = []
        for suite_entry in self._get_holdout_suite():
            match_results = [
                self._run_holdout_match(suite_entry["opponent_path"], env_id=match_idx)
                for match_idx in range(matches_per_opponent)
            ]
            wins = sum(1 for result in match_results if result["won"])
            suite_results.append(
                {
                    "category": suite_entry["category"],
                    "opponent": suite_entry["label"],
                    "opponent_path": suite_entry["opponent_path"],
                    "matches": len(match_results),
                    "wins": wins,
                    "win_rate": wins / max(1, len(match_results)),
                    "mean_damage_dealt": float(np.mean([result["damage_dealt"] for result in match_results])),
                    "mean_damage_taken": float(np.mean([result["damage_taken"] for result in match_results])),
                    "mean_fight_length": float(np.mean([result["fight_length"] for result in match_results])),
                    "mean_reward": float(np.mean([result["reward"] for result in match_results])),
                }
            )

        overall_matches = sum(result["matches"] for result in suite_results)
        overall_wins = sum(result["wins"] for result in suite_results)
        record = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_label": checkpoint_label,
            "level_index": int(self.progress.current_level),
            "level_name": self.get_current_level().name,
            "global_timestep": self._current_model_timesteps(),
            "global_total_episodes": int(self.progress.total_episodes),
            "overall_matches": overall_matches,
            "overall_wins": overall_wins,
            "overall_win_rate": overall_wins / max(1, overall_matches),
            "suite_results": suite_results,
        }
        append_jsonl(self.analysis_dir / "holdout_eval.jsonl", record)

    def _record_holdout_evaluation(self, checkpoint_label: str) -> None:
        """Run and record holdout evaluation without derailing training on failure."""
        try:
            self._evaluate_holdout_suite(checkpoint_label)
        except Exception as exc:
            self.logger.warning(f"Holdout evaluation failed for {checkpoint_label}: {exc}")
            self._record_failure_event(
                event_type="holdout_evaluation_failed",
                error_type=exc.__class__.__name__,
                message=str(exc),
                recovery_action="skip_holdout_and_continue",
                recovery_succeeded=False,
            )

    def _current_model_timesteps(self) -> int:
        """Return the current model timestep counter when available."""
        value = getattr(self.model, "num_timesteps", 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _begin_level_observation_window(self) -> None:
        """Reset per-level observability anchors for the active curriculum level."""
        self._current_level_started_at = time.time()
        self._current_level_start_total_episodes = int(self.progress.total_episodes)
        self._current_level_start_total_wins = int(self.progress.total_wins)
        self._current_level_start_timesteps = self._current_model_timesteps()
        self._current_level_summary_written = False

    def _build_reward_component_means(self) -> dict[str, float]:
        """Aggregate recent reward component breakdowns into mean values."""
        breakdowns = list(getattr(self.progress, "recent_reward_breakdowns", []))
        if not breakdowns:
            return {}

        aggregates: dict[str, list[float]] = {}
        for breakdown in breakdowns:
            for key, value in breakdown.items():
                if key == "total":
                    continue
                aggregates.setdefault(key, []).append(float(value))

        return {
            key: float(np.mean(values))
            for key, values in aggregates.items()
            if values and abs(float(np.mean(values))) > 0.1
        }

    def _write_level_summary(self, end_reason: str) -> None:
        """Append a structured summary for the current curriculum level."""
        if self._current_level_summary_written:
            return

        level = self.get_current_level()
        recent_episodes = list(self.progress.recent_episodes)
        recent_win_rate = (
            sum(recent_episodes) / len(recent_episodes)
            if recent_episodes
            else None
        )
        overall_win_rate = self.progress.wins_at_level / max(1, self.progress.episodes_at_level)
        recent_rewards = list(getattr(self.progress, "recent_rewards", []))
        record = {
            "timestamp": datetime.now().isoformat(),
            "level_index": int(self.progress.current_level),
            "level_name": level.name,
            "difficulty": level.difficulty.value,
            "end_reason": end_reason,
            "episodes_attempted": int(self.progress.episodes_at_level),
            "wins": int(self.progress.wins_at_level),
            "overall_win_rate": float(overall_win_rate),
            "recent_window_size": len(recent_episodes),
            "recent_win_rate": float(recent_win_rate) if recent_win_rate is not None else None,
            "level_wall_clock_seconds": float(time.time() - self._current_level_started_at),
            "timesteps_consumed": self._current_model_timesteps() - self._current_level_start_timesteps,
            "global_total_episodes": int(self.progress.total_episodes),
            "global_total_wins": int(self.progress.total_wins),
            "global_timesteps": self._current_model_timesteps(),
            "mean_recent_reward": float(np.mean(recent_rewards)) if recent_rewards else None,
            "reward_component_means": self._build_reward_component_means(),
            "graduation_recent_win_rate_required": float(level.graduation_win_rate),
            "graduation_window_episodes": int(level.graduation_episodes),
            "min_episodes": int(level.min_episodes),
        }
        append_jsonl(self.analysis_dir / "level_summaries.jsonl", record)
        self._current_level_summary_written = True

    def _record_failure_event_payload(self, payload: dict[str, Any]) -> None:
        """Append a structured failure/recovery event payload."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "current_level": int(self.progress.current_level),
            "global_timestep": self._current_model_timesteps(),
            **payload,
        }
        append_jsonl(self.analysis_dir / "failure_events.jsonl", event)

    def _record_failure_event(
        self,
        *,
        event_type: str,
        error_type: str,
        message: str,
        recovery_action: str,
        recovery_succeeded: bool | None,
        checkpoint_path: Path | None = None,
        debug_path: Path | None = None,
        nan_retries: int | None = None,
    ) -> None:
        """Convenience wrapper for common structured failure events."""
        self._record_failure_event_payload(
            {
                "event_type": event_type,
                "error_type": error_type,
                "message": message,
                "recovery_action": recovery_action,
                "recovery_succeeded": recovery_succeeded,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "debug_path": str(debug_path) if debug_path else None,
                "nan_retries": nan_retries,
            }
        )

    def check_training_sanity_gate(self) -> Optional[str]:
        """Return an abort reason when the Level 1 run is clearly unhealthy."""
        if not self.enable_level1_sanity_gate:
            return None
        if self.override_episodes_per_level is not None:
            return None
        if self.progress.current_level != 0:
            return None
        if self.progress.episodes_at_level < self.level1_sanity_gate_episode_threshold:
            return None
        if not self.progress.recent_episodes:
            return None

        recent_win_rate = sum(self.progress.recent_episodes) / len(self.progress.recent_episodes)
        overall_win_rate = self.progress.wins_at_level / max(1, self.progress.episodes_at_level)

        if recent_win_rate >= self.level1_sanity_gate_min_recent_win_rate:
            return None
        if overall_win_rate >= self.level1_sanity_gate_min_overall_win_rate:
            return None

        return (
            "Level 1 sanity gate triggered after "
            f"{self.progress.episodes_at_level} episodes: overall win rate {overall_win_rate:.1%} "
            f"(need >= {self.level1_sanity_gate_min_overall_win_rate:.1%}), recent win rate {recent_win_rate:.1%} "
            f"(need >= {self.level1_sanity_gate_min_recent_win_rate:.1%}). "
            "This run is likely unhealthy; stop early, verify pinned Colab dependencies, and retry with the same seed."
        )

    def should_graduate(self) -> bool:
        """Check if the fighter should graduate to the next level."""
        level = self.get_current_level()
        decision = self.graduation_policy.evaluate(
            progress=self.progress,
            level=level,
            curriculum_size=len(self.curriculum),
        )

        should_log = decision.should_graduate or (
            decision.recent_passed and self.progress.episodes_at_level % 100 == 0
        )
        if should_log and decision.reason != "override":
            self.progress_reporter.log_graduation_decision(decision)

        return decision.should_graduate

    def advance_level(self):
        """Advance to the next curriculum level."""
        current = self.get_current_level()
        self._write_level_summary(end_reason="graduated")
        checkpoint_label = (
            f"level_{self.progress.current_level + 1}_"
            f"{current.name.lower().replace(' ', '_')}_graduated"
        )
        self._record_holdout_evaluation(checkpoint_label)

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
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")

        transition = self.level_transition_state_machine.advance(
            progress=self.progress,
            curriculum=self.curriculum,
        )

        # Log the reset for debugging
        self.logger.info("📊 METRICS RESET FOR NEW LEVEL:")
        self.logger.info(f"   Episodes at level: {self.progress.episodes_at_level}")
        self.logger.info(f"   Wins at level: {self.progress.wins_at_level}")
        self.logger.info(f"   Recent episodes buffer: {len(self.progress.recent_episodes)} episodes")

        # Check if completed curriculum
        if transition.completed:
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

            self._begin_level_observation_window()

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

        # Save progressive replay index
        if self.progressive_recorder:
            self.progressive_recorder.save_progressive_index()

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

    def _log_training_loop_error(self, exc: CurriculumTrainingError):
        """Emit a consistent structured error block for training loop failures."""
        details = exc.details
        self._record_failure_event(
            event_type="training_loop_error",
            error_type=exc.__class__.__name__,
            message=str(exc),
            recovery_action="checkpoint_recovery" if details.checkpoint_path is not None else "abort_run",
            recovery_succeeded=False,
            checkpoint_path=details.checkpoint_path,
            debug_path=details.debug_path,
            nan_retries=details.nan_retries,
        )
        self.logger.error("=" * 80)
        self.logger.error("CURRICULUM TRAINING LOOP FAILED")
        self.logger.error("=" * 80)
        self.logger.error(f"Error Type: {exc.__class__.__name__}")
        self.logger.error(f"Error: {exc}")
        self.logger.error(f"Current level: {details.level}")
        self.logger.error(f"Completed steps: {details.completed_steps}/{details.total_timesteps}")
        self.logger.error(f"NaN retries: {details.nan_retries}")
        if details.checkpoint_path is not None:
            self.logger.error(f"Checkpoint path: {details.checkpoint_path}")
        if details.debug_path is not None:
            self.logger.error(f"Debug dump: {details.debug_path}")
        self.logger.error("=" * 80)

    def _capture_training_state(self, callback: CurriculumCallback) -> dict:
        """Capture curriculum + callback state for checkpoint resume."""
        progress_state = asdict(self.progress)
        progress_state["recent_rewards"] = list(getattr(self.progress, "recent_rewards", []))
        progress_state["recent_reward_breakdowns"] = list(
            getattr(self.progress, "recent_reward_breakdowns", [])
        )

        callback_state = {
            "episode_rewards": list(callback.episode_rewards),
            "episode_wins": list(callback.episode_wins),
            "recent_reward_components": list(callback.recent_reward_components),
            "rollout_count": getattr(callback, "rollout_count", 0),
        }
        return {
            "schema_version": 1,
            "progress": progress_state,
            "callback": callback_state,
            "timestamp": datetime.now().isoformat(),
        }

    def _sync_environment_to_level(self):
        """Ensure env/model opponents are aligned with restored curriculum level."""
        if self.envs is None or self.model is None:
            return
        if self.progress.current_level >= len(self.curriculum):
            return

        restored_level = self.get_current_level()
        if self.use_vmap:
            self.logger.info("Restoring vmap environments to checkpoint level state...")
            old_envs = self.envs
            self.envs = self.create_envs_for_level(restored_level)
            self.model.set_env(self.envs)
            if old_envs is not self.envs and hasattr(old_envs, "close"):
                old_envs.close()
            return

        for env_idx in range(self.n_envs):
            opponent_idx = env_idx % len(restored_level.opponents)
            opponent_path = restored_level.opponents[opponent_idx]
            opponent_func = self.load_opponent(opponent_path)
            self.envs.env_method("set_opponent", opponent_func, indices=[env_idx])

    def _restore_training_state(self, callback: CurriculumCallback, state: dict):
        """Restore curriculum + callback state from checkpoint payload."""
        progress_state = state.get("progress", {})
        previous_level = self.progress.current_level

        for field_name in [
            "current_level",
            "episodes_at_level",
            "wins_at_level",
            "total_episodes",
            "total_wins",
            "start_time",
        ]:
            if field_name in progress_state:
                setattr(self.progress, field_name, progress_state[field_name])

        self.progress.recent_episodes = list(progress_state.get("recent_episodes", []))
        self.progress.graduated_levels = list(progress_state.get("graduated_levels", []))
        self.progress.recent_rewards = list(progress_state.get("recent_rewards", []))
        self.progress.recent_reward_breakdowns = list(
            progress_state.get("recent_reward_breakdowns", [])
        )

        callback_state = state.get("callback", {})
        callback.episode_rewards = list(callback_state.get("episode_rewards", []))
        callback.episode_wins = list(callback_state.get("episode_wins", []))
        callback.recent_reward_components = list(
            callback_state.get("recent_reward_components", [])
        )
        callback.rollout_count = int(callback_state.get("rollout_count", 0))

        if self.progress.current_level != previous_level:
            self.logger.info(
                "Checkpoint restored level transition context: "
                f"{previous_level} -> {self.progress.current_level}"
            )
            self._sync_environment_to_level()

    def train(self, total_timesteps: int = 1_000_000, resume_from_latest: bool = False):
        """
        Train the fighter through the curriculum.

        Args:
            total_timesteps: Total training timesteps across all levels
            resume_from_latest: Load latest checkpoint bundle before training.
        """
        self.logger.info("Starting curriculum training...")

        # Start with first level
        self._begin_level_observation_window()
        level = self.get_current_level()
        self.logger.info(f"Level 1: {level.name}")
        self.logger.info(f"Description: {level.description}")

        # Create environments for first level
        self.envs = self.create_envs_for_level(level)

        # Initialize model
        self.initialize_model()

        # Create callback
        callback = CurriculumCallback(self, verbose=self.verbose)

        initial_step = 0
        if resume_from_latest:
            latest_bundle = self.recovery_manager.find_latest_checkpoint_bundle()
            if latest_bundle is None:
                self.logger.info("Resume requested but no checkpoint bundle found; starting fresh run.")
            else:
                self.logger.info(f"Resuming from checkpoint bundle: {latest_bundle.model_path}")
                self.model = self.model.__class__.load(latest_bundle.model_path, env=self.envs)
                checkpoint_state = self.recovery_manager.load_checkpoint_training_state(latest_bundle)
                if checkpoint_state is not None:
                    self._restore_training_state(callback, checkpoint_state)
                initial_step = int(latest_bundle.step)
                if self.verbose:
                    self.logger.info(
                        f"Resume state loaded from step {initial_step}; "
                        f"remaining timesteps target: {max(0, total_timesteps - initial_step)}"
                    )

        try:
            # Train with retry/recovery orchestration.
            self.model = self.level_runner.run(
                model=self.model,
                envs=self.envs,
                callback=callback,
                total_timesteps=total_timesteps,
                initial_step=initial_step,
                verbose=self.verbose,
                current_level_getter=lambda: self.progress.current_level,
                training_state_getter=lambda: self._capture_training_state(callback),
                training_state_restorer=lambda state: self._restore_training_state(callback, state),
                model_update_fn=lambda recovered_model: setattr(self, "model", recovered_model),
                env_getter=lambda: self.envs,
                event_recorder=self._record_failure_event_payload,
            )
        except CurriculumTrainingError as exc:
            self._write_level_summary(end_reason="training_error")
            self._log_training_loop_error(exc)
            raise
        finally:
            if self.envs:
                self.envs.close()

        if self.abort_reason is not None:
            self._write_level_summary(end_reason="sanity_gate")
            self._record_failure_event(
                event_type="sanity_gate_triggered",
                error_type="RuntimeError",
                message=self.abort_reason,
                recovery_action="abort_run",
                recovery_succeeded=False,
            )
            self.logger.error("="*80)
            self.logger.error("CURRICULUM SANITY GATE TRIGGERED")
            self.logger.error("="*80)
            self.logger.error(self.abort_reason)
            self.logger.error("="*80)
            raise RuntimeError(self.abort_reason)

        self.logger.info("Training complete!")

        # Check if we graduated at least the first level
        if len(self.progress.graduated_levels) == 0:
            self._write_level_summary(end_reason="max_budget")
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
            self._record_failure_event(
                event_type="max_budget_exhausted",
                error_type="RuntimeError",
                message="Curriculum failed to graduate Level 1 within allotted timesteps",
                recovery_action="increase_budget_or_adjust_config",
                recovery_succeeded=False,
            )
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
