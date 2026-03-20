"""
Composable components used by CurriculumTrainer.

These components isolate progression policy, reporting, and recovery behavior so
the trainer orchestrates state transitions instead of containing all logic.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize


@dataclass(frozen=True)
class GraduationDecision:
    """Structured result from graduation evaluation."""

    should_graduate: bool
    reason: str
    recent_win_rate: float
    overall_win_rate: float
    recent_wins: int
    recent_total: int
    required_recent_win_rate: float
    required_overall_win_rate: float
    recent_passed: bool
    overall_passed: bool
    episodes_at_level: int


@dataclass(frozen=True)
class LevelTransitionResult:
    """Result of advancing the curriculum state machine."""

    graduated_level_name: str
    next_level_index: int
    completed: bool


@dataclass(frozen=True)
class TrainingErrorDetails:
    """Structured context attached to training-domain exceptions."""

    level: int
    completed_steps: int
    total_timesteps: int
    nan_retries: int = 0
    checkpoint_path: Optional[Path] = None
    debug_path: Optional[Path] = None


class CurriculumTrainingError(RuntimeError):
    """Base error for curriculum training loop failures."""

    def __init__(self, message: str, *, details: TrainingErrorDetails):
        super().__init__(message)
        self.details = details


class NaNRecoveryError(CurriculumTrainingError):
    """Raised when NaN retries are exhausted and recovery fails."""


class CheckpointRecoveryError(CurriculumTrainingError):
    """Raised when checkpoint-based model recovery fails."""


class TrainingLoopExecutionError(CurriculumTrainingError):
    """Raised for unexpected non-NaN execution failures in the training loop."""


class GraduationPolicy:
    """Encapsulates level graduation rules."""

    def __init__(self, override_episodes_per_level: Optional[int], min_overall_win_rate: float = 0.5):
        self.override_episodes_per_level = override_episodes_per_level
        self.min_overall_win_rate = min_overall_win_rate

    def evaluate(self, *, progress, level, curriculum_size: int) -> GraduationDecision:
        """Evaluate whether the current level should graduate."""
        episodes = progress.episodes_at_level

        if progress.current_level >= curriculum_size:
            return GraduationDecision(
                should_graduate=False,
                reason="curriculum_complete",
                recent_win_rate=0.0,
                overall_win_rate=0.0,
                recent_wins=0,
                recent_total=0,
                required_recent_win_rate=level.graduation_win_rate,
                required_overall_win_rate=self.min_overall_win_rate,
                recent_passed=False,
                overall_passed=False,
                episodes_at_level=episodes,
            )

        if self.override_episodes_per_level is not None:
            return GraduationDecision(
                should_graduate=episodes >= self.override_episodes_per_level,
                reason="override",
                recent_win_rate=0.0,
                overall_win_rate=0.0,
                recent_wins=0,
                recent_total=0,
                required_recent_win_rate=0.0,
                required_overall_win_rate=0.0,
                recent_passed=False,
                overall_passed=False,
                episodes_at_level=episodes,
            )

        if episodes < level.min_episodes:
            return GraduationDecision(
                should_graduate=False,
                reason="below_min_episodes",
                recent_win_rate=0.0,
                overall_win_rate=progress.wins_at_level / max(1, episodes),
                recent_wins=0,
                recent_total=len(progress.recent_episodes),
                required_recent_win_rate=level.graduation_win_rate,
                required_overall_win_rate=self.min_overall_win_rate,
                recent_passed=False,
                overall_passed=False,
                episodes_at_level=episodes,
            )

        if len(progress.recent_episodes) < level.graduation_episodes:
            return GraduationDecision(
                should_graduate=False,
                reason="insufficient_recent_window",
                recent_win_rate=0.0,
                overall_win_rate=progress.wins_at_level / max(1, episodes),
                recent_wins=sum(progress.recent_episodes),
                recent_total=len(progress.recent_episodes),
                required_recent_win_rate=level.graduation_win_rate,
                required_overall_win_rate=self.min_overall_win_rate,
                recent_passed=False,
                overall_passed=False,
                episodes_at_level=episodes,
            )

        recent_wins = sum(progress.recent_episodes)
        recent_total = len(progress.recent_episodes)
        recent_win_rate = recent_wins / max(1, recent_total)
        overall_win_rate = progress.wins_at_level / max(1, episodes)

        recent_passed = recent_win_rate >= level.graduation_win_rate
        overall_passed = overall_win_rate >= self.min_overall_win_rate
        should_graduate = recent_passed and overall_passed

        if should_graduate:
            reason = "passed"
        elif recent_passed and not overall_passed:
            reason = "overall_too_low"
        elif not recent_passed:
            reason = "recent_too_low"
        else:
            reason = "failed"

        return GraduationDecision(
            should_graduate=should_graduate,
            reason=reason,
            recent_win_rate=recent_win_rate,
            overall_win_rate=overall_win_rate,
            recent_wins=recent_wins,
            recent_total=recent_total,
            required_recent_win_rate=level.graduation_win_rate,
            required_overall_win_rate=self.min_overall_win_rate,
            recent_passed=recent_passed,
            overall_passed=overall_passed,
            episodes_at_level=episodes,
        )


class ProgressReporter:
    """Updates progress state and emits periodic progress logs."""

    def __init__(self, logger):
        self.logger = logger

    def update_progress(self, *, progress, level, won: bool, reward: float = 0, info: Optional[dict] = None):
        progress.episodes_at_level += 1
        progress.total_episodes += 1

        if won:
            progress.wins_at_level += 1
            progress.total_wins += 1

        progress.recent_episodes.append(won)
        if len(progress.recent_episodes) > level.graduation_episodes:
            progress.recent_episodes.pop(0)

        if not hasattr(progress, "recent_rewards"):
            progress.recent_rewards = []
            progress.recent_reward_breakdowns = []

        progress.recent_rewards.append(reward)
        if len(progress.recent_rewards) > 100:
            progress.recent_rewards.pop(0)

        if info and "reward_breakdown" in info:
            progress.recent_reward_breakdowns.append(info["reward_breakdown"])
            if len(progress.recent_reward_breakdowns) > 20:
                progress.recent_reward_breakdowns.pop(0)

        if progress.episodes_at_level % 100 == 0:
            self._log_progress_snapshot(progress=progress, level=level)

    def log_graduation_decision(self, decision: GraduationDecision):
        status = "✅ PASSED" if decision.should_graduate else "❌ FAILED (overall too low)"
        self.logger.info(f"🎓 GRADUATION CHECK {status}")
        self.logger.info(f"   Recent wins: {decision.recent_wins}/{decision.recent_total}")
        self.logger.info(
            f"   Recent WR: {decision.recent_win_rate:.2%} "
            f"(need {decision.required_recent_win_rate:.1%}) "
            f"{'✓' if decision.recent_passed else '✗'}"
        )
        self.logger.info(
            f"   Overall WR: {decision.overall_win_rate:.2%} "
            f"(need {decision.required_overall_win_rate:.1%}) "
            f"{'✓' if decision.overall_passed else '✗'}"
        )
        self.logger.info(f"   Episodes at level: {decision.episodes_at_level}")

    def _log_progress_snapshot(self, *, progress, level):
        overall_win_rate = progress.wins_at_level / max(1, progress.episodes_at_level)

        if len(progress.recent_episodes) >= level.graduation_episodes:
            recent_win_rate = sum(progress.recent_episodes) / len(progress.recent_episodes)
            mean_reward = np.mean(progress.recent_rewards) if progress.recent_rewards else 0

            self.logger.info(
                f"Progress: Episode {progress.episodes_at_level} | "
                f"Overall WR: {overall_win_rate:.1%} | "
                f"Recent WR: {recent_win_rate:.1%} "
                f"(need {level.graduation_win_rate:.1%}) | "
                f"Mean Reward: {mean_reward:.1f}"
            )

            if progress.recent_reward_breakdowns:
                reward_summary = self._build_reward_summary(progress.recent_reward_breakdowns)
                if reward_summary:
                    self.logger.info(f"  Reward Components: {', '.join(reward_summary)}")

            recent_wins = sum(progress.recent_episodes)
            recent_losses = len(progress.recent_episodes) - recent_wins
            self.logger.info(
                f"  Last {len(progress.recent_episodes)} episodes: "
                f"{recent_wins} wins, {recent_losses} losses"
            )
            return

        self.logger.info(
            f"Progress: Episode {progress.episodes_at_level} | "
            f"Overall WR: {overall_win_rate:.1%}"
        )

    @staticmethod
    def _build_reward_summary(recent_breakdowns) -> list[str]:
        avg_breakdown = {}
        for breakdown in recent_breakdowns:
            for key, value in breakdown.items():
                avg_breakdown.setdefault(key, []).append(value)

        summary = []
        for key, values in avg_breakdown.items():
            if key == "total":
                continue
            avg_val = np.mean(values)
            if abs(avg_val) > 0.1:
                summary.append(f"{key}={avg_val:+.1f}")
        return summary


class RecoveryManager:
    """Handles checkpointing and recovery for training loops."""

    def __init__(self, *, models_dir: Path, logs_dir: Path, logger, checkpoint_interval: int = 100000, max_retries: int = 3):
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.logger = logger
        self.checkpoint_interval = checkpoint_interval
        self.max_retries = max_retries

    def maybe_save_checkpoint(self, model, start_timestep: int, verbose: bool) -> Optional[Path]:
        if start_timestep % self.checkpoint_interval != 0 or model is None:
            return None
        checkpoint_path = self.models_dir / f"checkpoint_{start_timestep}.zip"
        model.save(checkpoint_path)
        if verbose:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    @staticmethod
    def is_nan_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "nan" in text or "invalid values" in text

    @staticmethod
    def is_progress_conflict_error(exc: Exception) -> bool:
        return "Only one live display may be active at once" in str(exc)

    def recover_model_from_checkpoint(self, model, checkpoint_path: Path, envs) -> Tuple[Any, Optional[float]]:
        recovered_model = model.__class__.load(checkpoint_path, env=envs)
        current_lr = getattr(recovered_model, "learning_rate", None)
        if isinstance(current_lr, (int, float)):
            new_lr = float(current_lr) * 0.5
            recovered_model.learning_rate = new_lr
            return recovered_model, new_lr
        return recovered_model, None

    def write_nan_debug_dump(
        self,
        *,
        exc: Exception,
        current_level: int,
        total_steps: int,
        episode_rewards: list,
        nan_retries: int,
    ) -> Path:
        debug_info = {
            "error": str(exc),
            "level": current_level,
            "total_steps": total_steps,
            "episodes": len(episode_rewards),
            "recent_rewards": episode_rewards[-50:] if episode_rewards else [],
            "nan_retries": nan_retries,
        }
        debug_path = self.logs_dir / f"nan_error_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_path, "w") as f:
            json.dump(debug_info, f, indent=2)
        return debug_path


class LevelRunner:
    """Runs SB3 learning with robust retry/recovery behavior."""

    def __init__(self, *, logger, recovery_manager: RecoveryManager):
        self.logger = logger
        self.recovery_manager = recovery_manager

    def run(
        self,
        *,
        model,
        envs,
        callback,
        total_timesteps: int,
        verbose: bool,
        current_level_getter,
    ):
        remaining_timesteps = total_timesteps
        start_timestep = 0
        nan_retries = 0
        disable_sb3_progress_bar = False
        progress_bar_disable_reason: Optional[str] = None
        last_checkpoint: Optional[Path] = None

        while remaining_timesteps > 0 and nan_retries < self.recovery_manager.max_retries:
            try:
                checkpoint_path = self.recovery_manager.maybe_save_checkpoint(model, start_timestep, verbose)
                if checkpoint_path is not None:
                    last_checkpoint = checkpoint_path

                use_sb3_progress_bar = bool(
                    verbose and not disable_sb3_progress_bar and nan_retries == 0
                )
                if verbose and progress_bar_disable_reason:
                    self.logger.info(f"SB3 progress bar disabled: {progress_bar_disable_reason}")
                    progress_bar_disable_reason = None

                model.learn(
                    total_timesteps=remaining_timesteps,
                    callback=callback,
                    progress_bar=use_sb3_progress_bar,
                    reset_num_timesteps=False,
                )
                return model

            except ValueError as exc:
                if not self.recovery_manager.is_nan_error(exc):
                    raise

                nan_retries += 1
                current_level = int(current_level_getter())
                completed_steps = int(getattr(callback, "n_calls", 0))
                self.logger.error("=" * 80)
                self.logger.error(f"NaN ERROR DETECTED (Retry {nan_retries}/{self.recovery_manager.max_retries})")
                self.logger.error("=" * 80)
                self.logger.error(f"Error: {exc}")
                self.logger.error(f"Current level: {current_level}")
                self.logger.error(f"Total steps: {completed_steps}")

                if nan_retries < self.recovery_manager.max_retries and last_checkpoint:
                    self.logger.info(f"Attempting recovery from checkpoint: {last_checkpoint}")
                    try:
                        model, new_lr = self.recovery_manager.recover_model_from_checkpoint(
                            model,
                            last_checkpoint,
                            envs,
                        )
                    except Exception as recovery_exc:
                        details = TrainingErrorDetails(
                            level=current_level,
                            completed_steps=completed_steps,
                            total_timesteps=total_timesteps,
                            nan_retries=nan_retries,
                            checkpoint_path=Path(last_checkpoint),
                        )
                        raise CheckpointRecoveryError(
                            f"Failed to recover model from checkpoint: {last_checkpoint}",
                            details=details,
                        ) from recovery_exc

                    if new_lr is not None:
                        self.logger.info(f"Reduced learning rate to: {new_lr}")

                    disable_sb3_progress_bar = True
                    if progress_bar_disable_reason is None:
                        progress_bar_disable_reason = (
                            "retrying after NaN recovery to avoid Rich live display conflicts"
                        )

                    completed_steps = getattr(callback, "n_calls", 0)
                    remaining_timesteps = total_timesteps - completed_steps
                    start_timestep = completed_steps
                    self.logger.info(f"Resuming training from step {completed_steps}")
                    continue

                debug_path = self.recovery_manager.write_nan_debug_dump(
                    exc=exc,
                    current_level=current_level,
                    total_steps=completed_steps,
                    episode_rewards=list(getattr(callback, "episode_rewards", [])),
                    nan_retries=nan_retries,
                )
                self.logger.error(f"Debug info saved to: {debug_path}")
                details = TrainingErrorDetails(
                    level=current_level,
                    completed_steps=completed_steps,
                    total_timesteps=total_timesteps,
                    nan_retries=nan_retries,
                    debug_path=Path(debug_path),
                )
                raise NaNRecoveryError(
                    f"Training failed due to NaN after {nan_retries} retries",
                    details=details,
                ) from exc

            except Exception as exc:
                if self.recovery_manager.is_progress_conflict_error(exc) and not disable_sb3_progress_bar:
                    disable_sb3_progress_bar = True
                    progress_bar_disable_reason = (
                        "detected Rich live display conflict; retrying without SB3 progress bar"
                    )
                    self.logger.warning(
                        "Detected Rich live display conflict from SB3 progress bar. "
                        "Retrying without SB3 progress bar."
                    )
                    continue

                details = TrainingErrorDetails(
                    level=int(current_level_getter()),
                    completed_steps=int(getattr(callback, "n_calls", 0)),
                    total_timesteps=total_timesteps,
                    nan_retries=nan_retries,
                )
                raise TrainingLoopExecutionError(
                    f"Unexpected training loop failure: {exc}",
                    details=details,
                ) from exc

        return model


class LevelTransitionStateMachine:
    """Explicit state machine for curriculum level transitions."""

    @staticmethod
    def advance(progress, curriculum) -> LevelTransitionResult:
        if progress.current_level >= len(curriculum):
            return LevelTransitionResult(
                graduated_level_name=curriculum[-1].name if curriculum else "unknown",
                next_level_index=progress.current_level,
                completed=True,
            )

        current_level_name = curriculum[progress.current_level].name
        progress.graduated_levels.append(current_level_name)
        progress.current_level += 1
        progress.episodes_at_level = 0
        progress.wins_at_level = 0
        progress.recent_episodes = []

        completed = progress.current_level >= len(curriculum)
        return LevelTransitionResult(
            graduated_level_name=current_level_name,
            next_level_index=progress.current_level,
            completed=completed,
        )


class ReplayEvaluationService:
    """
    Runs evaluation matches and records progressive replays.

    This extracts replay-eval orchestration from CurriculumCallback.
    """

    def __init__(self, curriculum_trainer):
        self.curriculum_trainer = curriculum_trainer

    def record_evaluation_replay(self, episode_num: int, total_episodes: int, episode_rewards, episode_wins):
        import importlib.util
        from ...orchestrator.match_orchestrator import MatchOrchestrator
        from ...arena import WorldConfig
        from ..signal_engine import build_observation_from_snapshot

        config = WorldConfig()
        level = self.curriculum_trainer.get_current_level()

        model = self.curriculum_trainer.model
        if model is None:
            if self.curriculum_trainer.verbose:
                print("  ⚠️  Model not initialized yet, using random agent for replay")

            def ai_decide(snapshot):
                return {
                    "acceleration": random.choice([-1.0, 0.0, 1.0]),
                    "stance": random.choice(["neutral", "extended", "defending"]),
                }

        else:
            def ai_decide(snapshot):
                obs = build_observation_from_snapshot(snapshot, recent_damage=0.0)
                try:
                    action, _ = model.predict(
                        np.array([obs]),
                        deterministic=False,
                    )
                    if action.ndim > 1:
                        action = action[0]

                    acceleration_normalized = float(np.clip(action[0], -1.0, 1.0))
                    acceleration = acceleration_normalized * config.max_acceleration
                    stance_idx = int(np.clip(action[1], 0, 2))
                    stance = ["neutral", "extended", "defending"][stance_idx]
                    return {"acceleration": acceleration, "stance": stance}
                except Exception as exc:
                    if self.curriculum_trainer.verbose:
                        print(f"  ⚠️  Model prediction failed: {exc}")
                        import traceback
                        traceback.print_exc()
                    return {"acceleration": 0.0, "stance": "neutral"}

        opponent_path = level.opponents[0] if level.opponents else None
        opponent_name = "Dummy"
        opponent_decide = None

        if opponent_path and Path(opponent_path).exists():
            opponent_name = Path(opponent_path).stem
            spec = importlib.util.spec_from_file_location(opponent_name, opponent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "decide"):
                opponent_decide = module.decide

        if opponent_decide is None:
            def opponent_decide(snapshot):
                return {"acceleration": 0.0, "stance": "neutral"}

        orchestrator = MatchOrchestrator(
            config,
            max_ticks=self.curriculum_trainer.max_ticks,
            record_telemetry=True,
        )
        fighter_a_spec = {"name": "AI_Fighter", "mass": 70.0, "position": 2.0}
        fighter_b_spec = {"name": opponent_name, "mass": 70.0, "position": 10.0}

        match_seed = random.randint(0, 1000000)
        result = orchestrator.run_match(
            fighter_a_spec,
            fighter_b_spec,
            ai_decide,
            opponent_decide,
            seed=match_seed,
        )

        if result.telemetry is None or len(result.telemetry.get("ticks", [])) == 0:
            if self.curriculum_trainer.verbose:
                print(f"  ⚠️  Match produced no telemetry! Winner: {result.winner}, Ticks: {result.total_ticks}")
            return

        recent_wins = episode_wins[-100:] if len(episode_wins) > 100 else episode_wins
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
        recent_rewards = episode_rewards[-20:] if len(episode_rewards) > 20 else episode_rewards

        self.curriculum_trainer.progressive_recorder.record_episode_replay(
            telemetry=result.telemetry,
            match_result=result,
            level_name=level.name,
            level_num=self.curriculum_trainer.progress.current_level + 1,
            episode=episode_num,
            total_episodes=total_episodes,
            win_rate=win_rate,
            recent_rewards=recent_rewards,
            fighter_a_name="AI_Fighter",
            fighter_b_name=opponent_name,
        )


class CallbackStepProcessor:
    """Processes episode-completion infos for CurriculumCallback."""

    def __init__(
        self,
        curriculum_trainer,
        replay_evaluation_service: Optional[ReplayEvaluationService] = None,
        record_evaluation_replay_fn: Optional[Callable[[int, int], None]] = None,
    ):
        self.curriculum_trainer = curriculum_trainer
        self.replay_evaluation_service = replay_evaluation_service
        self.record_evaluation_replay_fn = record_evaluation_replay_fn

    def process_infos(self, infos, episode_rewards, episode_wins, recent_reward_components) -> bool:
        for info in infos:
            if "episode" not in info:
                continue

            reward = info["episode"]["r"]
            episode_rewards.append(reward)

            won = info.get("won", False)
            episode_wins.append(won)

            if "reward_breakdown" in info:
                recent_reward_components.append(info["reward_breakdown"])
                if len(recent_reward_components) > 100:
                    recent_reward_components.pop(0)

            self._maybe_record_progressive_replay(
                episode_rewards=episode_rewards,
                episode_wins=episode_wins,
            )

            self.curriculum_trainer.update_progress(won, reward, info)

            if self.curriculum_trainer.should_graduate():
                self.curriculum_trainer.advance_level()
                if self.curriculum_trainer.progress.current_level >= len(self.curriculum_trainer.curriculum):
                    self.curriculum_trainer.logger.info("Curriculum complete - stopping training early")
                    return False

        return True

    def _maybe_record_progressive_replay(self, *, episode_rewards, episode_wins):
        if self.curriculum_trainer.progressive_recorder is None:
            return

        episode_num = len(episode_rewards)
        total_episodes = max(10000, episode_num * 2)
        if not self.curriculum_trainer.progressive_recorder.should_record(episode_num, total_episodes):
            return

        try:
            if self.curriculum_trainer.verbose:
                self.curriculum_trainer.logger.info(f"  💾 Recording replay for episode {episode_num}...")
            if self.record_evaluation_replay_fn is not None:
                self.record_evaluation_replay_fn(episode_num, total_episodes)
            elif self.replay_evaluation_service is not None:
                self.replay_evaluation_service.record_evaluation_replay(
                    episode_num=episode_num,
                    total_episodes=total_episodes,
                    episode_rewards=episode_rewards,
                    episode_wins=episode_wins,
                )
        except Exception as exc:
            if self.curriculum_trainer.verbose:
                import traceback
                self.curriculum_trainer.logger.error(f"  ❌ Failed to record replay: {exc}")
                self.curriculum_trainer.logger.error(f"  Traceback: {traceback.format_exc()}")


class EnvFactory:
    """Creates environment stacks for curriculum levels."""

    def __init__(
        self,
        *,
        n_envs: int,
        max_ticks: int,
        use_vmap: bool,
        debug: bool,
        logs_dir: Path,
        verbose: bool,
        create_env_fn,
        vmap_adapter_cls,
    ):
        self.n_envs = n_envs
        self.max_ticks = max_ticks
        self.use_vmap = use_vmap
        self.debug = debug
        self.logs_dir = Path(logs_dir)
        self.verbose = verbose
        self.create_env_fn = create_env_fn
        self.vmap_adapter_cls = vmap_adapter_cls

    def create_envs_for_level(self, level):
        if self.use_vmap:
            return self._create_vmap_envs(level)
        return self._create_dummy_envs(level)

    def _create_vmap_envs(self, level):
        from ..vmap_env_wrapper import VmapEnvWrapper
        from ...arena import WorldConfig

        opponent_paths = level.opponents
        if self.verbose:
            print(f"🚀 Creating {self.n_envs} parallel JAX vmap environments...")
            print(f"   Opponents: {len(opponent_paths)} different types")
            if len(opponent_paths) > 1:
                envs_per_opponent = self.n_envs // len(opponent_paths)
                print(f"   Distribution: ~{envs_per_opponent} envs per opponent")
                for i, path in enumerate(opponent_paths):
                    print(f"     {i+1}. {Path(path).stem}")
            print("\n⏳ Initializing JAX vmap environment (this may take 30-60 seconds for JIT compilation)...", flush=True)

        vmap_env = VmapEnvWrapper(
            n_envs=self.n_envs,
            opponent_paths=opponent_paths,
            config=WorldConfig(),
            max_ticks=self.max_ticks,
            fighter_mass=70.0,
            opponent_mass=70.0,
            seed=42,
            debug=self.debug,
        )

        if self.verbose:
            print("✅ JAX vmap environment created successfully!", flush=True)

        adapted_env = self.vmap_adapter_cls(vmap_env)

        if self.verbose:
            print("✅ Environment adapter created!", flush=True)

        normalized_env = VecNormalize(
            adapted_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        return VecCheckNan(normalized_env, raise_exception=True, warn_once=True)

    def _create_dummy_envs(self, level):
        env_fns = []
        for i in range(self.n_envs):
            opponent_idx = i % len(level.opponents)
            opponent_path = level.opponents[opponent_idx]

            level_name = level.name.replace(" ", "_").lower()
            monitor_file = str(self.logs_dir / f"{level_name}_env_{i}")

            env_fn = lambda opp_path=opponent_path, mfile=monitor_file: Monitor(
                self.create_env_fn(opp_path),
                mfile,
                allow_early_resets=True,
            )
            env_fns.append(env_fn)

        dummy_env = DummyVecEnv(env_fns)
        normalized_env = VecNormalize(
            dummy_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        return VecCheckNan(normalized_env, raise_exception=True, warn_once=True)


class ModelFactory:
    """Creates RL models with project-stable defaults."""

    def __init__(self, *, logs_dir: Path, verbose: bool):
        self.logs_dir = Path(logs_dir)
        self.verbose = verbose

    def create_model(self, *, algorithm: str, envs, device: str):
        algo = algorithm.lower()
        actual_device = "cpu" if device == "auto" else device

        if algo == "ppo":
            if self.verbose:
                print(f"Initializing PPO with device: {actual_device} (forced CPU for MlpPolicy)")
            from ..utils.stable_ppo_config import get_stable_ppo_config

            stable_config = get_stable_ppo_config()
            model = PPO(
                "MlpPolicy",
                envs,
                device=actual_device,
                **stable_config,
            )
            model.verbose = 2 if self.verbose else 0
            model.tensorboard_log = str(self.logs_dir / "tensorboard")
            return model

        if algo == "sac":
            if self.verbose:
                print(f"Initializing SAC with device: {actual_device} (forced CPU for MlpPolicy)")
            return SAC(
                "MlpPolicy",
                envs,
                learning_rate=3e-4,
                buffer_size=50000,
                learning_starts=100,
                batch_size=64,
                tau=0.01,
                gamma=0.99,
                ent_coef="auto",
                target_update_interval=1,
                gradient_steps=1,
                verbose=1 if self.verbose else 0,
                tensorboard_log=str(self.logs_dir / "tensorboard"),
                device=actual_device,
            )

        raise ValueError(f"Unknown algorithm: {algorithm}")
