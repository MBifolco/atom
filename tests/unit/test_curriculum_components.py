import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from src.training.trainers.curriculum_components import (
    CallbackStepProcessor,
    CheckpointBundle,
    CheckpointRecoveryError,
    GraduationPolicy,
    NaNRecoveryError,
    LevelRunner,
    LevelTransitionStateMachine,
    ProgressReporter,
    RecoveryManager,
    TrainingLoopExecutionError,
)


def _make_progress():
    return SimpleNamespace(
        current_level=0,
        episodes_at_level=0,
        wins_at_level=0,
        recent_episodes=[],
        total_episodes=0,
        total_wins=0,
    )


def _make_level():
    return SimpleNamespace(
        min_episodes=10,
        graduation_episodes=5,
        graduation_win_rate=0.8,
    )


def _make_logger():
    logger = logging.getLogger("test_curriculum_components")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    return logger


def test_graduation_policy_respects_override():
    policy = GraduationPolicy(override_episodes_per_level=3, min_overall_win_rate=0.5)
    progress = _make_progress()
    level = _make_level()

    progress.episodes_at_level = 2
    decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
    assert not decision.should_graduate
    assert decision.reason == "override"

    progress.episodes_at_level = 3
    decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
    assert decision.should_graduate
    assert decision.reason == "override"


def test_graduation_policy_requires_recent_and_overall():
    policy = GraduationPolicy(override_episodes_per_level=None, min_overall_win_rate=0.5)
    progress = _make_progress()
    level = _make_level()

    progress.episodes_at_level = 20
    progress.wins_at_level = 8  # 40% overall
    progress.recent_episodes = [True, True, True, True, True]  # 100% recent
    decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
    assert not decision.should_graduate
    assert decision.reason == "overall_too_low"
    assert decision.recent_passed
    assert not decision.overall_passed

    progress.wins_at_level = 12  # 60% overall
    decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
    assert decision.should_graduate
    assert decision.reason == "passed"
    assert decision.recent_passed
    assert decision.overall_passed


def test_progress_reporter_updates_recent_lists():
    logger = _make_logger()
    reporter = ProgressReporter(logger)
    progress = _make_progress()
    level = _make_level()

    reporter.update_progress(
        progress=progress,
        level=level,
        won=True,
        reward=12.5,
        info={"reward_breakdown": {"damage": 1.0, "total": 1.0}},
    )

    assert progress.episodes_at_level == 1
    assert progress.wins_at_level == 1
    assert progress.total_episodes == 1
    assert progress.total_wins == 1
    assert progress.recent_episodes == [True]
    assert hasattr(progress, "recent_rewards")
    assert progress.recent_rewards == [12.5]
    assert hasattr(progress, "recent_reward_breakdowns")
    assert progress.recent_reward_breakdowns == [{"damage": 1.0, "total": 1.0}]


def test_recovery_manager_error_helpers():
    logger = _make_logger()
    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
        )

        assert manager.is_nan_error(ValueError("invalid values: tensor nan"))
        assert not manager.is_nan_error(ValueError("different error"))
        assert manager.is_progress_conflict_error(
            RuntimeError("Only one live display may be active at once")
        )


def test_level_runner_retries_on_rich_progress_conflict():
    logger = _make_logger()

    class FakeModel:
        load_calls = 0

        def __init__(self):
            self.learning_rate = 1e-3
            self.learn_calls = 0
            self.saved = []

        def save(self, path):
            self.saved.append(str(path))

        def learn(self, **kwargs):
            self.learn_calls += 1
            if self.learn_calls == 1:
                raise RuntimeError("Only one live display may be active at once")
            return self

        @classmethod
        def load(cls, path, env=None):
            cls.load_calls += 1
            loaded = cls()
            loaded.learning_rate = 1e-3
            return loaded

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        model = FakeModel()
        callback = SimpleNamespace(n_calls=0, episode_rewards=[])

        result_model = runner.run(
            model=model,
            envs=object(),
            callback=callback,
            total_timesteps=100,
            verbose=True,
            current_level_getter=lambda: 0,
        )

        assert isinstance(result_model, FakeModel)
        assert result_model.learn_calls == 2


def test_level_runner_recovers_from_nan_with_checkpoint():
    logger = _make_logger()

    class FakeModel:
        load_calls = 0
        nan_raised = False

        def __init__(self):
            self.learning_rate = 2e-3
            self.learn_calls = 0

        def save(self, path):
            pass

        def learn(self, **kwargs):
            self.learn_calls += 1
            if not FakeModel.nan_raised:
                FakeModel.nan_raised = True
                raise ValueError("invalid values in Normal distribution: nan")
            return self

        @classmethod
        def load(cls, path, env=None):
            cls.load_calls += 1
            return cls()

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        model = FakeModel()
        callback = SimpleNamespace(n_calls=42, episode_rewards=[1.0, 2.0, 3.0])

        result_model = runner.run(
            model=model,
            envs=object(),
            callback=callback,
            total_timesteps=1000,
            verbose=False,
            current_level_getter=lambda: 1,
        )

        assert isinstance(result_model, FakeModel)
        assert FakeModel.load_calls == 1
        assert result_model.learning_rate == 1e-3


def test_level_runner_raises_nan_recovery_error_after_retry_exhaustion():
    logger = _make_logger()

    class FakeModel:
        def __init__(self):
            self.learning_rate = 2e-3

        def save(self, path):
            pass

        def learn(self, **kwargs):
            raise ValueError("invalid values in Normal distribution: nan")

        @classmethod
        def load(cls, path, env=None):
            return cls()

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=2,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=17, episode_rewards=[1.0, 2.0])

        try:
            runner.run(
                model=FakeModel(),
                envs=object(),
                callback=callback,
                total_timesteps=1000,
                verbose=False,
                current_level_getter=lambda: 3,
            )
            assert False, "Expected NaNRecoveryError"
        except NaNRecoveryError as exc:
            assert exc.details.level == 3
            assert exc.details.completed_steps == 17
            assert exc.details.total_timesteps == 1000
            assert exc.details.nan_retries == 2
            assert exc.details.debug_path is not None


def test_level_runner_wraps_checkpoint_recovery_failures():
    logger = _make_logger()

    class FakeModel:
        def __init__(self):
            self.learning_rate = 2e-3
            self.learn_calls = 0

        def save(self, path):
            pass

        def learn(self, **kwargs):
            self.learn_calls += 1
            raise ValueError("invalid values in Normal distribution: nan")

        @classmethod
        def load(cls, path, env=None):
            raise RuntimeError("checkpoint load failed")

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=9, episode_rewards=[])

        try:
            runner.run(
                model=FakeModel(),
                envs=object(),
                callback=callback,
                total_timesteps=200,
                verbose=False,
                current_level_getter=lambda: 2,
            )
            assert False, "Expected CheckpointRecoveryError"
        except CheckpointRecoveryError as exc:
            assert exc.details.level == 2
            assert exc.details.nan_retries == 1
            assert exc.details.checkpoint_path is not None


def test_level_runner_wraps_unexpected_errors_with_context():
    logger = _make_logger()

    class FakeModel:
        def save(self, path):
            pass

        def learn(self, **kwargs):
            raise RuntimeError("boom")

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=5, episode_rewards=[])

        try:
            runner.run(
                model=FakeModel(),
                envs=object(),
                callback=callback,
                total_timesteps=500,
                verbose=False,
                current_level_getter=lambda: 1,
            )
            assert False, "Expected TrainingLoopExecutionError"
        except TrainingLoopExecutionError as exc:
            assert "Unexpected training loop failure" in str(exc)
            assert exc.details.level == 1
            assert exc.details.completed_steps == 5
            assert exc.details.total_timesteps == 500


def test_recovery_manager_checkpoint_roundtrip_mid_level_state():
    logger = _make_logger()

    class FakeModel:
        def save(self, path):
            Path(path).write_text("model")

    class FakeVecNormalize:
        def save(self, path):
            Path(path).write_text("vecnorm")

    class FakeVecWrapper:
        def __init__(self):
            self.venv = FakeVecNormalize()

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=10,
        )

        state = {
            "schema_version": 1,
            "progress": {
                "current_level": 2,
                "episodes_at_level": 321,
                "wins_at_level": 199,
            },
            "callback": {
                "episode_rewards": [1.0, 2.0, 3.0],
            },
        }

        bundle = manager.save_checkpoint_bundle(
            model=FakeModel(),
            envs=FakeVecWrapper(),
            step=1234,
            training_state=state,
            verbose=False,
        )

        assert isinstance(bundle, CheckpointBundle)
        assert bundle.model_path.exists()
        assert bundle.state_path is not None and bundle.state_path.exists()
        assert bundle.vecnormalize_path is not None and bundle.vecnormalize_path.exists()

        restored_state = manager.load_checkpoint_training_state(bundle)
        assert restored_state == state
        assert restored_state["progress"]["current_level"] == 2
        assert restored_state["progress"]["episodes_at_level"] == 321


def test_recovery_manager_checkpoint_roundtrip_post_transition_state():
    logger = _make_logger()

    class FakeModel:
        def save(self, path):
            Path(path).write_text("model")

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=10,
        )

        transitioned_state = {
            "schema_version": 1,
            "progress": {
                "current_level": 3,
                "episodes_at_level": 0,
                "wins_at_level": 0,
                "graduated_levels": ["Fundamentals", "Basic Skills", "Intermediate"],
                "recent_episodes": [],
            },
            "callback": {
                "episode_rewards": [10.0, 20.0],
                "episode_wins": [True, False],
            },
        }

        bundle = manager.save_checkpoint_bundle(
            model=FakeModel(),
            envs=object(),
            step=2000,
            training_state=transitioned_state,
            verbose=False,
        )

        restored = manager.load_checkpoint_training_state(bundle)
        assert restored == transitioned_state
        assert restored["progress"]["current_level"] == 3
        assert restored["progress"]["graduated_levels"][-1] == "Intermediate"


def test_level_runner_restores_training_state_on_nan_restart():
    logger = _make_logger()

    class FakeModel:
        load_calls = 0
        nan_raised = False

        def __init__(self):
            self.learning_rate = 1e-3

        def save(self, path):
            Path(path).write_text("model")

        def learn(self, **kwargs):
            if not FakeModel.nan_raised:
                FakeModel.nan_raised = True
                raise ValueError("invalid values in Normal distribution: nan")
            return self

        @classmethod
        def load(cls, path, env=None):
            cls.load_calls += 1
            return cls()

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
            base_backoff_seconds=0.0,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=64, episode_rewards=[1.0, 2.0], episode_wins=[True])

        restored_states = []
        current_state = {
            "schema_version": 1,
            "progress": {"current_level": 1, "episodes_at_level": 42},
            "callback": {"episode_rewards": [5.0], "episode_wins": [True]},
        }

        runner.run(
            model=FakeModel(),
            envs=object(),
            callback=callback,
            total_timesteps=200,
            verbose=False,
            current_level_getter=lambda: 1,
            training_state_getter=lambda: current_state,
            training_state_restorer=lambda state: restored_states.append(state),
            model_update_fn=lambda _model: None,
            sleep_fn=lambda _seconds: None,
        )

        assert FakeModel.load_calls == 1
        assert len(restored_states) == 1
        assert restored_states[0] == current_state


def test_level_runner_uses_restored_env_for_checkpoint_load():
    logger = _make_logger()

    class FakeModel:
        load_calls = 0
        load_envs = []
        nan_raised = False

        def __init__(self):
            self.learning_rate = 1e-3

        def save(self, path):
            Path(path).write_text("model")

        def learn(self, **kwargs):
            if not FakeModel.nan_raised:
                FakeModel.nan_raised = True
                raise ValueError("invalid values in Normal distribution: nan")
            return self

        @classmethod
        def load(cls, path, env=None):
            cls.load_calls += 1
            cls.load_envs.append(env)
            return cls()

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=1,
            max_retries=3,
            base_backoff_seconds=0.0,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=12, episode_rewards=[])

        old_env = object()
        new_env = object()
        env_box = {"env": old_env}

        runner.run(
            model=FakeModel(),
            envs=old_env,
            callback=callback,
            total_timesteps=100,
            verbose=False,
            current_level_getter=lambda: 0,
            training_state_getter=lambda: {"progress": {"current_level": 1}},
            training_state_restorer=lambda state: env_box.__setitem__("env", new_env),
            model_update_fn=lambda _model: None,
            env_getter=lambda: env_box["env"],
            sleep_fn=lambda _seconds: None,
        )

        assert FakeModel.load_calls == 1
        assert FakeModel.load_envs[-1] is new_env


def test_level_runner_respects_initial_step_budget():
    logger = _make_logger()

    class FakeModel:
        learned_steps = []

        def save(self, path):
            pass

        def learn(self, **kwargs):
            FakeModel.learned_steps.append(kwargs["total_timesteps"])
            return self

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=10,
            max_retries=2,
        )
        runner = LevelRunner(logger=logger, recovery_manager=manager)
        callback = SimpleNamespace(n_calls=0, episode_rewards=[])

        runner.run(
            model=FakeModel(),
            envs=object(),
            callback=callback,
            total_timesteps=1000,
            initial_step=250,
            verbose=False,
            current_level_getter=lambda: 0,
        )

        assert FakeModel.learned_steps == [750]


def test_recovery_manager_finds_latest_checkpoint_bundle():
    logger = _make_logger()

    class FakeModel:
        def save(self, path):
            Path(path).write_text("model")

    with TemporaryDirectory() as tmpdir:
        manager = RecoveryManager(
            models_dir=Path(tmpdir),
            logs_dir=Path(tmpdir),
            logger=logger,
            checkpoint_interval=10,
        )

        manager.save_checkpoint_bundle(
            model=FakeModel(),
            envs=object(),
            step=100,
            training_state={"progress": {"current_level": 1}},
            verbose=False,
        )
        manager.save_checkpoint_bundle(
            model=FakeModel(),
            envs=object(),
            step=400,
            training_state={"progress": {"current_level": 2}},
            verbose=False,
        )

        bundles = manager.list_checkpoint_bundles()
        assert [bundle.step for bundle in bundles] == [100, 400]
        latest = manager.find_latest_checkpoint_bundle()
        assert latest is not None
        assert latest.step == 400
        assert latest.state_path is not None and latest.state_path.exists()


def test_level_transition_state_machine_resets_level_metrics():
    progress = SimpleNamespace(
        current_level=0,
        episodes_at_level=123,
        wins_at_level=77,
        recent_episodes=[True, False, True],
        graduated_levels=[],
    )
    curriculum = [
        SimpleNamespace(name="Fundamentals"),
        SimpleNamespace(name="Basic Skills"),
    ]

    result = LevelTransitionStateMachine.advance(progress, curriculum)

    assert result.graduated_level_name == "Fundamentals"
    assert result.next_level_index == 1
    assert not result.completed
    assert progress.current_level == 1
    assert progress.episodes_at_level == 0
    assert progress.wins_at_level == 0
    assert progress.recent_episodes == []
    assert progress.graduated_levels == ["Fundamentals"]


def test_level_transition_state_machine_marks_curriculum_complete():
    progress = SimpleNamespace(
        current_level=1,
        episodes_at_level=50,
        wins_at_level=45,
        recent_episodes=[True] * 10,
        graduated_levels=["Fundamentals"],
    )
    curriculum = [
        SimpleNamespace(name="Fundamentals"),
        SimpleNamespace(name="Basic Skills"),
    ]

    result = LevelTransitionStateMachine.advance(progress, curriculum)

    assert result.graduated_level_name == "Basic Skills"
    assert result.next_level_index == 2
    assert result.completed
    assert progress.current_level == 2
    assert progress.graduated_levels == ["Fundamentals", "Basic Skills"]


def test_callback_step_processor_updates_episode_tracking_and_trainer():
    trainer = SimpleNamespace(
        progressive_recorder=None,
        verbose=False,
        logger=Mock(),
        progress=SimpleNamespace(current_level=0),
        curriculum=[SimpleNamespace(name="Fundamentals")],
        update_progress=Mock(),
        should_graduate=Mock(return_value=False),
        advance_level=Mock(),
    )
    processor = CallbackStepProcessor(curriculum_trainer=trainer)

    episode_rewards = []
    episode_wins = []
    reward_components = []
    infos = [{"episode": {"r": 15.0}, "won": True, "reward_breakdown": {"damage": 2.0}}]

    should_continue = processor.process_infos(infos, episode_rewards, episode_wins, reward_components)

    assert should_continue
    assert episode_rewards == [15.0]
    assert episode_wins == [True]
    assert reward_components == [{"damage": 2.0}]
    trainer.update_progress.assert_called_once_with(True, 15.0, infos[0])
    trainer.should_graduate.assert_called_once()
    trainer.advance_level.assert_not_called()


def test_callback_step_processor_stops_when_curriculum_completes():
    progress = SimpleNamespace(current_level=0)
    trainer = SimpleNamespace(
        progressive_recorder=None,
        verbose=False,
        logger=Mock(),
        progress=progress,
        curriculum=[SimpleNamespace(name="Fundamentals")],
        update_progress=Mock(),
        should_graduate=Mock(return_value=True),
        advance_level=Mock(side_effect=lambda: setattr(progress, "current_level", 1)),
    )
    processor = CallbackStepProcessor(curriculum_trainer=trainer)

    should_continue = processor.process_infos(
        [{"episode": {"r": 5.0}, "won": False}],
        [],
        [],
        [],
    )

    assert not should_continue
    trainer.advance_level.assert_called_once()
    trainer.logger.info.assert_called_with("Curriculum complete - stopping training early")


def test_callback_step_processor_uses_record_replay_fn_when_enabled():
    trainer = SimpleNamespace(
        progressive_recorder=SimpleNamespace(should_record=lambda episode_num, total_episodes: True),
        verbose=False,
        logger=Mock(),
        progress=SimpleNamespace(current_level=0),
        curriculum=[SimpleNamespace(name="Fundamentals")],
        update_progress=Mock(),
        should_graduate=Mock(return_value=False),
        advance_level=Mock(),
    )
    record_fn = Mock()
    processor = CallbackStepProcessor(
        curriculum_trainer=trainer,
        record_evaluation_replay_fn=record_fn,
    )

    should_continue = processor.process_infos(
        [{"episode": {"r": 7.5}, "won": True}],
        [],
        [],
        [],
    )

    assert should_continue
    record_fn.assert_called_once_with(1, 10000)
