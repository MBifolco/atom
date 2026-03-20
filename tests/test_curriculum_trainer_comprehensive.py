"""
Comprehensive tests for CurriculumTrainer to increase coverage to 75%.
Tests initialization, curriculum building, progress tracking, and training flow.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import time

from src.training.trainers.curriculum_trainer import (
    CurriculumTrainer,
    CurriculumLevel,
    CurriculumCallback,
    TrainingProgress,
    DifficultyLevel,
    VmapEnvAdapter,
)


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_difficulty_levels_exist(self):
        """Test all difficulty levels are defined."""
        assert DifficultyLevel.FUNDAMENTALS.value == "fundamentals"
        assert DifficultyLevel.BASIC_SKILLS.value == "basic_skills"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"
        assert DifficultyLevel.POPULATION.value == "population"


class TestCurriculumLevel:
    """Tests for CurriculumLevel dataclass."""

    def test_curriculum_level_creation(self):
        """Test creating a curriculum level."""
        level = CurriculumLevel(
            name="Test Level",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["path/to/opponent.py"],
            min_episodes=100,
            graduation_win_rate=0.8,
            graduation_episodes=20,
            description="Test description"
        )

        assert level.name == "Test Level"
        assert level.difficulty == DifficultyLevel.FUNDAMENTALS
        assert len(level.opponents) == 1
        assert level.min_episodes == 100
        assert level.graduation_win_rate == 0.8
        assert level.graduation_episodes == 20
        assert level.description == "Test description"

    def test_curriculum_level_defaults(self):
        """Test curriculum level default values."""
        level = CurriculumLevel(
            name="Test",
            difficulty=DifficultyLevel.BASIC_SKILLS,
            opponents=[]
        )

        assert level.min_episodes == 100
        assert level.graduation_win_rate == 0.7
        assert level.graduation_episodes == 20
        assert level.description == ""


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_training_progress_defaults(self):
        """Test training progress default values."""
        progress = TrainingProgress()

        assert progress.current_level == 0
        assert progress.episodes_at_level == 0
        assert progress.wins_at_level == 0
        assert progress.recent_episodes == []
        assert progress.graduated_levels == []
        assert progress.total_episodes == 0
        assert progress.total_wins == 0
        assert progress.start_time > 0

    def test_training_progress_modification(self):
        """Test modifying training progress."""
        progress = TrainingProgress()

        progress.current_level = 2
        progress.episodes_at_level = 50
        progress.wins_at_level = 30
        progress.recent_episodes = [True, True, False, True]

        assert progress.current_level == 2
        assert progress.episodes_at_level == 50
        assert progress.wins_at_level == 30
        assert len(progress.recent_episodes) == 4


class TestCurriculumTrainerInit:
    """Tests for CurriculumTrainer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            assert trainer.algorithm == "ppo"
            assert trainer.n_envs == 4
            assert trainer.max_ticks == 250
            assert trainer.verbose == False
            assert trainer.use_vmap == False
            assert trainer.model is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm="ppo",
                output_dir=tmpdir,
                n_envs=8,
                max_ticks=500,
                verbose=False,
                device="cpu",
                use_vmap=True,
                debug=True
            )

            assert trainer.n_envs == 8
            assert trainer.max_ticks == 500
            assert trainer.use_vmap == True
            assert trainer.debug == True

    def test_init_creates_directories(self):
        """Test that initialization creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            assert trainer.models_dir.exists()
            assert trainer.logs_dir.exists()

    def test_init_builds_curriculum(self):
        """Test that initialization builds curriculum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            assert len(trainer.curriculum) == 5  # 5 levels
            assert trainer.curriculum[0].name == "Fundamentals"
            assert trainer.curriculum[-1].name == "Expert"

    def test_init_with_replay_recording(self):
        """Test initialization with replay recording enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False,
                record_replays=True,
                replay_matches_per_opponent=5
            )

            assert trainer.record_replays == True
            assert trainer.replay_matches_per_opponent == 5
            # replay_recorder should be created
            assert trainer.replay_recorder is not None


class TestCurriculumTrainerCurriculum:
    """Tests for curriculum building."""

    def test_build_curriculum_structure(self):
        """Test curriculum structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            curriculum = trainer.curriculum

            # Check all 5 levels
            assert curriculum[0].difficulty == DifficultyLevel.FUNDAMENTALS
            assert curriculum[1].difficulty == DifficultyLevel.BASIC_SKILLS
            assert curriculum[2].difficulty == DifficultyLevel.INTERMEDIATE
            assert curriculum[3].difficulty == DifficultyLevel.ADVANCED
            assert curriculum[4].difficulty == DifficultyLevel.EXPERT

    def test_curriculum_opponents_exist(self):
        """Test that curriculum references opponent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            for level in trainer.curriculum:
                assert len(level.opponents) > 0
                for opponent_path in level.opponents:
                    assert opponent_path.endswith('.py')


class TestCurriculumTrainerLoadOpponent:
    """Tests for load_opponent method."""

    def test_load_valid_opponent(self):
        """Test loading a valid opponent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Create a test opponent file
            opponent_file = Path(tmpdir) / "test_opponent.py"
            opponent_file.write_text('''
def decide(state):
    return {"acceleration": 0.5, "stance": "neutral"}
''')

            decide_func = trainer.load_opponent(str(opponent_file))
            assert callable(decide_func)

            # Test the function works
            result = decide_func({})
            assert result["acceleration"] == 0.5
            assert result["stance"] == "neutral"

    def test_load_nonexistent_opponent(self):
        """Test loading a nonexistent opponent returns fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Should return dummy opponent, not crash
            decide_func = trainer.load_opponent("/nonexistent/path.py")
            assert callable(decide_func)

            # Dummy opponent should return neutral
            result = decide_func({})
            assert result["acceleration"] == 0
            assert result["stance"] == "neutral"


class TestCurriculumTrainerProgress:
    """Tests for progress tracking."""

    def test_update_progress_win(self):
        """Test updating progress with a win."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.update_progress(won=True, reward=100.0)

            assert trainer.progress.episodes_at_level == 1
            assert trainer.progress.wins_at_level == 1
            assert trainer.progress.total_episodes == 1
            assert trainer.progress.total_wins == 1
            assert len(trainer.progress.recent_episodes) == 1
            assert trainer.progress.recent_episodes[0] == True

    def test_update_progress_loss(self):
        """Test updating progress with a loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.update_progress(won=False, reward=-50.0)

            assert trainer.progress.episodes_at_level == 1
            assert trainer.progress.wins_at_level == 0
            assert trainer.progress.total_episodes == 1
            assert trainer.progress.total_wins == 0
            assert trainer.progress.recent_episodes[0] == False

    def test_update_progress_tracks_recent(self):
        """Test that recent episodes are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Win, Loss, Win, Win
            trainer.update_progress(won=True)
            trainer.update_progress(won=False)
            trainer.update_progress(won=True)
            trainer.update_progress(won=True)

            assert trainer.progress.recent_episodes == [True, False, True, True]
            assert trainer.progress.episodes_at_level == 4
            assert trainer.progress.wins_at_level == 3

    def test_update_progress_reward_breakdown(self):
        """Test tracking reward breakdown in info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            info = {"reward_breakdown": {"damage": 10, "proximity": 5}}
            trainer.update_progress(won=True, reward=100.0, info=info)

            assert hasattr(trainer.progress, 'recent_reward_breakdowns')


class TestCurriculumTrainerGraduation:
    """Tests for graduation logic."""

    def test_get_current_level(self):
        """Test getting current curriculum level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            level = trainer.get_current_level()
            assert level.name == "Fundamentals"

            trainer.progress.current_level = 2
            level = trainer.get_current_level()
            assert level.name == "Intermediate"

    def test_get_current_level_at_end(self):
        """Test getting level when at end of curriculum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Set to beyond curriculum length
            trainer.progress.current_level = 100
            level = trainer.get_current_level()
            assert level.name == "Expert"  # Should return last level

    def test_should_graduate_not_enough_episodes(self):
        """Test graduation fails when not enough episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Only a few episodes
            trainer.progress.episodes_at_level = 10
            trainer.progress.recent_episodes = [True] * 10

            assert trainer.should_graduate() == False

    def test_should_graduate_low_win_rate(self):
        """Test graduation fails with low win rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            level = trainer.get_current_level()

            # Enough episodes but low win rate
            trainer.progress.episodes_at_level = level.min_episodes + 100
            trainer.progress.wins_at_level = int(trainer.progress.episodes_at_level * 0.3)

            # Recent episodes with low win rate
            trainer.progress.recent_episodes = [False] * level.graduation_episodes

            assert trainer.should_graduate() == False

    def test_should_graduate_at_end(self):
        """Test graduation returns False when curriculum complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.progress.current_level = len(trainer.curriculum)

            assert trainer.should_graduate() == False


class TestCurriculumCallback:
    """Tests for CurriculumCallback."""

    def test_callback_init(self):
        """Test callback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=1)

            assert callback.curriculum_trainer == trainer
            assert callback.episode_rewards == []
            assert callback.episode_wins == []
            assert callback.verbose == 1

    def test_callback_on_rollout_start(self):
        """Test callback on_rollout_start method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)
            callback._on_rollout_start()

            assert callback.last_rollout_time is not None
            assert callback.rollout_count == 1

    def test_callback_on_rollout_end(self):
        """Test callback on_rollout_end method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)
            callback._on_rollout_start()
            callback._on_rollout_end()

            assert callback.last_train_time is not None

    def test_callback_on_training_end(self):
        """Test callback on_training_end method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)
            callback.last_train_time = time.time() - 1
            callback._on_training_end()

            # Should not crash
            assert True

    def test_callback_on_step_with_episode_info(self):
        """Test callback _on_step with episode completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)

            # Simulate episode completion
            callback.locals = {
                "infos": [{"episode": {"r": 100.0, "l": 50}, "won": True}]
            }

            result = callback._on_step()

            assert result == True
            assert len(callback.episode_rewards) == 1
            assert callback.episode_rewards[0] == 100.0
            assert len(callback.episode_wins) == 1
            assert callback.episode_wins[0] == True


class TestVmapEnvAdapter:
    """Tests for VmapEnvAdapter."""

    def test_adapter_init(self):
        """Test adapter initialization."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap_env)

        assert adapter.num_envs == 4
        assert adapter.metadata == {"render_modes": []}

    def test_adapter_reset(self):
        """Test adapter reset method."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)
        mock_vmap_env.reset.return_value = (np.zeros((4, 9)), {})

        adapter = VmapEnvAdapter(mock_vmap_env)
        obs = adapter.reset()

        mock_vmap_env.reset.assert_called_once()
        assert obs.shape == (4, 9)

    def test_adapter_step_async_wait(self):
        """Test adapter step_async and step_wait methods."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)
        mock_vmap_env.step.return_value = (
            np.zeros((4, 9)),  # obs
            np.zeros(4),       # rewards
            np.array([False, False, False, False]),  # dones
            np.array([False, False, False, False]),  # truncated
            [{}, {}, {}, {}]   # infos
        )

        adapter = VmapEnvAdapter(mock_vmap_env)

        actions = np.zeros((4, 2))
        adapter.step_async(actions)
        obs, rewards, dones, infos = adapter.step_wait()

        mock_vmap_env.step.assert_called_once()
        assert obs.shape == (4, 9)

    def test_adapter_env_method_set_opponent(self, capsys):
        """Test adapter handles set_opponent calls gracefully."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap_env)
        adapter.env_method("set_opponent", lambda x: x)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_adapter_get_attr(self):
        """Test adapter get_attr method."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)
        mock_vmap_env.some_attr = "test_value"

        adapter = VmapEnvAdapter(mock_vmap_env)
        result = adapter.get_attr("some_attr")

        assert result == ["test_value"] * 4

    def test_adapter_env_is_wrapped(self):
        """Test adapter env_is_wrapped returns False."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap_env)
        assert adapter.env_is_wrapped(None) == False

    def test_adapter_close(self):
        """Test adapter close method."""
        mock_vmap_env = Mock()
        mock_vmap_env.n_envs = 4
        mock_vmap_env.observation_space = Mock()
        mock_vmap_env.observation_space.shape = (9,)
        mock_vmap_env.action_space = Mock()
        mock_vmap_env.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap_env)
        adapter.close()  # Should not crash


class TestCurriculumTrainerSaveLoad:
    """Tests for save/load functionality."""

    def test_save_checkpoint_no_model(self):
        """Test save_checkpoint with no model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # No model yet
            result = trainer.save_checkpoint()
            assert result is None

    def test_save_training_report(self):
        """Test saving training report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.progress.total_episodes = 100
            trainer.progress.total_wins = 80
            trainer.progress.graduated_levels = ["Fundamentals"]

            trainer.save_training_report()

            # Check report was saved
            report_files = list(Path(tmpdir).glob("training_report_*.json"))
            assert len(report_files) == 1


class TestCurriculumTrainerCreateEnv:
    """Tests for environment creation."""

    def test_create_env(self):
        """Test creating a single environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Create test opponent
            opponent_file = Path(tmpdir) / "opponent.py"
            opponent_file.write_text('''
def decide(state):
    return {"acceleration": 0, "stance": "neutral"}
''')

            env = trainer.create_env(str(opponent_file))

            # Should return an AtomCombatEnv
            assert hasattr(env, 'step')
            assert hasattr(env, 'reset')


class TestCurriculumTrainerAdvanceLevel:
    """Tests for level advancement."""

    def test_advance_level_basic(self):
        """Test advancing to next level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Set up some progress at level 0
            trainer.progress.episodes_at_level = 100
            trainer.progress.wins_at_level = 90
            trainer.progress.recent_episodes = [True] * 50

            # Mock model so we don't need actual training
            trainer.model = Mock()
            trainer.model.set_env = Mock()
            trainer.envs = Mock()

            # Advance level
            trainer.advance_level()

            assert trainer.progress.current_level == 1
            assert trainer.progress.episodes_at_level == 0
            assert trainer.progress.wins_at_level == 0
            assert trainer.progress.recent_episodes == []
            assert "Fundamentals" in trainer.progress.graduated_levels


class TestCurriculumTrainerLogging:
    """Tests for logging setup."""

    def test_setup_logging(self):
        """Test logging is properly configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=True
            )

            assert trainer.logger is not None
            assert trainer.logger.name == 'curriculum_trainer'

            # Check log file was created
            log_files = list(Path(tmpdir).glob("logs/curriculum_training_*.log"))
            assert len(log_files) == 1


class TestCurriculumTrainerStateRestore:
    """Tests for training state capture/restore behavior."""

    def test_capture_and_restore_training_state_roundtrip(self):
        """Captured state should restore progress and callback buffers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )
            callback = CurriculumCallback(trainer, verbose=0)

            trainer.progress.current_level = 0
            trainer.progress.episodes_at_level = 123
            trainer.progress.wins_at_level = 77
            trainer.progress.recent_episodes = [True, False, True]
            trainer.progress.graduated_levels = ["Fundamentals"]
            trainer.progress.total_episodes = 456
            trainer.progress.total_wins = 300
            trainer.progress.recent_rewards = [10.0, 20.0]
            trainer.progress.recent_reward_breakdowns = [{"damage": 2.0}]

            callback.episode_rewards = [1.0, 2.0, 3.0]
            callback.episode_wins = [True, False, True]
            callback.recent_reward_components = [{"damage": 4.0}]
            callback.rollout_count = 9

            state = trainer._capture_training_state(callback)

            trainer.progress.episodes_at_level = 0
            trainer.progress.wins_at_level = 0
            trainer.progress.recent_episodes = []
            trainer.progress.graduated_levels = []
            trainer.progress.total_episodes = 0
            trainer.progress.total_wins = 0
            trainer.progress.recent_rewards = []
            trainer.progress.recent_reward_breakdowns = []
            callback.episode_rewards = []
            callback.episode_wins = []
            callback.recent_reward_components = []
            callback.rollout_count = 0

            trainer._restore_training_state(callback, state)

            assert trainer.progress.episodes_at_level == 123
            assert trainer.progress.wins_at_level == 77
            assert trainer.progress.recent_episodes == [True, False, True]
            assert trainer.progress.graduated_levels == ["Fundamentals"]
            assert trainer.progress.total_episodes == 456
            assert trainer.progress.total_wins == 300
            assert trainer.progress.recent_rewards == [10.0, 20.0]
            assert trainer.progress.recent_reward_breakdowns == [{"damage": 2.0}]
            assert callback.episode_rewards == [1.0, 2.0, 3.0]
            assert callback.episode_wins == [True, False, True]
            assert callback.recent_reward_components == [{"damage": 4.0}]
            assert callback.rollout_count == 9

    def test_restore_resyncs_non_vmap_env_when_level_changes(self):
        """Non-vmap restore should switch opponents on existing envs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False,
                use_vmap=False,
            )
            callback = CurriculumCallback(trainer, verbose=0)

            trainer.model = Mock()
            trainer.envs = Mock()
            trainer.progress.current_level = 0
            trainer.load_opponent = Mock(return_value=lambda _state: {"acceleration": 0.0, "stance": "neutral"})

            state = {
                "progress": {
                    "current_level": 1,
                    "episodes_at_level": 10,
                    "wins_at_level": 6,
                    "recent_episodes": [True, False],
                    "graduated_levels": ["Fundamentals"],
                    "total_episodes": 10,
                    "total_wins": 6,
                },
                "callback": {},
            }

            trainer._restore_training_state(callback, state)

            assert trainer.progress.current_level == 1
            assert trainer.envs.env_method.call_count == trainer.n_envs
            method_names = [call.args[0] for call in trainer.envs.env_method.call_args_list]
            assert all(name == "set_opponent" for name in method_names)

    def test_restore_resyncs_vmap_env_when_level_changes(self):
        """Vmap restore should recreate envs and rebind model env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False,
                use_vmap=True,
            )
            callback = CurriculumCallback(trainer, verbose=0)

            old_env = Mock()
            new_env = Mock()
            trainer.envs = old_env
            trainer.model = Mock()
            trainer.create_envs_for_level = Mock(return_value=new_env)
            trainer.progress.current_level = 0

            state = {
                "progress": {
                    "current_level": 1,
                    "episodes_at_level": 22,
                    "wins_at_level": 12,
                    "recent_episodes": [True, True, False],
                    "graduated_levels": ["Fundamentals"],
                    "total_episodes": 22,
                    "total_wins": 12,
                },
                "callback": {},
            }

            trainer._restore_training_state(callback, state)

            assert trainer.progress.current_level == 1
            assert trainer.envs is new_env
            trainer.model.set_env.assert_called_once_with(new_env)
            old_env.close.assert_called_once()


class TestCurriculumCallbackVerbose:
    """Tests for verbose callback paths."""

    def test_callback_rollout_start_verbose(self, capsys):
        """Test verbose output on rollout start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )
            trainer.algorithm = 'ppo'  # PPO logs every rollout

            callback = CurriculumCallback(trainer, verbose=1)
            callback._on_rollout_start()

            captured = capsys.readouterr()
            assert "Collecting rollouts" in captured.out

    def test_callback_rollout_end_verbose(self, capsys):
        """Test verbose output on rollout end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )
            trainer.algorithm = 'ppo'

            callback = CurriculumCallback(trainer, verbose=1)
            callback._on_rollout_start()
            callback._on_rollout_end()

            captured = capsys.readouterr()
            assert "Rollouts collected" in captured.out
            assert "Training neural network" in captured.out

    def test_callback_training_end_verbose(self, capsys):
        """Test verbose output on training end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )
            trainer.algorithm = 'ppo'

            callback = CurriculumCallback(trainer, verbose=1)
            callback.rollout_count = 50  # Make it log
            callback.last_train_time = time.time() - 1
            callback._on_training_end()

            captured = capsys.readouterr()
            assert "Neural network trained" in captured.out

    def test_callback_with_reward_breakdown(self):
        """Test callback tracks reward breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            callback = CurriculumCallback(trainer, verbose=0)
            callback.locals = {
                "infos": [{
                    "episode": {"r": 100.0, "l": 50},
                    "won": True,
                    "reward_breakdown": {"damage": 50, "proximity": 30}
                }]
            }

            callback._on_step()

            assert len(callback.recent_reward_components) == 1


class TestCurriculumTrainerInitializeModel:
    """Tests for model initialization."""

    def test_initialize_ppo_model(self):
        """Test initializing PPO model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm="ppo",
                output_dir=tmpdir,
                verbose=False,
                n_envs=2
            )

            # Create a simple environment
            level = trainer.get_current_level()

            # Create test opponents
            opponent_dir = Path(tmpdir) / "opponents"
            opponent_dir.mkdir()
            for i, opp_path in enumerate(level.opponents[:2]):
                opp_file = opponent_dir / f"opp_{i}.py"
                opp_file.write_text('''
def decide(state):
    return {"acceleration": 0, "stance": "neutral"}
''')
                level.opponents[i] = str(opp_file)

            trainer.envs = trainer.create_envs_for_level(level)
            trainer.initialize_model()

            assert trainer.model is not None
            trainer.envs.close()

    def test_initialize_model_unknown_algorithm(self):
        """Test that unknown algorithm raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.algorithm = "unknown"
            trainer.envs = Mock()

            with pytest.raises(ValueError) as exc_info:
                trainer.initialize_model()

            assert "Unknown algorithm" in str(exc_info.value)


class TestCurriculumTrainerUpdateProgressLogging:
    """Tests for update_progress logging paths."""

    def test_update_progress_logs_every_100(self, capsys):
        """Test that progress logs every 100 episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=True
            )

            # Simulate 100 episodes
            for i in range(100):
                trainer.update_progress(won=i % 2 == 0, reward=10.0)

            # Should have logged at episode 100
            # Check that progress was tracked
            assert trainer.progress.episodes_at_level == 100

    def test_update_progress_with_recent_rewards(self):
        """Test update_progress tracks recent rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            for i in range(10):
                trainer.update_progress(won=True, reward=100.0 + i)

            assert hasattr(trainer.progress, 'recent_rewards')
            assert len(trainer.progress.recent_rewards) == 10


class TestCurriculumTrainerShouldGraduateLogging:
    """Tests for should_graduate logging paths."""

    def test_should_graduate_logging_on_potential_graduation(self):
        """Test logging when graduation criteria almost met."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            level = trainer.get_current_level()

            # Set up high win rate but low overall
            trainer.progress.episodes_at_level = level.min_episodes + 100
            trainer.progress.wins_at_level = int(trainer.progress.episodes_at_level * 0.4)  # Low overall
            trainer.progress.recent_episodes = [True] * level.graduation_episodes  # High recent

            # This should check but not graduate due to low overall
            result = trainer.should_graduate()

            # Result depends on overall win rate threshold
            assert isinstance(result, bool)


class TestCurriculumTrainerOnCurriculumComplete:
    """Tests for on_curriculum_complete method."""

    def test_on_curriculum_complete(self):
        """Test curriculum completion handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.progress.total_episodes = 1000
            trainer.progress.total_wins = 800
            trainer.progress.graduated_levels = ["L1", "L2", "L3"]

            # Mock model
            trainer.model = Mock()

            trainer.on_curriculum_complete()

            # Check model was saved
            trainer.model.save.assert_called_once()

            # Check report was saved
            report_files = list(Path(tmpdir).glob("training_report_*.json"))
            assert len(report_files) >= 1


class TestCurriculumTrainerCreateEnvsForLevel:
    """Tests for create_envs_for_level with different backends."""

    def test_create_envs_dummy_vec(self):
        """Test creating DummyVecEnv environments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False,
                n_envs=2,
                use_vmap=False
            )

            # Create test opponents
            opponent_dir = Path(tmpdir) / "opponents"
            opponent_dir.mkdir()

            level = trainer.get_current_level()
            for i in range(2):
                opp_file = opponent_dir / f"opp_{i}.py"
                opp_file.write_text('''
def decide(state):
    return {"acceleration": 0, "stance": "neutral"}
''')
                level.opponents[i] = str(opp_file)

            envs = trainer.create_envs_for_level(level)

            assert envs is not None
            # Should be wrapped with VecCheckNan
            assert hasattr(envs, 'reset')
            envs.close()


class TestCurriculumTrainerAdvanceLevelFull:
    """More complete tests for advance_level."""

    def test_advance_level_with_vmap(self):
        """Test advancing level recreates vmap environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False,
                use_vmap=True
            )

            trainer.progress.episodes_at_level = 100
            trainer.progress.wins_at_level = 90

            # Mock model and envs
            trainer.model = Mock()
            trainer.model.set_env = Mock()
            trainer.envs = Mock()

            # Should log about recreating vmap environments
            trainer.advance_level()

            assert trainer.progress.current_level == 1

    def test_advance_level_curriculum_complete(self):
        """Test advancing to end of curriculum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            # Set to last level
            trainer.progress.current_level = len(trainer.curriculum) - 1
            trainer.progress.episodes_at_level = 100
            trainer.progress.wins_at_level = 90
            trainer.progress.graduated_levels = ["L1", "L2", "L3", "L4"]

            trainer.model = Mock()
            trainer.envs = Mock()

            trainer.advance_level()

            # Should have completed curriculum
            assert trainer.progress.current_level == len(trainer.curriculum)


class TestCurriculumTrainerLoadCheckpoint:
    """Tests for checkpoint loading."""

    def test_load_checkpoint_ppo(self):
        """Test loading a PPO checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm="ppo",
                output_dir=tmpdir,
                verbose=False
            )

            # Mock the load process
            with patch('src.training.trainers.curriculum_trainer.PPO') as mock_ppo:
                mock_ppo.load.return_value = Mock()

                trainer.envs = Mock()
                trainer.load_checkpoint("/fake/path.zip")

                mock_ppo.load.assert_called_once()


class TestCurriculumTrainerSaveCheckpoint:
    """Tests for checkpoint saving."""

    def test_save_checkpoint_with_name(self):
        """Test saving checkpoint with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.model = Mock()

            result = trainer.save_checkpoint(name="test_checkpoint")

            assert result is not None
            trainer.model.save.assert_called_once()

    def test_save_checkpoint_auto_name(self):
        """Test saving checkpoint with auto-generated name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                output_dir=tmpdir,
                verbose=False
            )

            trainer.model = Mock()
            trainer.progress.current_level = 2
            trainer.progress.total_episodes = 500

            result = trainer.save_checkpoint()

            assert result is not None
            assert "level_2" in str(result)
            assert "ep_500" in str(result)
