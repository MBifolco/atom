"""
Comprehensive tests for curriculum trainer structures and callbacks.
Tests dataclasses, VmapEnvAdapter, and CurriculumCallback.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock, patch

from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    CurriculumLevel,
    TrainingProgress,
    CurriculumCallback,
    VmapEnvAdapter,
)


class TestDifficultyLevel:
    """Tests for DifficultyLevel enum."""

    def test_all_difficulty_levels_exist(self):
        """Test all difficulty levels are defined."""
        assert DifficultyLevel.FUNDAMENTALS.value == "fundamentals"
        assert DifficultyLevel.BASIC_SKILLS.value == "basic_skills"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"
        assert DifficultyLevel.POPULATION.value == "population"

    def test_difficulty_level_iteration(self):
        """Test iterating over difficulty levels."""
        levels = list(DifficultyLevel)
        assert len(levels) == 6


class TestCurriculumLevel:
    """Tests for CurriculumLevel dataclass."""

    def test_create_level_with_required_fields(self):
        """Test creating level with required fields only."""
        level = CurriculumLevel(
            name="Test Level",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=["fighter1.py", "fighter2.py"]
        )
        assert level.name == "Test Level"
        assert level.difficulty == DifficultyLevel.FUNDAMENTALS
        assert len(level.opponents) == 2

    def test_create_level_with_defaults(self):
        """Test default values are set correctly."""
        level = CurriculumLevel(
            name="Basic",
            difficulty=DifficultyLevel.BASIC_SKILLS,
            opponents=[]
        )
        assert level.min_episodes == 100
        assert level.graduation_win_rate == 0.7
        assert level.graduation_episodes == 20
        assert level.description == ""

    def test_create_level_with_custom_values(self):
        """Test creating level with custom values."""
        level = CurriculumLevel(
            name="Advanced",
            difficulty=DifficultyLevel.ADVANCED,
            opponents=["adv1.py", "adv2.py", "adv3.py"],
            min_episodes=200,
            graduation_win_rate=0.8,
            graduation_episodes=30,
            description="Advanced training level"
        )
        assert level.min_episodes == 200
        assert level.graduation_win_rate == 0.8
        assert level.graduation_episodes == 30
        assert level.description == "Advanced training level"


class TestTrainingProgress:
    """Tests for TrainingProgress dataclass."""

    def test_create_progress_with_defaults(self):
        """Test default progress values."""
        progress = TrainingProgress()
        assert progress.current_level == 0
        assert progress.episodes_at_level == 0
        assert progress.wins_at_level == 0
        assert progress.recent_episodes == []
        assert progress.graduated_levels == []
        assert progress.total_episodes == 0
        assert progress.total_wins == 0
        assert progress.start_time > 0

    def test_progress_tracking(self):
        """Test modifying progress values."""
        progress = TrainingProgress()
        progress.current_level = 2
        progress.episodes_at_level = 50
        progress.wins_at_level = 35
        progress.recent_episodes = [True, True, False, True]
        progress.graduated_levels = ["Level 1", "Level 2"]

        assert progress.current_level == 2
        assert progress.wins_at_level == 35
        assert len(progress.recent_episodes) == 4
        assert len(progress.graduated_levels) == 2

    def test_progress_start_time_set(self):
        """Test that start_time is set automatically."""
        before = time.time()
        progress = TrainingProgress()
        after = time.time()

        assert before <= progress.start_time <= after


class TestVmapEnvAdapter:
    """Tests for VmapEnvAdapter class."""

    def test_adapter_initialization(self):
        """Test adapter initialization with mock vmap env."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 4
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)

        assert adapter.num_envs == 4
        assert adapter.vmap_env == mock_vmap
        assert adapter._supports_set_opponent is False

    def test_adapter_reset(self):
        """Test reset passes through to vmap env."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 2
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)
        mock_vmap.reset.return_value = (np.zeros((2, 9)), {})

        adapter = VmapEnvAdapter(mock_vmap)
        obs = adapter.reset()

        mock_vmap.reset.assert_called_once()
        assert obs.shape == (2, 9)

    def test_adapter_step_async_wait(self):
        """Test step_async and step_wait."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 2
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        # Mock step return
        obs = np.zeros((2, 9))
        rewards = np.array([1.0, -1.0])
        dones = np.array([False, True])
        truncated = np.array([False, False])
        infos = [{}, {}]
        mock_vmap.step.return_value = (obs, rewards, dones, truncated, infos)

        adapter = VmapEnvAdapter(mock_vmap)

        # Test step_async
        actions = np.array([[0.5, 1], [0.3, 2]])
        adapter.step_async(actions)
        assert np.array_equal(adapter.actions, actions)

        # Test step_wait
        result_obs, result_rewards, result_dones, result_infos = adapter.step_wait()

        mock_vmap.step.assert_called_once_with(actions)
        assert np.array_equal(result_obs, obs)
        assert np.array_equal(result_rewards, rewards)
        # Dones should be combined with truncated
        assert result_dones[1] == True  # True OR False = True

    def test_adapter_close(self):
        """Test close method exists and doesn't error."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.close()  # Should not raise

    def test_adapter_env_is_wrapped(self):
        """Test env_is_wrapped returns False."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)
        assert adapter.env_is_wrapped(Mock) is False

    def test_adapter_get_attr(self):
        """Test get_attr returns list of attributes."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 3
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)
        mock_vmap.some_attr = "test_value"

        adapter = VmapEnvAdapter(mock_vmap)
        result = adapter.get_attr("some_attr")

        assert len(result) == 3
        assert all(v == "test_value" for v in result)

    def test_adapter_set_attr(self):
        """Test set_attr doesn't raise."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.set_attr("test", "value")  # Should not raise

    def test_adapter_env_method_set_opponent_warning(self):
        """Test env_method with set_opponent shows warning once."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 2
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)

        # First call should trigger warning
        with patch('builtins.print') as mock_print:
            adapter.env_method("set_opponent", Mock())
            assert mock_print.called

        # Second call should not re-warn
        with patch('builtins.print') as mock_print:
            adapter.env_method("set_opponent", Mock())
            # Warning flag should prevent second print
            assert hasattr(adapter, '_warned_about_opponent_switch')

    def test_adapter_env_method_other(self):
        """Test env_method with other methods returns list of None."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 3
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        adapter = VmapEnvAdapter(mock_vmap)
        result = adapter.env_method("some_method")

        assert result == [None, None, None]


class TestCurriculumCallback:
    """Tests for CurriculumCallback class."""

    def test_callback_initialization(self):
        """Test callback initializes with trainer."""
        mock_trainer = Mock()
        callback = CurriculumCallback(mock_trainer)

        assert callback.curriculum_trainer == mock_trainer
        assert callback.episode_rewards == []
        assert callback.episode_wins == []
        assert callback.last_rollout_time is None
        assert callback.last_train_time is None

    def test_callback_on_rollout_start(self):
        """Test _on_rollout_start sets timestamp."""
        mock_trainer = Mock()
        callback = CurriculumCallback(mock_trainer)

        callback._on_rollout_start()

        assert callback.last_rollout_time is not None
        assert callback.rollout_count == 1

    def test_callback_on_rollout_start_increments_counter(self):
        """Test rollout counter increments."""
        mock_trainer = Mock()
        callback = CurriculumCallback(mock_trainer)

        callback._on_rollout_start()
        callback._on_rollout_start()
        callback._on_rollout_start()

        assert callback.rollout_count == 3

    def test_callback_with_verbose(self):
        """Test callback with verbose mode."""
        mock_trainer = Mock()
        callback = CurriculumCallback(mock_trainer, verbose=1)

        assert callback.verbose == 1


class TestCurriculumLevelEdgeCases:
    """Edge case tests for curriculum structures."""

    def test_level_with_empty_opponents(self):
        """Test level can have empty opponents list."""
        level = CurriculumLevel(
            name="Empty",
            difficulty=DifficultyLevel.FUNDAMENTALS,
            opponents=[]
        )
        assert len(level.opponents) == 0

    def test_level_with_many_opponents(self):
        """Test level with many opponents."""
        opponents = [f"fighter_{i}.py" for i in range(100)]
        level = CurriculumLevel(
            name="Many",
            difficulty=DifficultyLevel.POPULATION,
            opponents=opponents
        )
        assert len(level.opponents) == 100

    def test_progress_with_long_history(self):
        """Test progress with long episode history."""
        progress = TrainingProgress()
        progress.recent_episodes = [True, False] * 500  # 1000 entries
        assert len(progress.recent_episodes) == 1000

    def test_difficulty_comparison(self):
        """Test difficulty level values are strings."""
        for level in DifficultyLevel:
            assert isinstance(level.value, str)


class TestVmapEnvAdapterDonesCombination:
    """Tests for dones/truncated combination logic."""

    def test_dones_false_truncated_false(self):
        """Test combining dones=False with truncated=False."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        mock_vmap.step.return_value = (
            np.zeros((1, 9)),
            np.array([0.0]),
            np.array([False]),  # dones
            np.array([False]),  # truncated
            [{}]
        )

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.step_async(np.array([[0.0, 1]]))
        _, _, dones, _ = adapter.step_wait()

        assert dones[0] == False

    def test_dones_false_truncated_true(self):
        """Test combining dones=False with truncated=True."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        mock_vmap.step.return_value = (
            np.zeros((1, 9)),
            np.array([0.0]),
            np.array([False]),  # dones
            np.array([True]),   # truncated
            [{}]
        )

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.step_async(np.array([[0.0, 1]]))
        _, _, dones, _ = adapter.step_wait()

        assert dones[0] == True  # False OR True = True

    def test_dones_true_truncated_false(self):
        """Test combining dones=True with truncated=False."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        mock_vmap.step.return_value = (
            np.zeros((1, 9)),
            np.array([0.0]),
            np.array([True]),   # dones
            np.array([False]),  # truncated
            [{}]
        )

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.step_async(np.array([[0.0, 1]]))
        _, _, dones, _ = adapter.step_wait()

        assert dones[0] == True  # True OR False = True

    def test_dones_true_truncated_true(self):
        """Test combining dones=True with truncated=True."""
        mock_vmap = Mock()
        mock_vmap.n_envs = 1
        mock_vmap.observation_space = Mock()
        mock_vmap.observation_space.shape = (9,)
        mock_vmap.action_space = Mock()
        mock_vmap.action_space.shape = (2,)

        mock_vmap.step.return_value = (
            np.zeros((1, 9)),
            np.array([0.0]),
            np.array([True]),  # dones
            np.array([True]),  # truncated
            [{}]
        )

        adapter = VmapEnvAdapter(mock_vmap)
        adapter.step_async(np.array([[0.0, 1]]))
        _, _, dones, _ = adapter.step_wait()

        assert dones[0] == True  # True OR True = True
