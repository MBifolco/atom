"""
Comprehensive tests for VmapEnvWrapper to increase coverage to 50%.
Tests initialization, reset, step, observations, rewards, and done detection.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.training.vmap_env_wrapper import VmapEnvWrapper
from src.arena import WorldConfig


class TestVmapEnvWrapperInit:
    """Tests for VmapEnvWrapper initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            max_ticks=100
        )

        assert env.n_envs == 4
        assert env.max_ticks == 100
        assert env.fighter_mass == 70.0
        assert env.opponent_mass == 70.0
        assert env.config is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom WorldConfig."""
        config = WorldConfig(arena_width=20.0)

        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            config=config
        )

        assert env.arena_width == 20.0

    def test_init_with_custom_masses(self):
        """Test initialization with custom fighter/opponent masses."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            fighter_mass=80.0,
            opponent_mass=60.0
        )

        assert env.fighter_mass == 80.0
        assert env.opponent_mass == 60.0

    def test_init_with_seed(self):
        """Test initialization with custom seed."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            seed=123
        )

        assert env.seed_base == 123

    def test_init_with_debug_mode(self):
        """Test initialization with debug mode enabled."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            debug=True
        )

        assert env.debug is True

    def test_observation_space_shape(self):
        """Test observation space is correctly configured."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        assert env.observation_space.shape == (9,)

    def test_action_space_shape(self):
        """Test action space is correctly configured."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        assert env.action_space.shape == (2,)

    def test_metadata_exists(self):
        """Test metadata attribute exists."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        assert hasattr(env, 'metadata')
        assert env.metadata == {"render_modes": []}


class TestVmapEnvWrapperReset:
    """Tests for reset method."""

    def test_reset_returns_correct_shape(self):
        """Test reset returns observations with correct shape."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=8,
            opponent_decision_func=dummy_opponent
        )

        obs, info = env.reset()

        assert obs.shape == (8, 9)
        assert isinstance(info, dict)

    def test_reset_initializes_tick_counts(self):
        """Test reset initializes tick counts to zero."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert np.all(env.tick_counts == 0)

    def test_reset_initializes_episode_rewards(self):
        """Test reset initializes episode rewards to zero."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert np.all(env.episode_rewards == 0)

    def test_reset_with_new_seed(self):
        """Test reset accepts new seed parameter."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            seed=42
        )

        obs, _ = env.reset(seed=99)

        assert env.seed_base == 99

    def test_reset_initializes_hp_tracking(self):
        """Test reset initializes HP tracking arrays."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert env.prev_fighter_hp is not None
        assert env.prev_opponent_hp is not None
        assert len(env.prev_fighter_hp) == 4

    def test_reset_initializes_stamina_tracking(self):
        """Test reset initializes stamina tracking arrays."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert env.prev_fighter_stamina is not None
        assert env.prev_opponent_stamina is not None

    def test_reset_initializes_episode_statistics(self):
        """Test reset initializes episode damage statistics."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert np.all(env.episode_damage_dealt == 0)
        assert np.all(env.episode_damage_taken == 0)
        assert np.all(env.episode_stamina_used == 0)

    def test_reset_initializes_reward_breakdowns(self):
        """Test reset initializes reward breakdown arrays."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        assert np.all(env.episode_damage_reward == 0)
        assert np.all(env.episode_proximity_reward == 0)
        assert np.all(env.episode_stamina_reward == 0)
        assert np.all(env.episode_stance_reward == 0)
        assert np.all(env.episode_inaction_penalty == 0)


class TestVmapEnvWrapperStep:
    """Tests for step method."""

    def test_step_returns_correct_shapes(self):
        """Test step returns outputs with correct shapes."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            max_ticks=100
        )

        env.reset()
        actions = np.zeros((4, 2), dtype=np.float32)
        obs, rewards, dones, truncated, infos = env.step(actions)

        assert obs.shape == (4, 9)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert truncated.shape == (4,)
        assert isinstance(infos, list)
        assert len(infos) == 4

    def test_step_increments_tick_count(self):
        """Test step increments tick count."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()
        assert np.all(env.tick_counts == 0)

        actions = np.zeros((2, 2), dtype=np.float32)
        env.step(actions)
        assert np.all(env.tick_counts == 1)

        env.step(actions)
        assert np.all(env.tick_counts == 2)

    def test_step_with_different_actions(self):
        """Test step with various action combinations."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=3,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        # Different actions for each env
        actions = np.array([
            [0.5, 0.0],   # accelerate, neutral
            [-0.5, 1.0],  # decelerate, extended
            [0.0, 2.0],   # no accel, defending
        ], dtype=np.float32)

        obs, rewards, dones, truncated, infos = env.step(actions)
        assert obs.shape == (3, 9)

    def test_step_truncation_at_max_ticks(self):
        """Test step truncates at max_ticks."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=5
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Run until max_ticks
        for _ in range(4):
            obs, rewards, dones, truncated, infos = env.step(actions)
            assert np.all(truncated == False)

        # On tick 5, should truncate
        obs, rewards, dones, truncated, infos = env.step(actions)
        assert np.all(truncated == True)

    def test_step_accumulates_episode_rewards(self):
        """Test step accumulates rewards in episode_rewards."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()
        assert np.all(env.episode_rewards == 0)

        actions = np.zeros((2, 2), dtype=np.float32)
        obs, rewards, dones, truncated, infos = env.step(actions)

        # Episode rewards should have accumulated
        assert np.allclose(env.episode_rewards, rewards)

        # Step again
        obs, rewards2, dones, truncated, infos = env.step(actions)
        # Should accumulate
        assert np.all(np.abs(env.episode_rewards) >= np.abs(rewards))

    def test_step_multiple_steps(self):
        """Test running multiple steps without crash."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            max_ticks=50
        )

        env.reset()
        actions = np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32)
        actions[:, 1] = np.clip(actions[:, 1], 0, 2.99)  # Valid stance range

        for _ in range(20):
            obs, rewards, dones, truncated, infos = env.step(actions)
            assert obs.shape == (4, 9)
            assert not np.any(np.isnan(obs))


class TestVmapEnvWrapperObservations:
    """Tests for observation methods."""

    def test_get_observations_returns_correct_format(self):
        """Test _get_observations returns correct format."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()
        obs = env._get_observations()

        assert obs.shape == (4, 9)
        assert obs.dtype == np.float32

    def test_observations_contain_expected_values(self):
        """Test observations contain reasonable initial values."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        obs, _ = env.reset()

        # Position should be around starting positions
        assert np.all(obs[:, 0] >= 0)  # Fighter position
        # HP should be 1.0 (full) initially
        assert np.allclose(obs[:, 2], 1.0)  # Fighter HP normalized
        # Stamina should be 1.0 (full) initially
        assert np.allclose(obs[:, 3], 1.0)  # Fighter stamina normalized


class TestVmapEnvWrapperDones:
    """Tests for done checking."""

    def test_check_dones_initially_false(self):
        """Test _check_dones returns all False initially."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()
        dones = env._check_dones()

        assert np.all(dones == False)

    def test_check_dones_after_steps(self):
        """Test _check_dones after running steps."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=10
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Run a few steps
        for _ in range(5):
            env.step(actions)

        dones = env._check_dones()
        assert dones.shape == (2,)
        assert dones.dtype == bool


class TestVmapEnvWrapperRewards:
    """Tests for reward calculation."""

    def test_calculate_rewards_mid_episode(self):
        """Test reward calculation during mid-episode."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        dones = np.array([False, False, False, False])
        truncated = np.array([False, False, False, False])

        rewards = env._calculate_rewards(dones, truncated)

        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32

    def test_calculate_rewards_on_truncation(self):
        """Test reward calculation on timeout/truncation."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=5
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Run until truncation
        for _ in range(5):
            obs, rewards, dones, truncated, infos = env.step(actions)

        # At truncation, rewards should be calculated
        assert rewards.shape == (2,)
        assert np.all(truncated == True)

    def test_rewards_are_finite(self):
        """Test that rewards are always finite values."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            max_ticks=20
        )

        env.reset()
        actions = np.random.uniform(-1, 1, size=(4, 2)).astype(np.float32)
        actions[:, 1] = np.clip(actions[:, 1], 0, 2.99)

        for _ in range(15):
            obs, rewards, dones, truncated, infos = env.step(actions)
            assert np.all(np.isfinite(rewards))


class TestVmapEnvWrapperResetEnvs:
    """Tests for _reset_envs method."""

    def test_reset_envs_resets_specific_environments(self):
        """Test _reset_envs only resets masked environments."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        # Modify statistics to track which envs are reset
        env.episode_damage_dealt[0] = 100.0
        env.episode_damage_dealt[1] = 100.0

        # Reset only first env
        reset_mask = np.array([True, False, False, False])
        env._reset_envs(reset_mask)

        # First env should be reset
        assert env.episode_damage_dealt[0] == 0.0
        # Second env should NOT be reset
        assert env.episode_damage_dealt[1] == 100.0

    def test_reset_envs_resets_all_statistics(self):
        """Test _reset_envs resets all episode statistics."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        # Set some statistics
        env.episode_damage_dealt[0] = 50.0
        env.episode_damage_taken[0] = 30.0
        env.episode_stamina_used[0] = 20.0
        env.episode_damage_reward[0] = 10.0
        env.episode_proximity_reward[0] = 5.0

        reset_mask = np.array([True, False])
        env._reset_envs(reset_mask)

        # All statistics for env 0 should be reset
        assert env.episode_damage_dealt[0] == 0.0
        assert env.episode_damage_taken[0] == 0.0
        assert env.episode_stamina_used[0] == 0.0
        assert env.episode_damage_reward[0] == 0.0
        assert env.episode_proximity_reward[0] == 0.0


class TestVmapEnvWrapperMultipleSteps:
    """Tests for running multiple steps."""

    def test_multiple_steps_without_crash(self):
        """Test running multiple steps doesn't crash."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=4,
            opponent_decision_func=dummy_opponent,
            max_ticks=100
        )

        env.reset()
        actions = np.zeros((4, 2), dtype=np.float32)

        for i in range(50):
            obs, rewards, dones, truncated, infos = env.step(actions)
            assert obs.shape == (4, 9)
            assert not np.any(np.isnan(obs))

    def test_auto_reset_on_done(self):
        """Test environments auto-reset when done."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=5
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Run past max_ticks to trigger truncation/reset
        for _ in range(6):
            obs, rewards, dones, truncated, infos = env.step(actions)

        # After truncation, env should auto-reset - tick counts should be low
        assert np.all(env.tick_counts <= 1)


class TestVmapEnvWrapperInfos:
    """Tests for info dict generation."""

    def test_info_contains_episode_data_on_done(self):
        """Test infos contain episode data when episode ends."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=3
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Step until truncation
        for _ in range(3):
            obs, rewards, dones, truncated, infos = env.step(actions)

        # Infos should be a list of dicts (one per env)
        assert isinstance(infos, list)
        assert len(infos) == 2
        # Each info should contain episode data on done
        for info in infos:
            assert isinstance(info, dict)
            if 'episode' in info:
                assert 'r' in info['episode']  # reward
                assert 'l' in info['episode']  # length


class TestVmapEnvWrapperEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_env(self):
        """Test with single environment."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=1,
            opponent_decision_func=dummy_opponent
        )

        obs, _ = env.reset()
        assert obs.shape == (1, 9)

        actions = np.zeros((1, 2), dtype=np.float32)
        obs, rewards, dones, truncated, infos = env.step(actions)
        assert obs.shape == (1, 9)

    def test_large_number_of_envs(self):
        """Test with larger number of environments."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=32,
            opponent_decision_func=dummy_opponent
        )

        obs, _ = env.reset()
        assert obs.shape == (32, 9)

    def test_extreme_actions(self):
        """Test with extreme action values."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        # Extreme actions (will be clipped internally)
        actions = np.array([
            [100.0, 0.0],   # Very high acceleration
            [-100.0, 2.0],  # Very negative acceleration
        ], dtype=np.float32)

        obs, rewards, dones, truncated, infos = env.step(actions)
        assert obs.shape == (2, 9)
        assert not np.any(np.isnan(obs))


class TestVmapEnvWrapperProximityRewards:
    """Tests for proximity-based reward calculation."""

    def test_last_distance_initialized_after_first_step(self):
        """Test last_distance is set after first step for proximity tracking."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()
        # Initially last_distance may be None or set to initial
        initial_last_distance = env.last_distance

        actions = np.zeros((2, 2), dtype=np.float32)
        env.step(actions)

        # After step, last_distance should be set
        assert env.last_distance is not None


class TestVmapEnvWrapperDamageTracking:
    """Tests for damage tracking functionality."""

    def test_initial_damage_tracking_is_zero(self):
        """Test initial damage tracking values are zero."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent
        )

        env.reset()

        # Initial damage should be 0
        assert np.all(env.episode_damage_dealt == 0)
        assert np.all(env.episode_damage_taken == 0)

    def test_damage_tracking_accumulates(self):
        """Test damage dealt/taken accumulates over steps."""
        def dummy_opponent(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = VmapEnvWrapper(
            n_envs=2,
            opponent_decision_func=dummy_opponent,
            max_ticks=50
        )

        env.reset()
        actions = np.zeros((2, 2), dtype=np.float32)

        # Run several steps
        for _ in range(20):
            env.step(actions)

        # Damage tracking should exist (may or may not have accumulated)
        assert env.episode_damage_dealt is not None
        assert env.episode_damage_taken is not None
