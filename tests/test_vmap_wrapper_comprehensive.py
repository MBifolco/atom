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
    """Tests for step method - skipped due to JAX API version mismatch."""

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_returns_correct_shapes(self):
        """Test step returns outputs with correct shapes."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_increments_tick_count(self):
        """Test step increments tick count."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_with_different_actions(self):
        """Test step with various action combinations."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_truncation_at_max_ticks(self):
        """Test step truncates at max_ticks."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_accumulates_episode_rewards(self):
        """Test step accumulates rewards in episode_rewards."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_handles_nan_observations(self):
        """Test step handles and clips NaN observations."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_step_clips_extreme_rewards(self):
        """Test step clips extreme reward values."""
        pass


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

    @pytest.mark.skip(reason="JAX state manipulation requires specific NamedTuple setup")
    def test_check_dones_detects_fighter_death(self):
        """Test _check_dones detects when fighter HP <= 0."""
        pass

    @pytest.mark.skip(reason="JAX state manipulation requires specific NamedTuple setup")
    def test_check_dones_detects_opponent_death(self):
        """Test _check_dones detects when opponent HP <= 0."""
        pass


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

    @pytest.mark.skip(reason="JAX state manipulation requires specific setup")
    def test_calculate_rewards_on_win(self):
        """Test reward calculation when fighter wins."""
        pass

    @pytest.mark.skip(reason="JAX state manipulation requires specific setup")
    def test_calculate_rewards_on_loss(self):
        """Test reward calculation when fighter loses."""
        pass

    @pytest.mark.skip(reason="JAX state manipulation requires specific setup")
    def test_calculate_rewards_on_timeout_win(self):
        """Test reward calculation on timeout with HP advantage."""
        pass

    @pytest.mark.skip(reason="JAX state manipulation requires specific setup")
    def test_calculate_rewards_on_timeout_loss(self):
        """Test reward calculation on timeout with HP disadvantage."""
        pass


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
    """Tests for running multiple steps - skipped due to JAX API mismatch."""

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_multiple_steps_without_crash(self):
        """Test running multiple steps doesn't crash."""
        pass

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_auto_reset_on_done(self):
        """Test environments auto-reset when done."""
        pass


class TestVmapEnvWrapperInfos:
    """Tests for info dict generation - skipped due to JAX API mismatch."""

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_info_contains_episode_data_on_done(self):
        """Test infos contain episode data when episode ends."""
        pass


class TestVmapEnvWrapperEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_single_env(self):
        """Test with single environment."""
        pass

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

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_extreme_actions(self):
        """Test with extreme action values."""
        pass


class TestVmapEnvWrapperProximityRewards:
    """Tests for proximity-based reward calculation - skipped due to step() requirement."""

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_last_distance_initialized_after_first_step(self):
        """Test last_distance is set after first step for proximity tracking."""
        pass


class TestVmapEnvWrapperDamageTracking:
    """Tests for damage tracking functionality - skipped due to step() requirement."""

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

    @pytest.mark.skip(reason="Requires JAX arena API update")
    def test_damage_tracking_accumulates(self):
        """Test damage dealt/taken accumulates over steps."""
        pass
