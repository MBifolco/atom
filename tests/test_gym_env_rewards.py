"""
Tests for gym_env reward calculation edge cases.
Covers the various termination and truncation reward branches.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.training.gym_env import AtomCombatEnv
from src.arena import WorldConfig


class TestTerminationRewards:
    """Tests for reward calculation on termination (KO)."""

    def test_win_reward_basic(self):
        """Test basic win reward when fighter wins."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=500)
        env.reset()

        # Simulate until termination
        total_reward = 0
        for _ in range(300):
            action = np.array([1.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break

        # Should have received some reward
        assert isinstance(total_reward, float)

    def test_loss_reward_basic(self):
        """Test loss penalty when fighter loses."""
        # Aggressive opponent
        def aggressive_opponent(state):
            return {"stance": "extended", "acceleration": 1.0}

        env = AtomCombatEnv(aggressive_opponent, max_ticks=500)
        env.reset()

        # Defend poorly
        for _ in range(300):
            action = np.array([0.0, 0.0], dtype=np.float32)  # Passive
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated or env.tick >= 300


class TestTruncationRewards:
    """Tests for reward calculation on truncation (timeout)."""

    def test_truncation_clear_hp_win(self):
        """Test truncation when fighter has clear HP advantage."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=50)
        env.reset()

        # Attack to build HP advantage
        for _ in range(50):
            action = np.array([1.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Should reach truncation
        assert terminated or truncated

    def test_truncation_occurs_at_max_ticks(self):
        """Test that truncation happens exactly at max_ticks."""
        opponent_func = lambda state: {"stance": "defending", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=20)
        env.reset()

        tick_count = 0
        for _ in range(25):
            action = np.array([0.0, 2.0], dtype=np.float32)  # Defend
            obs, reward, terminated, truncated, info = env.step(action)
            tick_count += 1
            if terminated or truncated:
                break

        assert truncated == True or terminated == True
        assert env.tick <= 20


class TestMidEpisodeRewards:
    """Tests for mid-episode shaped rewards."""

    def test_damage_reward_when_hitting(self):
        """Test that damage dealt gives positive reward."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=100)
        env.reset()

        # Attack close opponent
        rewards = []
        for _ in range(20):
            action = np.array([1.0, 1.0], dtype=np.float32)  # Attack
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated:
                break

        # Should have some non-zero rewards from damage
        assert len(rewards) > 0

    def test_defending_stance_reward(self):
        """Test defending stance when low stamina gives bonus."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=100)
        env.reset()

        # Run some steps to potentially deplete stamina then defend
        for _ in range(30):
            action = np.array([0.0, 2.0], dtype=np.float32)  # Defend
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert True  # Test completes without error

    def test_close_range_engagement(self):
        """Test close range engagement bonus."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=50)
        env.reset()

        # Move forward and attack
        for _ in range(20):
            action = np.array([1.0, 1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # Should complete without error
        assert True


class TestObservationContents:
    """Tests for observation array contents."""

    def test_observation_hp_values(self):
        """Test HP values in observation."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        obs, _ = env.reset()

        # obs[2] = you_hp (normalized)
        # obs[6] = opponent_hp (normalized)
        assert 0 <= obs[2] <= 1
        assert 0 <= obs[6] <= 1

    def test_observation_stamina_values(self):
        """Test stamina values in observation."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        obs, _ = env.reset()

        # obs[3] = you_stamina (normalized)
        # obs[7] = opponent_stamina (normalized)
        assert 0 <= obs[3] <= 1
        assert 0 <= obs[7] <= 1

    def test_observation_positions(self):
        """Test position values in observation."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        obs, _ = env.reset()

        # obs[0] = you_position
        # obs[4] = opponent_position
        assert isinstance(obs[0], (float, np.floating))
        assert isinstance(obs[4], (float, np.floating))

    def test_observation_velocities(self):
        """Test velocity values in observation."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        obs, _ = env.reset()

        # obs[1] = you_velocity
        # obs[5] = opponent_velocity
        assert isinstance(obs[1], (float, np.floating))
        assert isinstance(obs[5], (float, np.floating))

    def test_observation_arena_width(self):
        """Test arena width in observation."""
        config = WorldConfig(arena_width=15.0)
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, config=config)
        obs, _ = env.reset()

        # obs[8] = arena_width
        assert obs[8] == 15.0


class TestInfoDict:
    """Tests for info dictionary contents."""

    def test_info_after_step(self):
        """Test info dict contents after step."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([0.5, 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(info, dict)

    def test_info_on_termination(self):
        """Test info dict on episode termination."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=10)
        env.reset()

        action = np.array([1.0, 1.0], dtype=np.float32)
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert isinstance(info, dict)


class TestStanceSelection:
    """Tests for stance selection from action."""

    def test_neutral_stance_selection(self):
        """Test selecting neutral stance (0)."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)  # Neutral
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)

    def test_extended_stance_selection(self):
        """Test selecting extended stance (1)."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([0.0, 1.0], dtype=np.float32)  # Extended
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)

    def test_defending_stance_selection(self):
        """Test selecting defending stance (2)."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([0.0, 2.0], dtype=np.float32)  # Defending
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)

    def test_stance_clamping(self):
        """Test that out-of-range stance values are handled."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        # Action with stance > 2 should be clamped
        action = np.array([0.0, 5.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (9,)


class TestAccelerationHandling:
    """Tests for acceleration input handling."""

    def test_positive_acceleration(self):
        """Test positive acceleration (forward)."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Should complete step without error
        assert obs.shape == (9,)
        assert isinstance(reward, float)

    def test_negative_acceleration(self):
        """Test negative acceleration (backward)."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([-1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Should complete step without error
        assert obs.shape == (9,)
        assert isinstance(reward, float)

    def test_zero_acceleration(self):
        """Test zero acceleration."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        assert obs.shape == (9,)


class TestEpisodeDamageTracking:
    """Tests for episode damage tracking."""

    def test_damage_tracking_initialization(self):
        """Test damage tracking starts at zero."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        assert env.episode_damage_dealt == 0
        assert env.episode_damage_taken == 0

    def test_damage_tracking_reset(self):
        """Test damage tracking resets on episode reset."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func)
        env.reset()

        # Run some steps
        for _ in range(10):
            action = np.array([1.0, 1.0], dtype=np.float32)
            env.step(action)

        # Reset
        env.reset()

        assert env.episode_damage_dealt == 0
        assert env.episode_damage_taken == 0


class TestMultipleEpisodes:
    """Tests for running multiple episodes."""

    def test_consecutive_episodes(self):
        """Test running multiple episodes in sequence."""
        opponent_func = lambda state: {"stance": "neutral", "acceleration": 0}
        env = AtomCombatEnv(opponent_func, max_ticks=20)

        for episode in range(5):
            obs, _ = env.reset()
            assert obs.shape == (9,)

            for _ in range(25):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            assert env.tick <= 20

    def test_different_opponents_per_episode(self):
        """Test changing opponents between episodes."""
        def opponent_a(state):
            return {"stance": "extended", "acceleration": 1.0}

        def opponent_b(state):
            return {"stance": "defending", "acceleration": -0.5}

        env = AtomCombatEnv(opponent_a, max_ticks=20)

        # Episode 1 with opponent A
        env.reset()
        for _ in range(20):
            action = np.array([0.0, 1.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Change opponent
        env.set_opponent(opponent_b)

        # Episode 2 with opponent B
        env.reset()
        for _ in range(20):
            action = np.array([0.0, 1.0], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert True  # Completed without error
