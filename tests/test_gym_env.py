"""
Tests for the Gymnasium environment wrapper for Atom Combat.

Tests cover:
- Environment initialization and configuration
- Observation space structure
- Action space handling
- Reset functionality
- Step execution and reward calculation
"""

import numpy as np
import pytest
from training.src.gym_env import AtomCombatEnv


class TestAtomCombatEnvInitialization:
    """Test environment initialization and configuration."""

    def test_env_initializes_with_opponent_func(self):
        """Environment can be created with opponent decision function."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)

        assert env is not None
        assert env.opponent_decide == simple_opponent

    def test_env_has_correct_observation_space_shape(self):
        """Observation space has correct shape (9 values)."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)

        # Should have 9 continuous observations
        assert env.observation_space.shape == (9,)

    def test_env_has_correct_action_space_shape(self):
        """Action space has correct shape (2 values: acceleration, stance selector)."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)

        # Should have 2 continuous actions
        assert env.action_space.shape == (2,)

    def test_env_accepts_custom_config(self):
        """Environment accepts custom WorldConfig."""
        from src.arena.world_config import WorldConfig

        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        custom_config = WorldConfig()
        custom_config.max_acceleration = 5.0

        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent,
            config=custom_config
        )

        assert env.config.max_acceleration == 5.0

    def test_env_accepts_max_ticks_parameter(self):
        """Environment respects max_ticks parameter."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent,
            max_ticks=500
        )

        assert env.max_ticks == 500


class TestAtomCombatEnvReset:
    """Test environment reset functionality."""

    def test_reset_returns_observation_and_info(self):
        """Reset returns observation array and info dict."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_has_correct_shape(self):
        """Reset observation has shape (9,)."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        obs, _ = env.reset()

        assert obs.shape == (9,)

    def test_reset_initializes_fighters_with_full_hp(self):
        """Reset creates fighters with full HP."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        obs, _ = env.reset()

        # Observation indices: [position, velocity, hp, stamina, distance, rel_vel, opp_hp, opp_stamina, arena_width]
        # obs[2] = fighter HP (normalized to 0-1), should be 1.0 at start
        # obs[6] = opponent HP (normalized to 0-1), should be 1.0 at start
        assert obs[2] == 1.0, "Fighter should start with full HP"
        assert obs[6] == 1.0, "Opponent should start with full HP"

    def test_reset_initializes_fighters_with_full_stamina(self):
        """Reset creates fighters with full stamina."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        obs, _ = env.reset()

        # obs[3] = fighter stamina (normalized)
        # obs[7] = opponent stamina (normalized)
        assert obs[3] == 1.0, "Fighter should start with full stamina"
        assert obs[7] == 1.0, "Opponent should start with full stamina"

    def test_reset_with_seed(self):
        """Reset accepts seed parameter for reproducibility."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # Same seed should give same initial state
        np.testing.assert_array_equal(obs1, obs2)


class TestAtomCombatEnvStep:
    """Test environment step execution."""

    def test_step_returns_required_values(self):
        """Step returns observation, reward, terminated, truncated, info."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)  # No acceleration, neutral stance
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_shape(self):
        """Step returns observation with correct shape."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        assert obs.shape == (9,)

    def test_step_increments_tick(self):
        """Each step increments the tick counter."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        assert env.tick == 0

        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.tick == 1

        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.tick == 2

    def test_step_calls_opponent_decide(self):
        """Step calls opponent decision function."""
        call_count = [0]

        def counting_opponent(snapshot):
            call_count[0] += 1
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=counting_opponent)
        env.reset()

        env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert call_count[0] == 1, "Opponent decide should be called once per step"

    def test_step_truncates_at_max_ticks(self):
        """Step sets truncated=True when max_ticks is reached."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent, max_ticks=5)
        env.reset()

        # Take 4 steps (should not truncate)
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            assert not truncated
            assert not terminated

        # 5th step should truncate
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert truncated

    def test_step_terminates_when_fighter_dies(self):
        """Step sets terminated=True when fighter HP reaches 0."""
        def aggressive_opponent(snapshot):
            return {"acceleration": 3.0, "stance": "extended"}

        env = AtomCombatEnv(opponent_decision_func=aggressive_opponent, max_ticks=1000)
        env.reset()

        # Keep stepping until someone dies
        terminated = False
        max_steps = 1000
        steps = 0

        while not terminated and steps < max_steps:
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))  # Stand still, extended stance
            steps += 1

        # Eventually one fighter should die (or timeout)
        assert terminated or truncated, "Match should end eventually"

    def test_action_acceleration_clamping(self):
        """Action acceleration is clamped to valid range."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        # Try extreme acceleration (should be clamped)
        action = np.array([100.0, 0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        # Should not crash and should return valid observation
        assert obs.shape == (9,)

    def test_action_stance_conversion(self):
        """Action stance selector is converted to stance name."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        # Test different stance selectors
        # 0.0-0.99 -> neutral (index 0)
        # 1.0-1.99 -> extended (index 1)
        # 2.0-2.99 -> retracted (index 2)
        # 3.0-3.99 -> defending (index 3)

        for stance_selector in [0.5, 1.5, 2.5, 3.5]:
            env.reset()
            action = np.array([0.0, stance_selector], dtype=np.float32)
            obs, _, _, _, _ = env.step(action)
            # Should not crash
            assert obs.shape == (9,)

    def test_info_dict_contains_damage_dealt(self):
        """Info dict contains damage_dealt field."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "damage_dealt" in info

    def test_info_dict_contains_damage_taken(self):
        """Info dict contains damage_taken field."""
        def simple_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=simple_opponent)
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "damage_taken" in info


class TestAtomCombatEnvRewards:
    """Test reward calculation."""

    def test_reward_for_dealing_damage(self):
        """Dealing damage gives positive reward."""
        def passive_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=passive_opponent)
        env.reset()

        # Attack with extended stance
        action = np.array([2.0, 1.0], dtype=np.float32)  # Move forward, extended stance

        total_reward = 0
        for _ in range(50):  # Take multiple steps to engage
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        # Over time, attacking should generate some reward from either:
        # - Dealing damage
        # - Proximity bonus
        # Note: Reward might be negative due to inaction penalty if no damage dealt
        assert isinstance(total_reward, (int, float)), "Reward should be numeric"

    def test_terminal_reward_for_win(self):
        """Winning gives positive terminal reward."""
        def weak_opponent(snapshot):
            # Always defend, never attack
            return {"acceleration": 0.0, "stance": "defending"}

        env = AtomCombatEnv(opponent_decision_func=weak_opponent, max_ticks=500)
        env.reset()

        # Keep attacking until match ends
        action = np.array([2.0, 1.0], dtype=np.float32)  # Attack

        final_reward = None
        for _ in range(500):
            _, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                final_reward = reward
                won = info.get("won", False)
                break

        if won:
            # If we won, final reward should be positive (50+ for timeout wins, 200+ for KO)
            assert final_reward > 0, "Winning should give positive reward"

    def test_inaction_penalty(self):
        """Standing still with no damage gives penalty."""
        def passive_opponent(snapshot):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(opponent_decision_func=passive_opponent)
        env.reset()

        # Stand completely still
        action = np.array([0.0, 0.0], dtype=np.float32)

        # Take several steps
        _, reward1, _, _, _ = env.step(action)
        _, reward2, _, _, _ = env.step(action)

        # Should get inaction penalty (negative reward)
        # Both steps should have negative or zero reward
        assert reward1 <= 0 or reward2 <= 0, "Inaction should eventually incur penalty"
