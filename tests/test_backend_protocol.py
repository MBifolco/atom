"""Tests for TrainingBackend protocol compliance."""

import numpy as np
import pytest

from src.atom.training.backends import SB3PPOBackend, TrainingBackend, BackendCapabilities


class TestSB3PPOBackendProtocol:
    """Verify SB3PPOBackend satisfies the TrainingBackend protocol."""

    def test_isinstance_check(self):
        """SB3PPOBackend should satisfy runtime_checkable TrainingBackend."""
        backend = SB3PPOBackend(device="cpu")
        assert isinstance(backend, TrainingBackend)

    def test_capabilities(self):
        backend = SB3PPOBackend(device="cpu")
        caps = backend.capabilities
        assert isinstance(caps, BackendCapabilities)
        assert caps.name == "sb3_ppo"
        assert caps.framework == "pytorch"
        assert caps.on_policy is True
        assert caps.flush_mode == "rollout_boundary"

    def test_get_policy_arch(self):
        backend = SB3PPOBackend(device="cpu")
        arch = backend.get_policy_arch()
        assert "net_arch" in arch
        assert arch["net_arch"] == [256, 256]
        assert "activation_fn" in arch

    def test_predict_returns_numpy(self):
        """predict() should return a numpy array."""
        backend = SB3PPOBackend(device="cpu")

        # Create a minimal model to test predict
        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            max_ticks=10,
        ))])
        model = backend.create_model(env, seed=42, mode="curriculum")
        obs = env.reset()
        action = backend.predict(model, obs)

        assert isinstance(action, np.ndarray)
        assert action.shape[-1] == 2  # 2D action space
        env.close()

    def test_predict_deterministic_flag(self):
        """predict() should respect deterministic flag."""
        backend = SB3PPOBackend(device="cpu")

        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            max_ticks=10,
        ))])
        model = backend.create_model(env, seed=42, mode="curriculum")
        obs = env.reset()

        # Deterministic should give same result each time
        a1 = backend.predict(model, obs, deterministic=True)
        a2 = backend.predict(model, obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)
        env.close()

    def test_create_model_curriculum_vs_population(self):
        """Curriculum and population modes should create models with same arch."""
        backend = SB3PPOBackend(device="cpu")

        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            max_ticks=10,
        ))])

        model_c = backend.create_model(env, seed=42, mode="curriculum")
        model_p = backend.create_model(env, seed=42, mode="population")

        # Same architecture
        assert str(model_c.policy.net_arch) == str(model_p.policy.net_arch)

        # Different learning rates
        assert model_c.learning_rate != model_p.learning_rate
        env.close()

    def test_handle_distribution_shift_noop(self):
        """PPO backend should silently handle distribution shifts."""
        backend = SB3PPOBackend(device="cpu")
        # Should not raise
        backend.handle_distribution_shift(None, "level_transition")
        backend.handle_distribution_shift(None, "pool_refresh")

    def test_device_owned_at_construction(self):
        """Device should be set at construction, not at create_model time."""
        backend_cpu = SB3PPOBackend(device="cpu")
        assert backend_cpu._device == "cpu"


class TestBackendProtocolCompleteness:
    """Verify all protocol methods exist on the backend."""

    def test_all_protocol_methods_present(self):
        backend = SB3PPOBackend(device="cpu")
        required = [
            "capabilities",
            "create_model",
            "load_model",
            "save_model",
            "predict",
            "learn",
            "replace_env",
            "get_policy_arch",
            "reduce_learning_rate",
            "clone_and_mutate",
            "create_dummy_env",
            "handle_distribution_shift",
        ]
        for method_name in required:
            assert hasattr(backend, method_name), f"Missing protocol method: {method_name}"


class TestSBXSACBackendProtocol:
    """Verify SBXSACBackend satisfies the TrainingBackend protocol."""

    def test_isinstance_check(self):
        from src.atom.training.backends import SBXSACBackend
        backend = SBXSACBackend()
        assert isinstance(backend, TrainingBackend)

    def test_capabilities(self):
        from src.atom.training.backends import SBXSACBackend
        backend = SBXSACBackend()
        caps = backend.capabilities
        assert caps.name == "sbx_sac"
        assert caps.framework == "jax"
        assert caps.on_policy is False
        assert caps.flush_mode == "step_interval"

    def test_all_protocol_methods_present(self):
        from src.atom.training.backends import SBXSACBackend
        backend = SBXSACBackend()
        required = [
            "capabilities", "create_model", "load_model", "save_model",
            "predict", "learn", "replace_env", "get_policy_arch",
            "reduce_learning_rate", "clone_and_mutate", "create_dummy_env",
            "handle_distribution_shift",
        ]
        for method_name in required:
            assert hasattr(backend, method_name), f"Missing: {method_name}"

    def test_create_predict_save_load(self):
        """End-to-end: create model, predict, save, load."""
        from src.atom.training.backends import SBXSACBackend
        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        import tempfile, os

        backend = SBXSACBackend()
        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            max_ticks=10,
        ))])

        model = backend.create_model(env, seed=42, mode="curriculum")
        obs = env.reset()

        # Predict
        action = backend.predict(model, obs)
        assert isinstance(action, np.ndarray)
        assert action.shape[-1] == 2

        # Save and load
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_sac")
            backend.save_model(model, path)
            loaded = backend.load_model(path, envs=env)
            action2 = backend.predict(loaded, obs, deterministic=True)
            assert isinstance(action2, np.ndarray)

        env.close()

    def test_handle_distribution_shift_preserves_buffer(self):
        """level_transition should preserve replay buffer for off-policy phases."""
        from src.atom.training.backends import SBXSACBackend
        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        backend = SBXSACBackend()
        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
            max_ticks=10,
        ))])

        model = backend.create_model(env, seed=42, mode="curriculum")
        model.learn(total_timesteps=200)
        assert model.replay_buffer.pos > 0

        old_pos = model.replay_buffer.pos
        backend.handle_distribution_shift(model, "level_transition")
        assert model.replay_buffer.pos == old_pos  # Buffer preserved

        # pool_refresh should NOT clear
        model.learn(total_timesteps=100)
        pos_before = model.replay_buffer.pos
        backend.handle_distribution_shift(model, "pool_refresh")
        assert model.replay_buffer.pos == pos_before

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
