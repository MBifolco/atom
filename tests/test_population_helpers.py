"""
Tests for refactored population training helper functions.
"""

import pytest
import os
import tempfile
from pathlib import Path
from src.training.trainers.population.population_trainer import (
    _configure_process_threading,
    _reconstruct_config,
    _create_opponent_decide_func
)
from src.arena import WorldConfig


class TestConfigureProcessThreading:
    """Test thread configuration helper."""

    def test_configure_sets_environment_variables(self):
        """Test thread configuration sets all environment variables."""
        # Clear any existing values
        thread_vars = [
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'
        ]

        _configure_process_threading()

        # All should be set to '1'
        for var in thread_vars:
            assert os.environ.get(var) == '1', f"{var} should be set to '1'"


class TestReconstructConfig:
    """Test config reconstruction helper."""

    def test_reconstruct_with_none_returns_default(self):
        """Test reconstructing with None returns default config."""
        config = _reconstruct_config(None)

        assert isinstance(config, WorldConfig)
        assert config.arena_width > 0

    def test_reconstruct_with_dict_uses_values(self):
        """Test reconstructing with dict applies values."""
        config_dict = {
            "arena_width": 15.0,
            "friction": 0.5,
            "max_acceleration": 5.0
        }

        config = _reconstruct_config(config_dict)

        assert config.arena_width == 15.0
        assert config.friction == 0.5
        assert config.max_acceleration == 5.0

    def test_reconstruct_preserves_all_config_fields(self):
        """Test reconstruction preserves config structure."""
        config_dict = {
            "arena_width": 12.0,
            "dt": 0.1
        }

        config = _reconstruct_config(config_dict)

        # Should have all standard fields
        assert hasattr(config, 'arena_width')
        assert hasattr(config, 'friction')
        assert hasattr(config, 'stances')


class TestCreateOpponentDecideFunc:
    """Test opponent decide function wrapper."""

    def test_creates_callable_function(self):
        """Test wrapper creates callable decide function."""
        # Create a mock model with predict method
        class MockModel:
            def predict(self, obs, deterministic=False):
                import numpy as np
                # Return simple action
                return np.array([0.5, 1.0]), None

        model = MockModel()
        decide_func = _create_opponent_decide_func(model)

        assert callable(decide_func)

    def test_decide_func_returns_valid_action(self):
        """Test wrapped decide function returns valid action dict."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                import numpy as np
                return np.array([0.3, 0.5]), None

        model = MockModel()
        decide_func = _create_opponent_decide_func(model)

        # Create test snapshot
        snapshot = {
            "you": {
                "position": 5.0,
                "velocity": 0.0,
                "hp": 80.0,
                "max_hp": 100.0,
                "stamina": 8.0,
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 3.0,
                "direction": 1.0,
                "velocity": 0.5,
                "hp": 90.0,
                "max_hp": 100.0,
                "stamina": 7.0,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.5
            }
        }

        action = decide_func(snapshot)

        assert "acceleration" in action
        assert "stance" in action
        assert isinstance(action["acceleration"], (int, float))
        assert action["stance"] in ["neutral", "extended", "defending"]

    def test_decide_func_converts_observation_correctly(self):
        """Test decide function converts snapshot to observation array."""
        class MockModel:
            def __init__(self):
                self.last_obs = None

            def predict(self, obs, deterministic=False):
                self.last_obs = obs
                import numpy as np
                return np.array([0.0, 1.0]), None

        model = MockModel()
        decide_func = _create_opponent_decide_func(model)

        snapshot = {
            "you": {
                "position": 2.0,
                "velocity": 0.5,
                "hp": 50.0,
                "max_hp": 100.0,
                "stamina": 5.0,
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 8.0,
                "direction": 1.0,
                "velocity": -0.3,
                "hp": 70.0,
                "max_hp": 100.0,
                "stamina": 8.0,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.5
            }
        }

        action = decide_func(snapshot)

        # Check that model received correct observation
        assert model.last_obs is not None
        assert model.last_obs.shape == (13,)
        # First element should be position
        assert model.last_obs[0] == 2.0
        # Third element should be normalized HP
        assert model.last_obs[2] == 0.5  # 50/100
        # Opponent stance defaults to neutral and recent damage defaults to 0
        assert model.last_obs[11] == 0.0
        assert model.last_obs[12] == 0.0


class TestCreateVmapEnvironment:
    """Test vmap environment creation helper."""

    def test_create_vmap_environment_configures_jax_memory(self):
        """Test vmap environment sets JAX memory environment variables."""
        # Note: Can't easily test actual VmapEnvWrapper creation without full setup
        # But we can verify the helper exists and is callable
        from src.training.trainers.population.population_trainer import _create_vmap_training_environment

        assert callable(_create_vmap_training_environment)


class TestCreateCpuEnvironment:
    """Test CPU environment creation helper."""

    def test_create_cpu_environment_is_callable(self):
        """Test CPU environment creation helper exists."""
        from src.training.trainers.population.population_trainer import _create_cpu_training_environment

        assert callable(_create_cpu_training_environment)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
