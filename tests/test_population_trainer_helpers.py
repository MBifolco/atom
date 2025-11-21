"""
Comprehensive tests for PopulationTrainer helper functions.
Tests the extracted helper methods to increase testability.
"""

import pytest
import os
import tempfile
from pathlib import Path
from src.training.trainers.population.population_trainer import (
    _configure_process_threading,
    _reconstruct_config,
    _create_opponent_decide_func,
    _load_opponent_models_for_training,
    PopulationTrainer
)
from src.arena import WorldConfig
import numpy as np


class TestConfigureProcessThreadingComplete:
    """Complete testing of process threading configuration."""

    def test_configures_omp_threads(self):
        """Test OMP_NUM_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('OMP_NUM_THREADS') == '1'

    def test_configures_mkl_threads(self):
        """Test MKL_NUM_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('MKL_NUM_THREADS') == '1'

    def test_configures_openblas_threads(self):
        """Test OPENBLAS_NUM_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('OPENBLAS_NUM_THREADS') == '1'

    def test_configures_veclib_threads(self):
        """Test VECLIB_MAXIMUM_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('VECLIB_MAXIMUM_THREADS') == '1'

    def test_configures_numexpr_threads(self):
        """Test NUMEXPR_NUM_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('NUMEXPR_NUM_THREADS') == '1'

    def test_configures_tf_intraop_threads(self):
        """Test TF_NUM_INTRAOP_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('TF_NUM_INTRAOP_THREADS') == '1'

    def test_configures_tf_interop_threads(self):
        """Test TF_NUM_INTEROP_THREADS is set."""
        _configure_process_threading()
        assert os.environ.get('TF_NUM_INTEROP_THREADS') == '1'

    def test_all_variables_set_to_one(self):
        """Test all thread variables are set to '1'."""
        _configure_process_threading()

        thread_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS'
        ]

        for var in thread_vars:
            assert os.environ.get(var) == '1', f"{var} should be '1'"


class TestReconstructConfigComplete:
    """Complete testing of config reconstruction."""

    def test_reconstruct_with_none_creates_default(self):
        """Test None input creates default WorldConfig."""
        config = _reconstruct_config(None)

        assert isinstance(config, WorldConfig)
        assert config.arena_width > 0
        assert config.dt > 0
        assert config.friction > 0

    def test_reconstruct_with_empty_dict_creates_default(self):
        """Test empty dict creates default WorldConfig."""
        config = _reconstruct_config({})

        assert isinstance(config, WorldConfig)
        assert len(config.stances) == 3

    def test_reconstruct_applies_arena_width(self):
        """Test reconstruction applies custom arena width."""
        config = _reconstruct_config({"arena_width": 15.0})

        assert config.arena_width == 15.0

    def test_reconstruct_applies_friction(self):
        """Test reconstruction applies custom friction."""
        config = _reconstruct_config({"friction": 0.3})

        assert config.friction == 0.3

    def test_reconstruct_applies_max_acceleration(self):
        """Test reconstruction applies max acceleration."""
        config = _reconstruct_config({"max_acceleration": 5.5})

        assert config.max_acceleration == 5.5

    def test_reconstruct_applies_multiple_parameters(self):
        """Test reconstruction applies multiple parameters."""
        config_dict = {
            "arena_width": 14.0,
            "friction": 0.25,
            "max_acceleration": 5.0,
            "dt": 0.1
        }

        config = _reconstruct_config(config_dict)

        assert config.arena_width == 14.0
        assert config.friction == 0.25
        assert config.max_acceleration == 5.0
        assert config.dt == 0.1

    def test_reconstructed_config_has_all_required_fields(self):
        """Test reconstructed config has all standard fields."""
        config = _reconstruct_config({"arena_width": 12.0})

        # Should have all standard fields even if not specified
        assert hasattr(config, 'friction')
        assert hasattr(config, 'max_velocity')
        assert hasattr(config, 'stances')
        assert hasattr(config, 'hit_cooldown_ticks')


class TestCreateOpponentDecideFuncComplete:
    """Complete testing of opponent decide function wrapper."""

    def test_wrapper_creates_callable(self):
        """Test wrapper returns callable function."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                return np.array([0.5, 1.0]), None

        decide_func = _create_opponent_decide_func(MockModel())

        assert callable(decide_func)

    def test_wrapper_function_accepts_snapshot(self):
        """Test wrapped function accepts snapshot dict."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                return np.array([0.3, 0.5]), None

        decide_func = _create_opponent_decide_func(MockModel())

        snapshot = {
            "you": {
                "position": 5.0,
                "velocity": 0.5,
                "hp": 85.0,
                "max_hp": 100.0,
                "stamina": 8.5,
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 4.0,
                "direction": 1.0,
                "velocity": -0.3,
                "hp": 92.0,
                "max_hp": 100.0,
                "stamina": 9.2,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.5
            }
        }

        action = decide_func(snapshot)

        assert isinstance(action, dict)

    def test_wrapper_returns_valid_acceleration(self):
        """Test wrapped function returns valid acceleration."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                return np.array([0.8, 1.5]), None

        decide_func = _create_opponent_decide_func(MockModel())

        snapshot = {
            "you": {"position": 5.0, "velocity": 0.0, "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0},
            "opponent": {"distance": 3.0, "direction": 1.0, "velocity": 0.0, "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0},
            "arena": {"width": 12.0}
        }

        action = decide_func(snapshot)

        assert "acceleration" in action
        assert isinstance(action["acceleration"], (int, float))
        # Scaled by 4.5
        assert -5.0 <= action["acceleration"] <= 5.0

    def test_wrapper_returns_valid_stance(self):
        """Test wrapped function returns valid stance."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                return np.array([0.0, 2.5]), None  # Stance 2 = defending

        decide_func = _create_opponent_decide_func(MockModel())

        snapshot = {
            "you": {"position": 6.0, "velocity": 0.0, "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0},
            "opponent": {"distance": 2.0, "direction": -1.0, "velocity": 0.0, "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0},
            "arena": {"width": 12.0}
        }

        action = decide_func(snapshot)

        assert "stance" in action
        assert action["stance"] in ["neutral", "extended", "defending"]

    def test_wrapper_converts_snapshot_to_observation(self):
        """Test wrapper correctly converts snapshot to observation array."""
        class MockModel:
            def __init__(self):
                self.last_obs = None

            def predict(self, obs, deterministic=False):
                self.last_obs = obs
                return np.array([0.0, 0.0]), None

        model = MockModel()
        decide_func = _create_opponent_decide_func(model)

        snapshot = {
            "you": {"position": 4.0, "velocity": 1.0, "hp": 60.0, "max_hp": 100.0, "stamina": 5.0, "max_stamina": 10.0},
            "opponent": {"distance": 7.0, "direction": 1.0, "velocity": -0.5, "hp": 80.0, "max_hp": 100.0, "stamina": 8.0, "max_stamina": 10.0},
            "arena": {"width": 12.5}
        }

        decide_func(snapshot)

        # Check observation was created correctly
        assert model.last_obs is not None
        assert model.last_obs.shape == (9,)
        assert model.last_obs.dtype == np.float32

        # Check values
        assert model.last_obs[0] == 4.0  # position
        assert model.last_obs[1] == 1.0  # velocity
        assert model.last_obs[2] == 0.6  # hp normalized (60/100)
        assert model.last_obs[3] == 0.5  # stamina normalized (5/10)
        assert model.last_obs[4] == 7.0  # distance
        assert model.last_obs[8] == 12.5  # arena width


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
