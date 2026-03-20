"""
Comprehensive tests for curriculum trainer.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    VmapEnvAdapter,
    CurriculumCallback
)
from stable_baselines3.common.callbacks import BaseCallback


class TestCurriculumCallback:
    """Test curriculum callback."""

    def test_callback_is_base_callback(self):
        """Test curriculum callback extends BaseCallback."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=0)

        assert isinstance(callback, BaseCallback)

    def test_callback_initialization(self):
        """Test callback initializes with curriculum trainer."""
        class MockTrainer:
            algorithm = "ppo"

        callback = CurriculumCallback(curriculum_trainer=MockTrainer(), verbose=1)

        assert callback.curriculum_trainer is not None
        assert callback.verbose == 1


class TestVmapEnvAdapterMethods:
    """Test VmapEnvAdapter methods."""

    def test_adapter_has_required_methods(self):
        """Test adapter implements required VecEnv methods."""
        # Just test the class structure without creating instance
        assert hasattr(VmapEnvAdapter, 'reset')
        assert hasattr(VmapEnvAdapter, 'step_async')
        assert hasattr(VmapEnvAdapter, 'step_wait')
        assert hasattr(VmapEnvAdapter, 'close')
        assert hasattr(VmapEnvAdapter, 'env_is_wrapped')
        assert hasattr(VmapEnvAdapter, 'get_attr')
        assert hasattr(VmapEnvAdapter, 'set_attr')
        assert hasattr(VmapEnvAdapter, 'env_method')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
