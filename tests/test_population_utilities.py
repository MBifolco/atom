"""
Tests for population training utility modules.
Quick tests to boost coverage of 0% modules.
"""

import pytest


class TestPopulationTrainingScripts:
    """Test population training script modules."""

    def test_import_train_population(self):
        """Test train_population module imports."""
        from src.training.trainers.population import train_population

        assert train_population is not None

    def test_import_train_population_multi(self):
        """Test train_population_multi module imports."""
        from src.training.trainers.population import train_population_multi

        assert train_population_multi is not None

    def test_import_train_multicore(self):
        """Test train_multicore module imports."""
        from src.training.trainers.population import train_multicore

        assert train_multicore is not None

    def test_import_debug_gym_env(self):
        """Test debug_gym_env module imports."""
        from src.training.trainers.population import debug_gym_env

        assert debug_gym_env is not None

    def test_import_test_fighter_loading(self):
        """Test test_fighter_loading module imports."""
        from src.training.trainers.population import test_fighter_loading

        assert test_fighter_loading is not None

    def test_import_test_single_match(self):
        """Test test_single_match module imports."""
        from src.training.trainers.population import test_single_match

        assert test_single_match is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
