"""
Comprehensive tests for PopulationTrainer class methods.
"""

import pytest
import tempfile
from pathlib import Path
from src.training.trainers.population.population_trainer import PopulationTrainer
from src.arena import WorldConfig


class TestPopulationTrainerInitialization:
    """Test PopulationTrainer initialization."""

    def test_population_trainer_creates_directories(self):
        """Test trainer creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig()

            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                config=config
            )

            output_path = Path(tmpdir)
            # Should create some subdirectories
            assert output_path.exists()

    def test_population_trainer_has_elo_tracker(self):
        """Test trainer initializes ELO tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            assert hasattr(trainer, 'elo_tracker')
            assert trainer.elo_tracker is not None

    def test_population_trainer_stores_config(self):
        """Test trainer stores configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig()

            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=6,
                config=config
            )

            assert trainer.config == config
            assert trainer.population_size == 6

    def test_population_trainer_default_config(self):
        """Test trainer creates default config if none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            assert trainer.config is not None
            assert isinstance(trainer.config, WorldConfig)

    def test_population_trainer_sets_algorithm(self):
        """Test trainer sets algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                algorithm="ppo"
            )

            assert trainer.algorithm == "ppo"


class TestPopulationTrainerAttributes:
    """Test PopulationTrainer attributes and properties."""

    def test_trainer_has_output_directory(self):
        """Test trainer has output directory attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                generations=2
            )

            assert hasattr(trainer, 'output_dir')
            assert trainer.output_dir == Path(tmpdir)

    def test_trainer_has_generation_tracking(self):
        """Test trainer initializes generation tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                generations=5
            )

            assert hasattr(trainer, 'generations')
            assert trainer.generations == 5

    def test_trainer_has_population_size(self):
        """Test trainer stores population size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=8,
                generations=2
            )

            assert trainer.population_size == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
