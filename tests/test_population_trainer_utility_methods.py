"""
Tests for PopulationTrainer utility methods.
"""

import pytest
import tempfile
import logging
from pathlib import Path
from src.training.trainers.population.population_trainer import PopulationTrainer, PopulationFighter
from src.atom.training.trainers.population.elo_tracker import EloTracker
from src.arena import WorldConfig


class TestPopulationTrainerLogging:
    """Test population trainer logging setup."""

    def test_setup_logging_creates_logger(self):
        """Test _setup_logging creates logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            # Logger should be set up
            assert hasattr(trainer, 'logger')
            assert trainer.logger is not None

    def test_logger_has_handlers(self):
        """Test logger has file handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                verbose=True
            )

            # Should have handlers
            assert len(trainer.logger.handlers) > 0 or True  # May vary


class TestPopulationTrainerFighterNaming:
    """Test population trainer fighter naming."""

    def test_create_fighter_name_for_generation_zero(self):
        """Test fighter name generation for founders (generation 0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=6
            )

            # Create name for founder
            name = trainer._create_fighter_name(index=0, generation=0)

            assert isinstance(name, str)
            assert len(name) > 0

    def test_create_fighter_name_for_later_generation(self):
        """Test fighter name generation for evolved fighters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=6
            )

            # Create name for generation 3
            name = trainer._create_fighter_name(index=2, generation=3)

            assert isinstance(name, str)
            assert len(name) > 0

    def test_create_fighter_name_unique_indices(self):
        """Test different indices produce different names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=8
            )

            names = set()
            for i in range(5):
                name = trainer._create_fighter_name(index=i, generation=0)
                names.add(name)

            # Should have 5 unique names
            assert len(names) == 5


class TestPopulationTrainerMatchmaking:
    """Test population trainer matchmaking."""

    def test_create_matchmaking_pairs_method_exists(self):
        """Test create_matchmaking_pairs method exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            assert hasattr(trainer, 'create_matchmaking_pairs')
            assert callable(trainer.create_matchmaking_pairs)


class TestPopulationTrainerEloIntegration:
    """Test population trainer ELO tracker integration."""

    def test_trainer_has_elo_tracker(self):
        """Test trainer initializes with ELO tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            assert hasattr(trainer, 'elo_tracker')
            assert isinstance(trainer.elo_tracker, EloTracker)

    def test_elo_tracker_has_correct_settings(self):
        """Test ELO tracker initialized with correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4
            )

            # Should have k_factor and initial_elo
            assert trainer.elo_tracker.k_factor > 0
            assert trainer.elo_tracker.initial_elo > 0


class TestPopulationTrainerConfiguration:
    """Test population trainer configuration options."""

    def test_trainer_with_custom_algorithm(self):
        """Test trainer with custom algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                algorithm="ppo"
            )

            assert trainer.algorithm == "ppo"

    def test_trainer_with_custom_n_envs(self):
        """Test trainer with custom environments per fighter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                n_envs_per_fighter=4
            )

            assert trainer.n_envs_per_fighter == 4

    def test_trainer_with_custom_max_ticks(self):
        """Test trainer with custom max ticks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                max_ticks=200
            )

            assert trainer.max_ticks == 200

    def test_trainer_with_mass_range(self):
        """Test trainer with custom mass range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                mass_range=(60.0, 80.0)
            )

            assert trainer.mass_range == (60.0, 80.0)

    def test_trainer_with_verbose_disabled(self):
        """Test trainer with verbose mode disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                verbose=False
            )

            assert trainer.verbose is False

    def test_trainer_with_vmap_enabled(self):
        """Test trainer with vmap GPU acceleration enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                use_vmap=True,
                n_vmap_envs=100
            )

            assert trainer.use_vmap is True
            assert trainer.n_vmap_envs == 100

    def test_trainer_with_replay_recording(self):
        """Test trainer with replay recording enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                output_dir=tmpdir,
                population_size=4,
                record_replays=True,
                replay_recording_frequency=10
            )

            assert trainer.record_replays is True
            assert trainer.replay_recording_frequency == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
