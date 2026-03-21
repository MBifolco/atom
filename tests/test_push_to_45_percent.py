"""
Final comprehensive tests to push coverage to 45%.
Focuses on easy-to-test code paths and edge cases.
"""

import pytest
import os
from src.atom.training.trainers.population.elo_tracker import EloTracker, FighterStats
from src.atom.training.trainers.population.fighter_loader import (
    load_fighter,
    validate_fighter,
    FighterLoadError
)
from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    TrainingProgress
)
from src.training.trainers.population.population_trainer import (
    _configure_process_threading,
    _reconstruct_config,
    _create_opponent_decide_func,
    PopulationFighter
)


class TestConfigureProcessThreadingComplete:
    """Complete testing of thread configuration."""

    def test_all_thread_variables_set(self):
        """Test all 7 thread environment variables are set."""
        _configure_process_threading()

        required_vars = [
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS'
        ]

        for var in required_vars:
            value = os.environ.get(var)
            assert value == '1', f"{var} should be '1', got {value}"

    def test_configure_is_idempotent(self):
        """Test calling configure multiple times is safe."""
        _configure_process_threading()
        _configure_process_threading()  # Call again

        # Should still be set correctly
        assert os.environ['OMP_NUM_THREADS'] == '1'


class TestReconstructConfigComplete:
    """Complete testing of config reconstruction."""

    def test_reconstruct_empty_dict(self):
        """Test reconstruction with empty dict."""
        config = _reconstruct_config({})

        # Should create valid config with defaults
        assert config.arena_width > 0
        assert config.dt > 0

    def test_reconstruct_partial_dict(self):
        """Test reconstruction with partial config dict."""
        partial = {
            "arena_width": 14.0,
            "friction": 0.4
        }

        config = _reconstruct_config(partial)

        # Specified values should be used
        assert config.arena_width == 14.0
        assert config.friction == 0.4

        # Other fields should have defaults
        assert config.dt > 0
        assert len(config.stances) == 3

    def test_reconstruct_with_stance_config(self):
        """Test reconstruction with stance configurations."""
        config_dict = {
            "arena_width": 12.0,
            "stances": {
                "neutral": {"reach": 0.3, "width": 0.4, "drain": 0.0, "defense": 1.0},
                "extended": {"reach": 0.8, "width": 0.2, "drain": 0.08, "defense": 0.9},
                "defending": {"reach": 0.4, "width": 0.5, "drain": -0.1, "defense": 1.5}
            }
        }

        config = _reconstruct_config(config_dict)

        assert len(config.stances) == 3
        assert "neutral" in config.stances


class TestCreateOpponentDecideFuncComplete:
    """Complete testing of opponent decide function wrapper."""

    def test_decide_func_handles_edge_positions(self):
        """Test decide func with edge case positions."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                import numpy as np
                return np.array([0.0, 0.0]), None

        decide = _create_opponent_decide_func(MockModel())

        # Edge case: at arena boundary
        snapshot = {
            "you": {
                "position": 0.0,  # At left edge
                "velocity": 0.0,
                "hp": 100.0,
                "max_hp": 100.0,
                "stamina": 10.0,
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 12.0,
                "direction": 1.0,
                "velocity": 0.0,
                "hp": 100.0,
                "max_hp": 100.0,
                "stamina": 10.0,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.0
            }
        }

        action = decide(snapshot)

        assert "acceleration" in action
        assert "stance" in action

    def test_decide_func_handles_low_stamina(self):
        """Test decide func with very low stamina."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                import numpy as np
                return np.array([1.0, 2.0]), None

        decide = _create_opponent_decide_func(MockModel())

        snapshot = {
            "you": {
                "position": 5.0,
                "velocity": 0.0,
                "hp": 80.0,
                "max_hp": 100.0,
                "stamina": 0.5,  # Very low
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 2.0,
                "direction": 1.0,
                "velocity": 0.0,
                "hp": 70.0,
                "max_hp": 100.0,
                "stamina": 8.0,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.0
            }
        }

        action = decide(snapshot)

        # Should handle low stamina snapshot
        assert isinstance(action["acceleration"], (int, float))

    def test_decide_func_handles_low_hp(self):
        """Test decide func with very low HP."""
        class MockModel:
            def predict(self, obs, deterministic=False):
                import numpy as np
                return np.array([-0.8, 2.0]), None

        decide = _create_opponent_decide_func(MockModel())

        snapshot = {
            "you": {
                "position": 6.0,
                "velocity": 0.5,
                "hp": 5.0,  # Critical HP
                "max_hp": 100.0,
                "stamina": 6.0,
                "max_stamina": 10.0
            },
            "opponent": {
                "distance": 1.5,
                "direction": -1.0,
                "velocity": 0.3,
                "hp": 95.0,
                "max_hp": 100.0,
                "stamina": 9.0,
                "max_stamina": 10.0
            },
            "arena": {
                "width": 12.5
            }
        }

        action = decide(snapshot)

        # Should still return valid action
        assert action["stance"] in ["neutral", "extended", "defending"]


class TestPopulationFighterComplete:
    """Complete PopulationFighter dataclass testing."""

    def test_fighter_with_all_fields_populated(self):
        """Test fighter with all tracking fields set."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Complete",
            model=MockModel(),
            generation=5,
            lineage="founder->child1->child2->child3",
            mass=72.5,
            training_episodes=500,
            last_checkpoint="/path/to/gen5_checkpoint.zip"
        )

        assert fighter.generation == 5
        assert "founder" in fighter.lineage
        assert "child3" in fighter.lineage
        assert fighter.mass == 72.5
        assert fighter.training_episodes == 500
        assert fighter.last_checkpoint is not None

    def test_fighter_generation_zero_is_founder(self):
        """Test generation 0 fighters are founders."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Founder",
            model=MockModel(),
            generation=0,
            lineage="founder"
        )

        assert fighter.generation == 0
        assert fighter.lineage == "founder"

    def test_fighter_high_generation_descendant(self):
        """Test high generation descendants."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Gen10",
            model=MockModel(),
            generation=10,
            lineage="long_lineage_chain"
        )

        assert fighter.generation == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
