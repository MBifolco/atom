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


class TestComputeTrainingSummary:
    """Tests for _compute_training_summary helper."""

    def test_empty_results_returns_zeros(self):
        """Test empty results returns zero counts."""
        from src.training.trainers.population.population_trainer import _compute_training_summary

        summary = _compute_training_summary([])
        assert summary["successful"] == 0
        assert summary["total"] == 0

    def test_all_successful_results(self):
        """Test summary with all successful results."""
        from src.training.trainers.population.population_trainer import _compute_training_summary

        results = [
            {"mean_reward": 10.0, "episodes": 100},
            {"mean_reward": 20.0, "episodes": 200},
            {"mean_reward": 30.0, "episodes": 150}
        ]

        summary = _compute_training_summary(results)

        assert summary["successful"] == 3
        assert summary["total"] == 3
        assert summary["success_rate"] == 1.0
        assert summary["average_reward"] == 20.0
        assert summary["best_reward"] == 30.0
        assert summary["worst_reward"] == 10.0
        assert summary["total_episodes"] == 450

    def test_mixed_results(self):
        """Test summary with some failed results."""
        from src.training.trainers.population.population_trainer import _compute_training_summary

        results = [
            {"mean_reward": 15.0, "episodes": 100},
            {"error": "training failed"},  # No mean_reward
            {"mean_reward": 25.0, "episodes": 200}
        ]

        summary = _compute_training_summary(results)

        assert summary["successful"] == 2
        assert summary["total"] == 3
        assert summary["success_rate"] == pytest.approx(2/3)
        assert summary["average_reward"] == 20.0

    def test_results_without_episodes(self):
        """Test results that have mean_reward but missing episodes."""
        from src.training.trainers.population.population_trainer import _compute_training_summary

        results = [
            {"mean_reward": 10.0},  # No episodes key
            {"mean_reward": 20.0, "episodes": 100}
        ]

        summary = _compute_training_summary(results)

        assert summary["successful"] == 2
        assert summary["total_episodes"] == 100  # Only counts present episodes


class TestComputeTrainingProgress:
    """Tests for _compute_training_progress helper."""

    def test_progress_at_start(self):
        """Test progress at start of training."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(0, 10000, 0.1)

        assert progress["progress_pct"] == 0
        # At start, timesteps_per_sec is 0, so eta_sec is 0 (can't estimate)
        assert progress["eta_sec"] == 0

    def test_progress_midway(self):
        """Test progress at 50%."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(5000, 10000, 10.0)

        assert progress["progress_pct"] == 50
        assert progress["timesteps_per_sec"] == 500.0
        assert progress["eta_sec"] == pytest.approx(10.0)  # 5000 remaining / 500 per sec

    def test_progress_complete(self):
        """Test progress at completion."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(10000, 10000, 20.0)

        assert progress["progress_pct"] == 100
        assert progress["eta_sec"] == pytest.approx(0.0)

    def test_progress_caps_at_100(self):
        """Test progress doesn't exceed 100%."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(15000, 10000, 10.0)

        assert progress["progress_pct"] == 100

    def test_progress_with_zero_timesteps(self):
        """Test handling of zero total timesteps."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(0, 0, 1.0)

        assert progress["progress_pct"] == 0

    def test_progress_with_zero_elapsed(self):
        """Test handling of zero elapsed time."""
        from src.training.trainers.population.population_trainer import _compute_training_progress

        progress = _compute_training_progress(1000, 10000, 0.0)

        assert progress["timesteps_per_sec"] == 0
        assert progress["eta_sec"] == 0


class TestCalculateWinRate:
    """Tests for _calculate_win_rate helper."""

    def test_all_wins(self):
        """Test 100% win rate."""
        from src.training.trainers.population.population_trainer import _calculate_win_rate

        rate = _calculate_win_rate(10, 0, 0)
        assert rate == 1.0

    def test_all_losses(self):
        """Test 0% win rate."""
        from src.training.trainers.population.population_trainer import _calculate_win_rate

        rate = _calculate_win_rate(0, 10, 0)
        assert rate == 0.0

    def test_all_draws(self):
        """Test draws count as half wins."""
        from src.training.trainers.population.population_trainer import _calculate_win_rate

        rate = _calculate_win_rate(0, 0, 10)
        assert rate == 0.5

    def test_mixed_results(self):
        """Test mixed win/loss/draw."""
        from src.training.trainers.population.population_trainer import _calculate_win_rate

        # 5 wins + 0.5*2 draws = 6 / 10 = 0.6
        rate = _calculate_win_rate(5, 3, 2)
        assert rate == pytest.approx(0.6)

    def test_zero_matches(self):
        """Test zero matches returns 0."""
        from src.training.trainers.population.population_trainer import _calculate_win_rate

        rate = _calculate_win_rate(0, 0, 0)
        assert rate == 0.0


class TestSelectParentWeighted:
    """Tests for _select_parent_weighted helper."""

    def test_returns_fighter_from_list(self):
        """Test that returned parent is from the survivor list."""
        from src.training.trainers.population.population_trainer import _select_parent_weighted
        from unittest.mock import Mock

        survivors = [Mock(name="fighter_1"), Mock(name="fighter_2"), Mock(name="fighter_3")]
        elo_tracker = Mock()

        parent = _select_parent_weighted(survivors, elo_tracker)

        assert parent in survivors

    def test_single_survivor(self):
        """Test with single survivor."""
        from src.training.trainers.population.population_trainer import _select_parent_weighted
        from unittest.mock import Mock

        survivor = Mock(name="only_fighter")
        elo_tracker = Mock()

        parent = _select_parent_weighted([survivor], elo_tracker)

        assert parent == survivor


class TestApplyWeightMutation:
    """Tests for _apply_weight_mutation helper."""

    def test_mutation_changes_weights(self):
        """Test that mutation modifies model weights."""
        from src.training.trainers.population.population_trainer import _apply_weight_mutation
        import torch
        from unittest.mock import Mock, MagicMock

        # Create mock model with real tensor parameters
        mock_model = Mock()
        mock_policy = Mock()

        # Create real parameter tensors
        param1 = torch.nn.Parameter(torch.ones(10, 10))
        param2 = torch.nn.Parameter(torch.zeros(5))

        mock_policy.parameters.return_value = [param1, param2]
        mock_model.policy = mock_policy

        # Store original values
        original_param1 = param1.clone()
        original_param2 = param2.clone()

        # Apply mutation
        _apply_weight_mutation(mock_model, mutation_rate=0.5)

        # Verify weights changed
        assert not torch.equal(param1, original_param1)
        assert not torch.equal(param2, original_param2)

    def test_mutation_rate_zero_minimal_change(self):
        """Test that zero mutation rate produces minimal change."""
        from src.training.trainers.population.population_trainer import _apply_weight_mutation
        import torch
        from unittest.mock import Mock

        mock_model = Mock()
        mock_policy = Mock()

        param = torch.nn.Parameter(torch.ones(10))
        mock_policy.parameters.return_value = [param]
        mock_model.policy = mock_policy

        original = param.clone()

        # Zero mutation rate - should be very small changes
        _apply_weight_mutation(mock_model, mutation_rate=0.0)

        # With zero mutation, noise scale is 0, so no change expected
        # But due to numerical precision, use close comparison
        assert torch.allclose(param, original, atol=1e-6)


class TestFormatTrainingBanner:
    """Tests for _format_training_banner helper."""

    def test_banner_contains_required_fields(self):
        """Test banner contains all required fields."""
        from src.training.trainers.population.population_trainer import _format_training_banner

        banner = _format_training_banner(
            population_size=10,
            generations=5,
            episodes_per_generation=1000,
            evolution_frequency=2
        )

        assert "STARTING POPULATION TRAINING" in banner
        assert "Population Size: 10" in banner
        assert "Generations: 5" in banner
        assert "Episodes per Generation: 1000" in banner
        assert "Evolution Frequency: Every 2 generations" in banner

    def test_banner_with_base_model(self):
        """Test banner includes base model path."""
        from src.training.trainers.population.population_trainer import _format_training_banner

        banner = _format_training_banner(
            population_size=8,
            generations=3,
            episodes_per_generation=500,
            evolution_frequency=1,
            base_model_path="/path/to/model.zip"
        )

        assert "Base Model: /path/to/model.zip" in banner

    def test_banner_without_base_model(self):
        """Test banner without base model."""
        from src.training.trainers.population.population_trainer import _format_training_banner

        banner = _format_training_banner(
            population_size=8,
            generations=3,
            episodes_per_generation=500,
            evolution_frequency=1,
            base_model_path=None
        )

        assert "Base Model" not in banner


class TestFormatGenerationHeader:
    """Tests for _format_generation_header helper."""

    def test_header_format(self):
        """Test header format is correct."""
        from src.training.trainers.population.population_trainer import _format_generation_header

        header = _format_generation_header(1, 10)

        assert "GENERATION 1/10" in header
        assert "=" in header

    def test_header_different_values(self):
        """Test header with different generation values."""
        from src.training.trainers.population.population_trainer import _format_generation_header

        header = _format_generation_header(5, 20)

        assert "GENERATION 5/20" in header


class TestFormatFinalReport:
    """Tests for _format_final_report helper."""

    def test_report_contains_required_fields(self):
        """Test report contains all required fields."""
        from src.training.trainers.population.population_trainer import _format_final_report

        metrics = {
            "elo_range": 150.0,
            "elo_std": 45.5
        }

        report = _format_final_report(
            total_generations=10,
            total_matches=100,
            diversity_metrics=metrics
        )

        assert "TRAINING COMPLETE" in report
        assert "Total Generations: 10" in report
        assert "Total Matches: 100" in report
        assert "ELO Spread: 150" in report
        assert "ELO Std Dev: 45.5" in report

    def test_report_with_win_rate_std(self):
        """Test report includes win rate variance."""
        from src.training.trainers.population.population_trainer import _format_final_report

        metrics = {
            "elo_range": 100.0,
            "elo_std": 30.0,
            "win_rate_std": 0.15
        }

        report = _format_final_report(
            total_generations=5,
            total_matches=50,
            diversity_metrics=metrics
        )

        assert "Win Rate Variance: 0.150" in report

    def test_report_without_win_rate_std(self):
        """Test report without win rate variance."""
        from src.training.trainers.population.population_trainer import _format_final_report

        metrics = {
            "elo_range": 100.0,
            "elo_std": 30.0
        }

        report = _format_final_report(
            total_generations=5,
            total_matches=50,
            diversity_metrics=metrics
        )

        assert "Win Rate Variance" not in report


class TestSelectOpponentsForFighter:
    """Tests for _select_opponents_for_fighter helper."""

    def test_select_from_pairs(self):
        """Test selecting opponents from pairs."""
        from src.training.trainers.population.population_trainer import _select_opponents_for_fighter
        from unittest.mock import Mock

        fighter1 = Mock(name="fighter1")
        fighter2 = Mock(name="fighter2")
        fighter3 = Mock(name="fighter3")

        pairs = [(fighter1, fighter2), (fighter1, fighter3)]
        all_fighters = [fighter1, fighter2, fighter3]

        opponents = _select_opponents_for_fighter(fighter1, pairs, all_fighters)

        assert fighter2 in opponents
        assert fighter3 in opponents
        assert fighter1 not in opponents

    def test_select_from_second_position_in_pair(self):
        """Test selecting when fighter is second in pair."""
        from src.training.trainers.population.population_trainer import _select_opponents_for_fighter
        from unittest.mock import Mock

        fighter1 = Mock(name="fighter1")
        fighter2 = Mock(name="fighter2")

        pairs = [(fighter1, fighter2)]
        all_fighters = [fighter1, fighter2]

        opponents = _select_opponents_for_fighter(fighter2, pairs, all_fighters)

        assert fighter1 in opponents

    def test_random_selection_when_no_pairs(self):
        """Test random selection when fighter not in any pairs."""
        from src.training.trainers.population.population_trainer import _select_opponents_for_fighter
        from unittest.mock import Mock

        fighter1 = Mock(name="fighter1")
        fighter2 = Mock(name="fighter2")
        fighter3 = Mock(name="fighter3")
        fighter4 = Mock(name="fighter4")

        # Pairs don't include fighter4
        pairs = [(fighter1, fighter2), (fighter2, fighter3)]
        all_fighters = [fighter1, fighter2, fighter3, fighter4]

        opponents = _select_opponents_for_fighter(fighter4, pairs, all_fighters)

        # Should get random opponents (not fighter4)
        assert len(opponents) <= 3
        assert fighter4 not in opponents

    def test_empty_pairs_list(self):
        """Test with empty pairs list."""
        from src.training.trainers.population.population_trainer import _select_opponents_for_fighter
        from unittest.mock import Mock

        fighter1 = Mock(name="fighter1")
        fighter2 = Mock(name="fighter2")

        all_fighters = [fighter1, fighter2]

        opponents = _select_opponents_for_fighter(fighter1, [], all_fighters)

        assert fighter2 in opponents


class TestCalculateBatchEta:
    """Tests for _calculate_batch_eta helper."""

    def test_eta_calculation(self):
        """Test basic ETA calculation."""
        from src.training.trainers.population.population_trainer import _calculate_batch_eta

        # 5 completed in 10 seconds, 5 remaining
        eta = _calculate_batch_eta(completed=5, total=10, elapsed_time=10.0)

        assert eta == pytest.approx(10.0)  # 5 * 2 seconds each

    def test_eta_zero_completed(self):
        """Test ETA when nothing completed."""
        from src.training.trainers.population.population_trainer import _calculate_batch_eta

        eta = _calculate_batch_eta(completed=0, total=10, elapsed_time=0.0)

        assert eta == 0.0

    def test_eta_all_completed(self):
        """Test ETA when all completed."""
        from src.training.trainers.population.population_trainer import _calculate_batch_eta

        eta = _calculate_batch_eta(completed=10, total=10, elapsed_time=20.0)

        assert eta == pytest.approx(0.0)

    def test_eta_partial_completion(self):
        """Test ETA with partial completion."""
        from src.training.trainers.population.population_trainer import _calculate_batch_eta

        # 3 completed in 15 seconds (5 sec each), 7 remaining
        eta = _calculate_batch_eta(completed=3, total=10, elapsed_time=15.0)

        assert eta == pytest.approx(35.0)  # 7 * 5 seconds


class TestFormatTrainingResultLine:
    """Tests for _format_training_result_line helper."""

    def test_format_basic(self):
        """Test basic formatting."""
        from src.training.trainers.population.population_trainer import _format_training_result_line

        line = _format_training_result_line(
            completed=3,
            total=10,
            fighter_name="test_fighter",
            episodes=50,
            mean_reward=25.5,
            elapsed=12.3
        )

        assert "[3/10]" in line
        assert "test_fighter" in line
        assert "50 episodes" in line
        assert "mean reward: 25.5" in line
        assert "time: 12.3s" in line

    def test_format_completion(self):
        """Test formatting at completion."""
        from src.training.trainers.population.population_trainer import _format_training_result_line

        line = _format_training_result_line(
            completed=10,
            total=10,
            fighter_name="last_fighter",
            episodes=100,
            mean_reward=75.0,
            elapsed=60.0
        )

        assert "[10/10]" in line
        assert "last_fighter" in line

    def test_format_decimal_precision(self):
        """Test decimal precision in formatting."""
        from src.training.trainers.population.population_trainer import _format_training_result_line

        line = _format_training_result_line(
            completed=1,
            total=5,
            fighter_name="fighter",
            episodes=10,
            mean_reward=12.345,
            elapsed=5.678
        )

        assert "mean reward: 12.3" in line  # One decimal place
        assert "time: 5.7s" in line  # One decimal place


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
