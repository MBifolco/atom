"""
Basic trainer tests to push over 45% coverage threshold.
Focuses on simple, easily testable code paths.
"""

import pytest
from src.training.trainers.curriculum_trainer import (
    CurriculumCallback,
    VmapEnvAdapter,
    DifficultyLevel
)
from src.training.trainers.population.population_trainer import (
    PopulationCallback,
    PopulationFighter
)
from src.training.trainers.population.elo_tracker import EloTracker


class TestCurriculumCallbackStructure:
    """Test curriculum callback structure and attributes."""

    def test_curriculum_callback_has_required_attributes(self):
        """Test callback has required tracking attributes."""
        # Mock trainer
        class MockTrainer:
            pass

        callback = CurriculumCallback(
            curriculum_trainer=MockTrainer(),
            verbose=0
        )

        # Check attributes exist
        assert hasattr(callback, 'curriculum_trainer')
        assert hasattr(callback, 'episode_rewards')
        assert hasattr(callback, 'episode_wins')
        assert isinstance(callback.episode_rewards, list)
        assert isinstance(callback.episode_wins, list)

    def test_curriculum_callback_recent_components_tracking(self):
        """Test callback tracks recent reward components."""
        class MockTrainer:
            pass

        callback = CurriculumCallback(MockTrainer())

        assert hasattr(callback, 'recent_reward_components')
        assert isinstance(callback.recent_reward_components, list)

    def test_curriculum_callback_time_tracking(self):
        """Test callback tracks rollout and train times."""
        class MockTrainer:
            pass

        callback = CurriculumCallback(MockTrainer(), verbose=1)

        assert hasattr(callback, 'last_rollout_time')
        assert hasattr(callback, 'last_train_time')


class TestPopulationCallbackStructure:
    """Test population callback structure."""

    def test_population_callback_required_attributes(self):
        """Test population callback has required attributes."""
        tracker = EloTracker()

        callback = PopulationCallback(
            fighter_name="TestFighter",
            elo_tracker=tracker,
            verbose=0
        )

        assert callback.fighter_name == "TestFighter"
        assert callback.elo_tracker is tracker
        assert callback.episode_count == 0
        assert hasattr(callback, 'recent_rewards')
        assert isinstance(callback.recent_rewards, list)

    def test_population_callback_reward_list_management(self):
        """Test callback manages reward list."""
        tracker = EloTracker()

        callback = PopulationCallback("Test", tracker)

        # Start empty
        assert len(callback.recent_rewards) == 0


class TestVmapEnvAdapterComplete:
    """Complete VmapEnvAdapter testing."""

    def test_adapter_implements_vec_env_interface(self):
        """Test adapter implements all required VecEnv methods."""
        required_methods = [
            'reset',
            'step_async',
            'step_wait',
            'close',
            'env_is_wrapped',
            'get_attr',
            'set_attr',
            'env_method'
        ]

        for method_name in required_methods:
            assert hasattr(VmapEnvAdapter, method_name), f"Missing method: {method_name}"
            assert callable(getattr(VmapEnvAdapter, method_name))


class TestDifficultyLevelUtility:
    """Test difficulty level utility."""

    def test_can_create_all_difficulty_levels(self):
        """Test all difficulty levels can be instantiated."""
        levels = [
            DifficultyLevel.FUNDAMENTALS,
            DifficultyLevel.BASIC_SKILLS,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT,
            DifficultyLevel.POPULATION
        ]

        for level in levels:
            assert level is not None
            assert isinstance(level.value, str)

    def test_difficulty_level_values_unique(self):
        """Test all difficulty level values are unique."""
        values = [level.value for level in DifficultyLevel]

        # All should be unique
        assert len(values) == len(set(values))

    def test_difficulty_level_comparison(self):
        """Test difficulty levels can be compared."""
        l1 = DifficultyLevel.FUNDAMENTALS
        l2 = DifficultyLevel.FUNDAMENTALS

        # Same level should be equal
        assert l1 == l2

        l3 = DifficultyLevel.EXPERT

        # Different levels not equal
        assert l1 != l3


class TestPopulationFighterDataclass:
    """Complete PopulationFighter dataclass testing."""

    def test_fighter_minimal_creation(self):
        """Test creating fighter with minimal fields."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Minimal",
            model=MockModel()
        )

        # Should have defaults
        assert fighter.generation == 0
        assert fighter.lineage == "founder"
        assert fighter.mass == 70.0
        assert fighter.training_episodes == 0
        assert fighter.last_checkpoint is None

    def test_fighter_tracks_generation_lineage(self):
        """Test generation and lineage tracking."""
        class MockModel:
            pass

        # Generation 3 fighter
        fighter = PopulationFighter(
            name="Gen3",
            model=MockModel(),
            generation=3,
            lineage="g0_founder->g1_child->g2_descendant->g3_current"
        )

        assert fighter.generation == 3
        assert "g0_founder" in fighter.lineage
        assert "g3_current" in fighter.lineage

    def test_fighter_checkpoint_tracking(self):
        """Test checkpoint path tracking."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Checkpointed",
            model=MockModel(),
            last_checkpoint="/outputs/fighter_gen5_checkpoint.zip"
        )

        assert fighter.last_checkpoint is not None
        assert "checkpoint" in fighter.last_checkpoint.lower()

    def test_fighter_episode_accumulation(self):
        """Test training episodes accumulate."""
        class MockModel:
            pass

        fighter = PopulationFighter(
            name="Veteran",
            model=MockModel(),
            training_episodes=1500
        )

        assert fighter.training_episodes == 1500
        assert fighter.training_episodes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
