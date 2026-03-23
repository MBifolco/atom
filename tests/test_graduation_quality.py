"""Tests for the graduation quality gate — combat metrics required for graduation."""

from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for TrainingProgress / CurriculumLevel
# so these tests don't depend on the full trainer import graph.
# ---------------------------------------------------------------------------


@dataclass
class _FakeProgress:
    current_level: int = 0
    episodes_at_level: int = 100
    wins_at_level: int = 90
    recent_episodes: List[bool] = field(default_factory=lambda: [True] * 50)
    recent_damage_dealt: List[float] = field(default_factory=list)
    graduated_levels: List[str] = field(default_factory=list)
    total_episodes: int = 100
    total_wins: int = 90


@dataclass
class _FakeLevel:
    name: str = "Fundamentals"
    min_episodes: int = 50
    graduation_win_rate: float = 0.9
    graduation_episodes: int = 50
    description: str = ""


# ---------------------------------------------------------------------------
# Tests — stochastic quality metrics
# ---------------------------------------------------------------------------


class TestGraduationQualityGate:
    """Tests for min_mean_damage_dealt and min_nonzero_damage_rate."""

    def test_zero_damage_blocks_graduation(self):
        """A fighter with high win rate but 0 damage should NOT graduate."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=[0.0] * 50,
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert not decision.should_graduate
        assert not decision.combat_quality_passed

    def test_good_damage_allows_graduation(self):
        """A fighter with high win rate AND good damage should graduate."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=[10.0] * 50,  # all episodes have damage
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert decision.should_graduate
        assert decision.combat_quality_passed

    def test_low_nonzero_rate_blocks_graduation(self):
        """Fighter with enough mean damage but rare hits should NOT graduate."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        # 5 episodes with 50 damage each, 45 with 0 → mean = 5.0, rate = 10%
        damage = [50.0] * 5 + [0.0] * 45
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=damage,
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert not decision.should_graduate
        assert not decision.combat_quality_passed

    def test_thresholds_configurable(self):
        """Custom thresholds should be respected."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=1.0,  # very low
            min_nonzero_damage_rate=0.1,  # very low
        )
        # 6 episodes with 10 damage, 44 with 0 → mean = 1.2, rate = 12%
        damage = [10.0] * 6 + [0.0] * 44
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=damage,
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert decision.should_graduate
        assert decision.combat_quality_passed

    def test_override_mode_bypasses_quality_checks(self):
        """Override graduation (fixed episode count) should skip combat quality."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=50,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        progress = _FakeProgress(
            episodes_at_level=50,
            wins_at_level=0,
            recent_episodes=[False] * 50,
            recent_damage_dealt=[0.0] * 50,
        )
        level = _FakeLevel()

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert decision.should_graduate
        assert decision.reason == "override"

    def test_win_rate_failure_still_blocks_even_with_good_damage(self):
        """Win rate gates should still apply even if damage is good."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=30,  # low overall win rate
            recent_episodes=[False] * 50,  # low recent win rate
            recent_damage_dealt=[20.0] * 50,  # good damage
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert not decision.should_graduate

    def test_empty_damage_list_blocks(self):
        """No damage data means quality gate should fail."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=[],  # no data
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert not decision.should_graduate
        assert not decision.combat_quality_passed

    def test_decision_includes_damage_metrics(self):
        """GraduationDecision should expose damage metrics for logging."""
        from src.atom.training.trainers.curriculum_components import GraduationPolicy

        policy = GraduationPolicy(
            override_episodes_per_level=None,
            min_mean_damage_dealt=5.0,
            min_nonzero_damage_rate=0.3,
        )
        damage = [15.0] * 30 + [0.0] * 20
        progress = _FakeProgress(
            episodes_at_level=100,
            wins_at_level=95,
            recent_episodes=[True] * 50,
            recent_damage_dealt=damage,
        )
        level = _FakeLevel(graduation_win_rate=0.9, graduation_episodes=50)

        decision = policy.evaluate(progress=progress, level=level, curriculum_size=5)
        assert hasattr(decision, "mean_damage_dealt")
        assert hasattr(decision, "nonzero_damage_rate")
        assert decision.mean_damage_dealt == pytest.approx(9.0)
        assert decision.nonzero_damage_rate == pytest.approx(0.6)
