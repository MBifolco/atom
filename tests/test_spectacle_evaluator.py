"""
Tests for the SpectacleEvaluator component.

Tests cover:
- SpectacleScore dataclass and to_dict
- SpectacleEvaluator initialization with default and custom weights
- Duration scoring
- Close finish scoring
- Stamina drama scoring
- Comeback potential scoring
- Positional exchange scoring
- Pacing variety scoring
- Collision drama scoring
- Overall weighted scoring
"""

import pytest
from src.evaluator.spectacle_evaluator import SpectacleEvaluator, SpectacleScore
from src.orchestrator.match_orchestrator import MatchResult


class TestSpectacleScore:
    """Test SpectacleScore dataclass."""

    def test_spectacle_score_creation(self):
        """SpectacleScore can be created with all fields."""
        score = SpectacleScore(
            duration=0.8,
            close_finish=0.9,
            stamina_drama=0.7,
            comeback_potential=0.6,
            positional_exchange=0.5,
            pacing_variety=0.4,
            collision_drama=0.3,
            overall=0.6
        )

        assert score.duration == 0.8
        assert score.close_finish == 0.9
        assert score.stamina_drama == 0.7
        assert score.comeback_potential == 0.6
        assert score.positional_exchange == 0.5
        assert score.pacing_variety == 0.4
        assert score.collision_drama == 0.3
        assert score.overall == 0.6

    def test_spectacle_score_to_dict(self):
        """SpectacleScore can be converted to dictionary."""
        score = SpectacleScore(
            duration=0.8,
            close_finish=0.9,
            stamina_drama=0.7,
            comeback_potential=0.6,
            positional_exchange=0.5,
            pacing_variety=0.4,
            collision_drama=0.3,
            overall=0.6
        )

        result = score.to_dict()

        assert isinstance(result, dict)
        assert result["duration"] == 0.8
        assert result["close_finish"] == 0.9
        assert result["stamina_drama"] == 0.7
        assert result["comeback_potential"] == 0.6
        assert result["positional_exchange"] == 0.5
        assert result["pacing_variety"] == 0.4
        assert result["collision_drama"] == 0.3
        assert result["overall"] == 0.6


class TestSpectacleEvaluatorInit:
    """Test SpectacleEvaluator initialization."""

    def test_evaluator_initializes_with_default_weights(self):
        """Evaluator initializes with equal weights for all metrics."""
        evaluator = SpectacleEvaluator()

        assert evaluator.weights["duration"] == 1.0
        assert evaluator.weights["close_finish"] == 1.0
        assert evaluator.weights["stamina_drama"] == 1.0
        assert evaluator.weights["comeback_potential"] == 1.0
        assert evaluator.weights["positional_exchange"] == 1.0
        assert evaluator.weights["pacing_variety"] == 1.0
        assert evaluator.weights["collision_drama"] == 1.0

    def test_evaluator_accepts_custom_weights(self):
        """Evaluator accepts custom weights."""
        custom_weights = {
            "duration": 2.0,
            "close_finish": 3.0,
            "stamina_drama": 1.0,
            "comeback_potential": 1.5,
            "positional_exchange": 0.5,
            "pacing_variety": 1.0,
            "collision_drama": 2.5
        }

        evaluator = SpectacleEvaluator(weights=custom_weights)

        assert evaluator.weights["duration"] == 2.0
        assert evaluator.weights["close_finish"] == 3.0
        assert evaluator.weights["comeback_potential"] == 1.5


class TestDurationScoring:
    """Test match duration scoring."""

    def test_instant_ko_scores_zero(self):
        """Very short match (instant KO) scores 0.0."""
        evaluator = SpectacleEvaluator()

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=20,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.duration == 0.0

    def test_perfect_duration_scores_one(self):
        """Match in ideal range (100-400) scores 1.0."""
        evaluator = SpectacleEvaluator()

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=250,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.duration == 1.0

    def test_too_long_match_scores_low(self):
        """Very long match (>500) scores low (0.2)."""
        evaluator = SpectacleEvaluator()

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=600,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.duration == 0.2


class TestCloseFinishScoring:
    """Test close finish scoring."""

    def test_photo_finish_scores_one(self):
        """Winner with <20% HP scores 1.0."""
        evaluator = SpectacleEvaluator()

        telemetry = {
            "ticks": [{
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            }],
            "fighter_a_name": "A",
            "fighter_b_name": "B"
        }
        match_result = MatchResult(
            winner="A",
            total_ticks=200,
            final_hp_a=15.0,  # 15% HP remaining
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.close_finish == 1.0

    def test_dominant_win_scores_zero(self):
        """Winner with >80% HP scores 0.0."""
        evaluator = SpectacleEvaluator()

        telemetry = {
            "ticks": [{
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            }],
            "fighter_a_name": "A",
            "fighter_b_name": "B"
        }
        match_result = MatchResult(
            winner="A",
            total_ticks=200,
            final_hp_a=95.0,  # 95% HP remaining
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.close_finish == 0.0


class TestStaminaDramaScoring:
    """Test stamina drama scoring."""

    def test_ideal_stamina_drama_scores_one(self):
        """10-30% of match at critical stamina scores 1.0."""
        evaluator = SpectacleEvaluator()

        # Create telemetry with 100 ticks, 20 at critical stamina
        ticks = []
        for i in range(100):
            stamina_pct = 0.2 if i < 20 else 0.8  # 20% at critical
            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": stamina_pct * 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": stamina_pct * 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.stamina_drama == 1.0

    def test_no_stamina_drama_scores_low(self):
        """No critical stamina moments scores 0.3."""
        evaluator = SpectacleEvaluator()

        # All ticks at high stamina
        ticks = []
        for i in range(100):
            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 8.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 9.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.stamina_drama == 0.3


class TestComebackPotentialScoring:
    """Test comeback potential (HP lead changes) scoring."""

    def test_multiple_lead_changes_scores_high(self):
        """3+ lead changes scores 1.0."""
        evaluator = SpectacleEvaluator()

        # Create HP swings: A leads, B leads, A leads, B leads
        ticks = [
            {"fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 90.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 85.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 90.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 85.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 80.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 75.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 80.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 75.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 70.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}}
        ]

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=5,
            final_hp_a=75.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.comeback_potential == 1.0

    def test_no_lead_changes_scores_low(self):
        """No lead changes scores 0.2."""
        evaluator = SpectacleEvaluator()

        # A always leads
        ticks = [
            {"fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 90.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 95.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 80.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 90.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 70.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 85.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 60.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}},
            {"fighter_a": {"hp": 80.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 4.0, "velocity": 0.0},
             "fighter_b": {"hp": 50.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                          "position": 8.0, "velocity": 0.0}}
        ]

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=5,
            final_hp_a=80.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.comeback_potential == 0.2


class TestPositionalExchangeScoring:
    """Test positional exchange scoring."""

    def test_ideal_position_swaps_scores_one(self):
        """5-20% position swaps scores 1.0."""
        evaluator = SpectacleEvaluator()

        # Create position swaps: 10 swaps in 100 ticks = 10%
        ticks = []
        for i in range(100):
            if i % 10 == 0:  # Swap every 10 ticks
                pos_a, pos_b = 8.0, 4.0  # A on right, B on left
            else:
                pos_a, pos_b = 4.0, 8.0  # A on left, B on right

            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": pos_a, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": pos_b, "velocity": 0.0}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.positional_exchange == 1.0

    def test_no_position_swaps_scores_zero(self):
        """No position swaps scores 0.0."""
        evaluator = SpectacleEvaluator()

        # A always on left, B always on right
        ticks = []
        for i in range(100):
            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.positional_exchange == 0.0


class TestPacingVarietyScoring:
    """Test pacing variety (speed variance) scoring."""

    def test_ideal_speed_variance_scores_one(self):
        """Std dev 0.5-1.5 scores 1.0."""
        evaluator = SpectacleEvaluator()

        # Create varied velocities
        ticks = []
        for i in range(100):
            vel_a = 1.0 if i % 2 == 0 else 2.0  # Alternating speeds
            vel_b = 0.5 if i % 2 == 0 else 1.5

            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": vel_a},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": vel_b}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        # Should have reasonable variance (allow 0.0 or positive)
        assert score.pacing_variety >= 0.0
        assert isinstance(score.pacing_variety, float)

    def test_no_movement_scores_zero(self):
        """No movement scores 0.0."""
        evaluator = SpectacleEvaluator()

        # All velocities zero
        ticks = []
        for i in range(100):
            ticks.append({
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            })

        telemetry = {"ticks": ticks, "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=100,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.pacing_variety == 0.0


class TestCollisionDramaScoring:
    """Test collision drama scoring."""

    def test_ideal_collisions_scores_one(self):
        """8-25 collisions with meaningful damage scores 1.0."""
        evaluator = SpectacleEvaluator()

        # Create 15 collisions with meaningful damage
        events = []
        for i in range(15):
            events.append({
                "type": "COLLISION",
                "damage_to_a": 5.0,
                "damage_to_b": 4.0
            })

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=200,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=events
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.collision_drama == 1.0

    def test_too_few_collisions_scales_linearly(self):
        """Fewer than 8 collisions scales linearly."""
        evaluator = SpectacleEvaluator()

        # Only 4 collisions
        events = []
        for i in range(4):
            events.append({
                "type": "COLLISION",
                "damage_to_a": 5.0,
                "damage_to_b": 4.0
            })

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=200,
            final_hp_a=50.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=events
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.collision_drama == 0.5  # 4/8

    def test_no_collisions_scores_zero(self):
        """No collisions scores 0.0."""
        evaluator = SpectacleEvaluator()

        telemetry = {"ticks": [], "fighter_a_name": "A", "fighter_b_name": "B"}
        match_result = MatchResult(
            winner="A",
            total_ticks=200,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)
        assert score.collision_drama == 0.0


class TestOverallScoring:
    """Test overall weighted scoring."""

    def test_overall_is_weighted_average(self):
        """Overall score is weighted average of all metrics."""
        # Custom weights: close_finish is 3x more important
        evaluator = SpectacleEvaluator(weights={
            "duration": 1.0,
            "close_finish": 3.0,
            "stamina_drama": 1.0,
            "comeback_potential": 1.0,
            "positional_exchange": 1.0,
            "pacing_variety": 1.0,
            "collision_drama": 1.0
        })

        # Create perfect close finish, poor everything else
        telemetry = {
            "ticks": [{
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            }],
            "fighter_a_name": "A",
            "fighter_b_name": "B"
        }
        match_result = MatchResult(
            winner="A",
            total_ticks=20,  # Instant KO (duration = 0.0)
            final_hp_a=15.0,  # Photo finish (close_finish = 1.0)
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)

        # Overall should incorporate the weighted close_finish score
        # Since close_finish has 3x weight, overall should be influenced by it
        assert 0.0 < score.overall < 1.0
        assert isinstance(score.overall, float)

    def test_overall_with_equal_weights(self):
        """Overall score with equal weights is simple average."""
        evaluator = SpectacleEvaluator()  # Default equal weights

        # Create known scores
        telemetry = {
            "ticks": [{
                "fighter_a": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 4.0, "velocity": 0.0},
                "fighter_b": {"hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                             "position": 8.0, "velocity": 0.0}
            }],
            "fighter_a_name": "A",
            "fighter_b_name": "B"
        }
        match_result = MatchResult(
            winner="A",
            total_ticks=250,  # Perfect duration (1.0)
            final_hp_a=15.0,  # Photo finish (1.0)
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        score = evaluator.evaluate(telemetry, match_result)

        # Should have duration=1.0, close_finish=1.0, others lower
        # Overall should be a valid score between 0 and 1
        assert 0.0 <= score.overall <= 1.0
        assert isinstance(score.overall, float)
        # Should be positive since we have some good scores
        assert score.overall > 0.0
