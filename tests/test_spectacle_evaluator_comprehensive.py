"""
Comprehensive tests for spectacle_evaluator to increase coverage.
Tests all score calculation branches and edge cases.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.evaluator.spectacle_evaluator import SpectacleEvaluator, SpectacleScore


class TestSpectacleScore:
    """Tests for SpectacleScore dataclass."""

    def test_spectacle_score_creation(self):
        """Test creating a SpectacleScore."""
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
        assert score.overall == 0.6

    def test_spectacle_score_to_dict(self):
        """Test converting SpectacleScore to dict."""
        score = SpectacleScore(
            duration=1.0, close_finish=0.9, stamina_drama=0.8,
            comeback_potential=0.7, positional_exchange=0.6,
            pacing_variety=0.5, collision_drama=0.4, overall=0.7
        )
        d = score.to_dict()
        assert d["duration"] == 1.0
        assert d["close_finish"] == 0.9
        assert d["stamina_drama"] == 0.8
        assert d["comeback_potential"] == 0.7
        assert d["positional_exchange"] == 0.6
        assert d["pacing_variety"] == 0.5
        assert d["collision_drama"] == 0.4
        assert d["overall"] == 0.7


class TestSpectacleEvaluatorInit:
    """Tests for SpectacleEvaluator initialization."""

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        evaluator = SpectacleEvaluator()
        assert evaluator.weights["duration"] == 1.0
        assert evaluator.weights["close_finish"] == 1.0
        assert len(evaluator.weights) == 7

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        custom_weights = {
            "duration": 2.0,
            "close_finish": 1.5,
            "stamina_drama": 1.0,
            "comeback_potential": 1.0,
            "positional_exchange": 0.5,
            "pacing_variety": 0.5,
            "collision_drama": 1.0
        }
        evaluator = SpectacleEvaluator(weights=custom_weights)
        assert evaluator.weights["duration"] == 2.0
        assert evaluator.weights["close_finish"] == 1.5


class TestDurationScoring:
    """Tests for duration score calculation."""

    def _create_match_result(self, total_ticks, final_hp_a, final_hp_b):
        """Helper to create match result mock."""
        result = Mock()
        result.total_ticks = total_ticks
        result.final_hp_a = final_hp_a
        result.final_hp_b = final_hp_b
        result.events = []
        return result

    def _create_basic_telemetry(self, num_ticks):
        """Helper to create basic telemetry."""
        ticks = []
        for i in range(num_ticks):
            ticks.append({
                "fighter_a": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 5, "velocity": 0.5},
                "fighter_b": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 8, "velocity": -0.3},
                "collision": False
            })
        return {"ticks": ticks, "events": []}

    def test_duration_instant_ko(self):
        """Test duration score for very short match (< 30 ticks)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_basic_telemetry(20)
        result = self._create_match_result(20, 100, 0)

        score = evaluator.evaluate(telemetry, result)
        assert score.duration == 0.0  # Instant KO

    def test_duration_too_long(self):
        """Test duration score for very long match (> 500 ticks)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_basic_telemetry(600)
        result = self._create_match_result(600, 50, 40)

        score = evaluator.evaluate(telemetry, result)
        assert score.duration == 0.2  # Boring slugfest

    def test_duration_perfect_length(self):
        """Test duration score for ideal match length (100-400 ticks)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_basic_telemetry(250)
        result = self._create_match_result(250, 10, 0)

        score = evaluator.evaluate(telemetry, result)
        assert score.duration == 1.0  # Perfect length

    def test_duration_short_but_not_instant(self):
        """Test duration score for short match (30-100 ticks)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_basic_telemetry(60)
        result = self._create_match_result(60, 80, 0)

        score = evaluator.evaluate(telemetry, result)
        assert 0 < score.duration < 1.0  # Scaled

    def test_duration_long_but_not_endless(self):
        """Test duration score for longish match (400-500 ticks)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_basic_telemetry(450)
        result = self._create_match_result(450, 60, 40)

        score = evaluator.evaluate(telemetry, result)
        assert 0.2 < score.duration < 1.0  # Partially penalized


class TestCloseFinishScoring:
    """Tests for close_finish score calculation."""

    def _create_match_result(self, total_ticks, final_hp_a, final_hp_b):
        result = Mock()
        result.total_ticks = total_ticks
        result.final_hp_a = final_hp_a
        result.final_hp_b = final_hp_b
        result.events = []
        return result

    def _create_telemetry_with_hp(self, winner_max_hp=100, loser_max_hp=100):
        """Create telemetry with fighter HP info."""
        return {
            "ticks": [{
                "fighter_a": {"hp": 100, "max_hp": winner_max_hp, "stamina": 100, "max_stamina": 100, "position": 5, "velocity": 0.5},
                "fighter_b": {"hp": 100, "max_hp": loser_max_hp, "stamina": 100, "max_stamina": 100, "position": 8, "velocity": -0.3},
                "collision": False
            }],
            "events": []
        }

    def test_close_finish_photo_finish(self):
        """Test close_finish when winner has < 20% HP."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry_with_hp(100, 100)
        result = self._create_match_result(200, 15, 0)  # Winner at 15% HP

        score = evaluator.evaluate(telemetry, result)
        assert score.close_finish == 1.0

    def test_close_finish_close_call(self):
        """Test close_finish when winner has 20-40% HP."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry_with_hp(100, 100)
        result = self._create_match_result(200, 35, 0)  # Winner at 35% HP

        score = evaluator.evaluate(telemetry, result)
        assert score.close_finish == 0.9

    def test_close_finish_competitive(self):
        """Test close_finish when winner has 40-60% HP."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry_with_hp(100, 100)
        result = self._create_match_result(200, 55, 0)  # Winner at 55% HP

        score = evaluator.evaluate(telemetry, result)
        assert score.close_finish == 0.7

    def test_close_finish_dominant(self):
        """Test close_finish when winner has 60-80% HP."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry_with_hp(100, 100)
        result = self._create_match_result(200, 75, 0)  # Winner at 75% HP

        score = evaluator.evaluate(telemetry, result)
        assert score.close_finish == 0.4

    def test_close_finish_boring_stomp(self):
        """Test close_finish when winner has > 80% HP."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry_with_hp(100, 100)
        result = self._create_match_result(200, 95, 0)  # Winner at 95% HP

        score = evaluator.evaluate(telemetry, result)
        assert score.close_finish == 0.0

    def test_close_finish_empty_telemetry(self):
        """Test close_finish with empty telemetry."""
        evaluator = SpectacleEvaluator()
        telemetry = {"ticks": [], "events": []}
        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 50
        result.final_hp_b = 0
        result.events = []

        score = evaluator.evaluate(telemetry, result)
        # Should fall back to default behavior
        assert isinstance(score.close_finish, float)


class TestStaminaDramaScoring:
    """Tests for stamina_drama score calculation."""

    def _create_match_result(self, total_ticks):
        result = Mock()
        result.total_ticks = total_ticks
        result.final_hp_a = 100
        result.final_hp_b = 0
        result.events = []
        return result

    def _create_telemetry_with_stamina(self, stamina_values_a, stamina_values_b, max_stam=100):
        """Create telemetry with specific stamina profiles."""
        ticks = []
        for stam_a, stam_b in zip(stamina_values_a, stamina_values_b):
            ticks.append({
                "fighter_a": {"hp": 100, "max_hp": 100, "stamina": stam_a, "max_stamina": max_stam, "position": 5, "velocity": 0.5},
                "fighter_b": {"hp": 100, "max_hp": 100, "stamina": stam_b, "max_stamina": max_stam, "position": 8, "velocity": -0.3},
                "collision": False
            })
        return {"ticks": ticks, "events": []}

    def test_stamina_drama_ideal(self):
        """Test stamina_drama with ideal 10-30% critical moments."""
        evaluator = SpectacleEvaluator()
        # 20% critical moments (20 out of 100 ticks at <30% stamina)
        stamina_a = [100] * 80 + [20] * 20
        stamina_b = [100] * 100
        telemetry = self._create_telemetry_with_stamina(stamina_a, stamina_b)
        result = self._create_match_result(100)

        score = evaluator.evaluate(telemetry, result)
        assert score.stamina_drama == 1.0

    def test_stamina_drama_low(self):
        """Test stamina_drama with low drama (5-10% critical)."""
        evaluator = SpectacleEvaluator()
        # 7% critical moments
        stamina_a = [100] * 93 + [20] * 7
        stamina_b = [100] * 100
        telemetry = self._create_telemetry_with_stamina(stamina_a, stamina_b)
        result = self._create_match_result(100)

        score = evaluator.evaluate(telemetry, result)
        assert score.stamina_drama == 0.7

    def test_stamina_drama_too_much(self):
        """Test stamina_drama with excessive exhaustion (>30% critical)."""
        evaluator = SpectacleEvaluator()
        # 40% critical moments - too exhausted
        stamina_a = [100] * 60 + [20] * 40
        stamina_b = [100] * 100
        telemetry = self._create_telemetry_with_stamina(stamina_a, stamina_b)
        result = self._create_match_result(100)

        score = evaluator.evaluate(telemetry, result)
        assert score.stamina_drama == 0.5

    def test_stamina_drama_none(self):
        """Test stamina_drama with no drama (<5% critical)."""
        evaluator = SpectacleEvaluator()
        # 2% critical moments
        stamina_a = [100] * 98 + [20] * 2
        stamina_b = [100] * 100
        telemetry = self._create_telemetry_with_stamina(stamina_a, stamina_b)
        result = self._create_match_result(100)

        score = evaluator.evaluate(telemetry, result)
        assert score.stamina_drama == 0.3

    def test_stamina_drama_empty(self):
        """Test stamina_drama with no samples."""
        evaluator = SpectacleEvaluator()
        telemetry = {"ticks": [], "events": []}
        result = Mock()
        result.total_ticks = 0
        result.final_hp_a = 100
        result.final_hp_b = 0
        result.events = []

        score = evaluator.evaluate(telemetry, result)
        assert score.stamina_drama == 0.0


class TestComebackPotentialScoring:
    """Tests for comeback_potential score calculation."""

    def _create_match_result(self, total_ticks):
        result = Mock()
        result.total_ticks = total_ticks
        result.final_hp_a = 100
        result.final_hp_b = 0
        result.events = []
        return result

    def _create_telemetry_with_hp_changes(self, hp_pairs):
        """Create telemetry with specific HP progression."""
        ticks = []
        for hp_a, hp_b in hp_pairs:
            ticks.append({
                "fighter_a": {"hp": hp_a, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 5, "velocity": 0.5},
                "fighter_b": {"hp": hp_b, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 8, "velocity": -0.3},
                "collision": False
            })
        return {"ticks": ticks, "events": []}

    def test_comeback_many_lead_changes(self):
        """Test comeback_potential with 3+ lead changes."""
        evaluator = SpectacleEvaluator()
        # A leads, B leads, A leads, B leads - 3 changes
        hp_pairs = [(100, 80), (80, 90), (90, 70), (70, 80), (90, 60)]
        telemetry = self._create_telemetry_with_hp_changes(hp_pairs)
        result = self._create_match_result(5)

        score = evaluator.evaluate(telemetry, result)
        assert score.comeback_potential == 1.0

    def test_comeback_two_lead_changes(self):
        """Test comeback_potential with 2 lead changes."""
        evaluator = SpectacleEvaluator()
        # A leads, B leads, A leads - 2 changes
        hp_pairs = [(100, 80), (80, 90), (90, 70), (95, 60), (100, 50)]
        telemetry = self._create_telemetry_with_hp_changes(hp_pairs)
        result = self._create_match_result(5)

        score = evaluator.evaluate(telemetry, result)
        assert score.comeback_potential == 0.8

    def test_comeback_one_lead_change(self):
        """Test comeback_potential with 1 lead change."""
        evaluator = SpectacleEvaluator()
        # A leads, B leads, done - 1 change
        hp_pairs = [(100, 80), (80, 90), (70, 90), (60, 85), (50, 80)]
        telemetry = self._create_telemetry_with_hp_changes(hp_pairs)
        result = self._create_match_result(5)

        score = evaluator.evaluate(telemetry, result)
        assert score.comeback_potential == 0.5

    def test_comeback_no_lead_changes(self):
        """Test comeback_potential with no lead changes."""
        evaluator = SpectacleEvaluator()
        # A always leads
        hp_pairs = [(100, 80), (95, 70), (90, 60), (85, 50), (80, 40)]
        telemetry = self._create_telemetry_with_hp_changes(hp_pairs)
        result = self._create_match_result(5)

        score = evaluator.evaluate(telemetry, result)
        assert score.comeback_potential == 0.2

    def test_comeback_too_few_samples(self):
        """Test comeback_potential with < 5 samples."""
        evaluator = SpectacleEvaluator()
        hp_pairs = [(100, 80), (90, 70), (80, 60)]
        telemetry = self._create_telemetry_with_hp_changes(hp_pairs)
        result = self._create_match_result(3)

        score = evaluator.evaluate(telemetry, result)
        assert score.comeback_potential == 0.3


class TestIntegration:
    """Integration tests for SpectacleEvaluator."""

    def test_full_evaluation(self):
        """Test full evaluation with realistic telemetry."""
        evaluator = SpectacleEvaluator()

        # Create realistic match telemetry
        ticks = []
        for i in range(200):
            hp_a = max(0, 100 - i * 0.3)
            hp_b = max(0, 100 - i * 0.4)
            stam_a = 50 + 50 * abs(((i % 40) - 20) / 20)  # Oscillating
            stam_b = 50 + 50 * abs(((i % 30) - 15) / 15)
            pos_a = 3 + (i % 5)
            pos_b = 8 - (i % 3)
            vel_a = 0.5 * (1 if i % 2 == 0 else -1)
            vel_b = 0.3 * (1 if i % 3 == 0 else -1)
            ticks.append({
                "fighter_a": {"hp": hp_a, "max_hp": 100, "stamina": stam_a, "max_stamina": 100, "position": pos_a, "velocity": vel_a},
                "fighter_b": {"hp": hp_b, "max_hp": 100, "stamina": stam_b, "max_stamina": 100, "position": pos_b, "velocity": vel_b},
                "collision": i % 15 == 0
            })

        telemetry = {"ticks": ticks, "events": []}
        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 40
        result.final_hp_b = 20
        result.events = []

        score = evaluator.evaluate(telemetry, result)

        assert isinstance(score, SpectacleScore)
        assert 0 <= score.duration <= 1
        assert 0 <= score.close_finish <= 1
        assert 0 <= score.stamina_drama <= 1
        assert 0 <= score.comeback_potential <= 1
        assert 0 <= score.overall <= 1

    def test_overall_weighted_average(self):
        """Test that overall is a weighted average of other scores."""
        evaluator = SpectacleEvaluator()

        # Create simple telemetry
        ticks = [{
            "fighter_a": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 5, "velocity": 0.5},
            "fighter_b": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 8, "velocity": -0.3},
            "collision": False
        } for _ in range(200)]

        telemetry = {"ticks": ticks, "events": []}
        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 50
        result.final_hp_b = 0
        result.events = []

        score = evaluator.evaluate(telemetry, result)

        # Overall should be somewhere between min and max of other scores
        other_scores = [
            score.duration, score.close_finish, score.stamina_drama,
            score.comeback_potential, score.positional_exchange,
            score.pacing_variety, score.collision_drama
        ]
        assert min(other_scores) <= score.overall <= max(other_scores)


class TestCollisionDrama:
    """Tests for collision_drama scoring (lines 228-241)."""

    def _create_telemetry(self, num_ticks):
        """Create telemetry with velocity data."""
        return {
            "ticks": [{
                "fighter_a": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 3+i*0.1, "velocity": 0.5},
                "fighter_b": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 9-i*0.1, "velocity": -0.5},
            } for i in range(num_ticks)]
        }

    def test_collision_drama_ideal_count_and_damage(self):
        """Test collision drama with ideal collision count (8-25) and high damage."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry(200)

        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 20
        result.final_hp_b = 0
        # 15 collisions with good damage (avg > 3.0 per collision)
        result.events = [
            {"type": "COLLISION", "damage_to_a": 5, "damage_to_b": 5}
            for _ in range(15)
        ]

        score = evaluator.evaluate(telemetry, result)
        assert score.collision_drama == 1.0

    def test_collision_drama_few_collisions(self):
        """Test collision drama with few collisions (< 8)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry(200)

        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 80
        result.final_hp_b = 0
        # Only 4 collisions
        result.events = [
            {"type": "COLLISION", "damage_to_a": 5, "damage_to_b": 5}
            for _ in range(4)
        ]

        score = evaluator.evaluate(telemetry, result)
        assert score.collision_drama == 4 / 8  # Scaled

    def test_collision_drama_too_many_collisions(self):
        """Test collision drama with too many collisions (> 25) - wall grinding."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry(200)

        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 20
        result.final_hp_b = 0
        # 30 collisions - wall grinding
        result.events = [
            {"type": "COLLISION", "damage_to_a": 5, "damage_to_b": 5}
            for _ in range(30)
        ]

        score = evaluator.evaluate(telemetry, result)
        assert score.collision_drama == 0.4

    def test_collision_drama_in_range_but_low_damage(self):
        """Test collision drama with ideal count but low avg damage (< 3.0)."""
        evaluator = SpectacleEvaluator()
        telemetry = self._create_telemetry(200)

        result = Mock()
        result.total_ticks = 200
        result.final_hp_a = 90
        result.final_hp_b = 85
        # 15 collisions but very low damage (avg < 3.0)
        result.events = [
            {"type": "COLLISION", "damage_to_a": 1, "damage_to_b": 1}
            for _ in range(15)
        ]

        score = evaluator.evaluate(telemetry, result)
        assert score.collision_drama == 0.5


class TestPacingVarietyEdgeCases:
    """Tests for pacing_variety scoring edge cases."""

    def test_pacing_variety_high_std_dev(self):
        """Test pacing variety with high velocity std_dev (> 1.5)."""
        evaluator = SpectacleEvaluator()

        # Create telemetry with high velocity variance
        ticks = []
        for i in range(50):
            vel = 5.0 if i % 2 == 0 else 0.1  # Alternating high/low
            ticks.append({
                "fighter_a": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 5, "velocity": vel},
                "fighter_b": {"hp": 100, "max_hp": 100, "stamina": 100, "max_stamina": 100, "position": 7, "velocity": -vel},
            })

        telemetry = {"ticks": ticks}
        result = Mock()
        result.total_ticks = 50
        result.final_hp_a = 50
        result.final_hp_b = 0
        result.events = []

        score = evaluator.evaluate(telemetry, result)
        # High std_dev should result in reduced but capped score
        assert 0 <= score.pacing_variety <= 1.0
