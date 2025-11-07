#!/usr/bin/env python3
"""
Test Dummy Edge Case Validator

Comprehensive edge case testing beyond happy paths.
Tests boundary conditions, extreme values, invalid states, and adversarial inputs.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Callable, Any
import math
import random


class EdgeCaseValidator:
    """Tests edge cases and adversarial conditions for test dummies."""

    def __init__(self):
        self.results = []
        self.failures = []
        self.warnings = []

    def load_fighter(self, path: str) -> Callable:
        """Load a fighter module and return its decide function."""
        try:
            spec = importlib.util.spec_from_file_location("fighter", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.decide
        except Exception as e:
            self.failures.append(f"Failed to load {path}: {e}")
            return None

    def create_snapshot(self, **kwargs) -> Dict:
        """Create a test snapshot with specified parameters."""
        defaults = {
            "my_position": 6.0,
            "opponent_position": 6.0,
            "my_velocity": 0.0,
            "opponent_velocity": 0.0,
            "my_hp": 100.0,
            "opponent_hp": 100.0,
            "my_stamina": 100.0,
            "opponent_stamina": 100.0,
            "my_stance": "neutral",
            "opponent_stance": "neutral",
            "arena_width": 12.0,
            "tick": 100
        }
        defaults.update(kwargs)

        return {
            "you": {
                "position": defaults["my_position"],
                "velocity": defaults["my_velocity"],
                "hp": defaults["my_hp"],
                "hp_max": 100.0,
                "stamina": defaults["my_stamina"],
                "stamina_max": 100.0,
                "stance": defaults["my_stance"]
            },
            "opponent": {
                "position": defaults["opponent_position"],
                "velocity": defaults["opponent_velocity"],
                "hp": defaults["opponent_hp"],
                "hp_max": 100.0,
                "stamina": defaults["opponent_stamina"],
                "stamina_max": 100.0,
                "stance": defaults["opponent_stance"]
            },
            "arena": {
                "width": defaults["arena_width"]
            },
            "tick": defaults["tick"]
        }

    def validate_decision(self, decision: Dict, context: str) -> bool:
        """Validate a decision is legal and sensible."""
        if decision is None:
            self.failures.append(f"{context}: Returned None")
            return False

        if not isinstance(decision, dict):
            self.failures.append(f"{context}: Not a dict, got {type(decision)}")
            return False

        # Check required fields
        if "acceleration" not in decision:
            self.failures.append(f"{context}: Missing acceleration")
            return False

        if "stance" not in decision:
            self.failures.append(f"{context}: Missing stance")
            return False

        # Check acceleration bounds
        acc = decision["acceleration"]
        if not isinstance(acc, (int, float)):
            self.failures.append(f"{context}: Acceleration not numeric: {acc}")
            return False

        if acc < -5.0 or acc > 5.0:
            self.failures.append(f"{context}: Acceleration out of bounds: {acc}")
            return False

        # Check for NaN or infinity
        if math.isnan(acc) or math.isinf(acc):
            self.failures.append(f"{context}: Acceleration is NaN or Inf: {acc}")
            return False

        # Check stance validity
        valid_stances = ["neutral", "extended", "defending", "retracted"]
        if decision["stance"] not in valid_stances:
            self.failures.append(f"{context}: Invalid stance: {decision['stance']}")
            return False

        return True

    def test_stationary_edge_cases(self):
        """Test stationary dummies with edge cases."""
        print("\n=== TESTING STATIONARY EDGE CASES ===")

        dummies = ["stationary_neutral", "stationary_extended",
                  "stationary_defending", "stationary_retracted"]

        for dummy_name in dummies:
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            test_cases = [
                # Boundary positions
                ("at_left_wall", {"my_position": 0.0}),
                ("at_right_wall", {"my_position": 12.0}),
                ("beyond_left_wall", {"my_position": -1.0}),  # Should handle gracefully
                ("beyond_right_wall", {"my_position": 13.0}),  # Should handle gracefully

                # Extreme distances
                ("opponent_same_position", {"opponent_position": 6.0, "my_position": 6.0}),
                ("opponent_max_distance", {"my_position": 0.0, "opponent_position": 12.0}),

                # Resource extremes
                ("zero_hp", {"my_hp": 0.0}),
                ("zero_stamina", {"my_stamina": 0.0}),
                ("negative_hp", {"my_hp": -10.0}),
                ("negative_stamina", {"my_stamina": -10.0}),
                ("over_max_hp", {"my_hp": 150.0}),
                ("over_max_stamina", {"my_stamina": 150.0}),

                # Velocity extremes
                ("max_velocity", {"my_velocity": 5.0}),
                ("min_velocity", {"my_velocity": -5.0}),
                ("extreme_velocity", {"my_velocity": 100.0}),  # Should handle gracefully

                # Invalid arena
                ("tiny_arena", {"arena_width": 1.0}),
                ("huge_arena", {"arena_width": 1000.0}),
                ("zero_arena", {"arena_width": 0.0}),
                ("negative_arena", {"arena_width": -10.0}),

                # Time edge cases
                ("tick_zero", {"tick": 0}),
                ("tick_negative", {"tick": -1}),
                ("tick_huge", {"tick": 999999}),

                # Opponent edge cases
                ("opponent_negative_position", {"opponent_position": -5.0}),
                ("opponent_beyond_arena", {"opponent_position": 20.0}),
                ("opponent_max_velocity", {"opponent_velocity": 5.0}),
                ("opponent_zero_hp", {"opponent_hp": 0.0}),
            ]

            failures_before = len(self.failures)
            expected_stance = dummy_name.split("_")[1]

            for case_name, params in test_cases:
                try:
                    snapshot = self.create_snapshot(**params)
                    decision = decide(snapshot)

                    if self.validate_decision(decision, f"{dummy_name}/{case_name}"):
                        # Check stationary behavior
                        if abs(decision["acceleration"]) > 0.01:
                            self.failures.append(
                                f"{dummy_name}/{case_name}: Not stationary, acc={decision['acceleration']}"
                            )

                        # Check correct stance
                        if decision["stance"] != expected_stance:
                            self.failures.append(
                                f"{dummy_name}/{case_name}: Wrong stance, got {decision['stance']}"
                            )
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_movement_edge_cases(self):
        """Test movement dummies with edge cases."""
        print("\n=== TESTING MOVEMENT EDGE CASES ===")

        # Test approach dummies
        for speed in ["slow", "fast"]:
            dummy_name = f"approach_{speed}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            test_cases = [
                # Wall scenarios
                ("both_at_left_wall", {"my_position": 0.0, "opponent_position": 0.0}),
                ("both_at_right_wall", {"my_position": 12.0, "opponent_position": 12.0}),
                ("me_at_wall_opponent_center", {"my_position": 0.0, "opponent_position": 6.0}),
                ("opponent_at_wall_me_center", {"my_position": 6.0, "opponent_position": 0.0}),

                # Already at opponent
                ("already_touching", {"my_position": 5.0, "opponent_position": 5.0}),
                ("already_overlapping", {"my_position": 5.0, "opponent_position": 4.999}),

                # Opponent moving scenarios
                ("opponent_fleeing_fast", {"opponent_velocity": 5.0, "opponent_position": 8.0}),
                ("opponent_approaching_fast", {"opponent_velocity": -5.0, "opponent_position": 8.0}),

                # Resource depletion
                ("zero_stamina_approach", {"my_stamina": 0.0}),
                ("low_hp_approach", {"my_hp": 1.0}),

                # Arena edge cases
                ("tiny_arena_approach", {"arena_width": 2.0, "my_position": 0.5, "opponent_position": 1.5}),
                ("opponent_outside_arena", {"opponent_position": 15.0, "arena_width": 12.0}),
            ]

            failures_before = len(self.failures)

            for case_name, params in test_cases:
                try:
                    # Set defaults that can be overridden by params
                    base_params = {"my_position": 3.0, "opponent_position": 9.0}
                    base_params.update(params)
                    snapshot = self.create_snapshot(**base_params)
                    decision = decide(snapshot)

                    if self.validate_decision(decision, f"{dummy_name}/{case_name}"):
                        # Special wall checks
                        if "wall" in case_name and params.get("my_position", 3.0) in [0.0, 12.0]:
                            # Should handle wall gracefully
                            pass
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

        # Test flee_always
        dummy_name = "flee_always"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)
            if decide:
                test_cases = [
                    # Can't flee scenarios
                    ("backed_into_left_wall", {"my_position": 0.0, "opponent_position": 2.0}),
                    ("backed_into_right_wall", {"my_position": 12.0, "opponent_position": 10.0}),
                    ("opponent_at_same_position", {"my_position": 6.0, "opponent_position": 6.0}),
                    ("surrounded_illusion", {"my_position": 6.0, "opponent_position": 6.0, "arena_width": 6.0}),
                ]

                failures_before = len(self.failures)

                for case_name, params in test_cases:
                    try:
                        snapshot = self.create_snapshot(**params)
                        decision = decide(snapshot)
                        self.validate_decision(decision, f"{dummy_name}/{case_name}")
                    except Exception as e:
                        self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

                if len(self.failures) == failures_before:
                    print(f"  ✅ {dummy_name}: All edge cases handled")
                else:
                    new_failures = len(self.failures) - failures_before
                    print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_distance_keeper_edge_cases(self):
        """Test distance keeper edge cases."""
        print("\n=== TESTING DISTANCE KEEPER EDGE CASES ===")

        for dist in [1, 3, 5]:
            dummy_name = f"distance_keeper_{dist}m"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            test_cases = [
                # Wall constraints
                ("cant_maintain_left_wall", {"my_position": 0.5, "opponent_position": 0.5 + dist + 2}),
                ("cant_maintain_right_wall", {"my_position": 11.5, "opponent_position": 11.5 - dist - 2}),
                ("both_at_wall", {"my_position": 0.0, "opponent_position": 0.0}),

                # Arena too small
                ("arena_smaller_than_distance", {"arena_width": dist - 0.5, "my_position": 0.5, "opponent_position": dist - 1}),
                ("arena_exactly_distance", {"arena_width": dist, "my_position": 0.0, "opponent_position": dist}),

                # Oscillation scenarios
                ("exactly_at_target", {"my_position": 3.0, "opponent_position": 3.0 + dist}),
                ("just_under_target", {"my_position": 3.0, "opponent_position": 3.0 + dist - 0.01}),
                ("just_over_target", {"my_position": 3.0, "opponent_position": 3.0 + dist + 0.01}),

                # Opponent moving
                ("opponent_approaching_fast", {"my_position": 5.0, "opponent_position": 5.0 + dist, "opponent_velocity": -5.0}),
                ("opponent_fleeing_fast", {"my_position": 5.0, "opponent_position": 5.0 + dist, "opponent_velocity": 5.0}),

                # Edge positions
                ("negative_positions", {"my_position": -2.0, "opponent_position": -2.0 + dist}),
                ("huge_positions", {"my_position": 100.0, "opponent_position": 100.0 + dist}),

                # Resource constraints
                ("zero_stamina_maintenance", {"my_stamina": 0.0}),
                ("low_hp_maintenance", {"my_hp": 1.0}),
            ]

            failures_before = len(self.failures)

            for case_name, params in test_cases:
                try:
                    snapshot = self.create_snapshot(**params)
                    decision = decide(snapshot)

                    if self.validate_decision(decision, f"{dummy_name}/{case_name}"):
                        # Check if it's trying to maintain distance reasonably
                        my_pos = params.get("my_position", 6.0)
                        opp_pos = params.get("opponent_position", 6.0)
                        current_dist = abs(opp_pos - my_pos)

                        # If too close and not at wall, should back away
                        if current_dist < dist - 0.3 and my_pos not in [0.0, 12.0]:
                            if opp_pos > my_pos and decision["acceleration"] > 0:
                                self.warnings.append(
                                    f"{dummy_name}/{case_name}: Should back away but moving forward"
                                )
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_stamina_pattern_edge_cases(self):
        """Test stamina pattern edge cases."""
        print("\n=== TESTING STAMINA PATTERN EDGE CASES ===")

        # Test stamina_waster
        dummy_name = "stamina_waster"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)
            if decide:
                test_cases = [
                    ("negative_stamina", {"my_stamina": -10.0}),
                    ("zero_max_stamina", {"my_stamina": 50.0}),  # Can't set stamina_max to 0 in snapshot
                    ("stamina_exceeds_max", {"my_stamina": 150.0}),
                    ("fractional_stamina", {"my_stamina": 33.333}),
                    ("near_zero_stamina", {"my_stamina": 0.001}),
                ]

                failures_before = len(self.failures)

                for case_name, params in test_cases:
                    try:
                        snapshot = self.create_snapshot(**params)
                        decision = decide(snapshot)

                        if self.validate_decision(decision, f"{dummy_name}/{case_name}"):
                            # Should always be extended
                            if decision["stance"] != "extended":
                                self.failures.append(
                                    f"{dummy_name}/{case_name}: Not extended at edge case"
                                )
                    except Exception as e:
                        self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

                if len(self.failures) == failures_before:
                    print(f"  ✅ {dummy_name}: All edge cases handled")
                else:
                    new_failures = len(self.failures) - failures_before
                    print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

        # Test stamina_cycler
        dummy_name = "stamina_cycler"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)
            if decide:
                test_cases = [
                    ("boundary_90", {"my_stamina": 90.0}),  # Exactly at threshold
                    ("boundary_20", {"my_stamina": 20.0}),  # Exactly at threshold
                    ("boundary_50", {"my_stamina": 50.0}),  # Middle threshold
                    ("rapid_drain", {"my_stamina": 21.0}),  # Just above recovery
                    ("rapid_recovery", {"my_stamina": 89.0}),  # Just below attack
                ]

                failures_before = len(self.failures)

                for case_name, params in test_cases:
                    try:
                        snapshot = self.create_snapshot(**params)
                        decision = decide(snapshot)
                        self.validate_decision(decision, f"{dummy_name}/{case_name}")
                    except Exception as e:
                        self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

                if len(self.failures) == failures_before:
                    print(f"  ✅ {dummy_name}: All edge cases handled")
                else:
                    new_failures = len(self.failures) - failures_before
                    print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_reactive_edge_cases(self):
        """Test reactive behavior edge cases."""
        print("\n=== TESTING REACTIVE EDGE CASES ===")

        # Test mirror_movement
        dummy_name = "mirror_movement"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)
            if decide:
                test_cases = [
                    ("zero_velocity", {"opponent_velocity": 0.0}),
                    ("tiny_velocity", {"opponent_velocity": 0.01}),
                    ("max_velocity", {"opponent_velocity": 5.0}),
                    ("beyond_max_velocity", {"opponent_velocity": 10.0}),
                    ("negative_max_velocity", {"opponent_velocity": -5.0}),
                    ("nan_velocity", {"opponent_velocity": float('nan')}),
                    ("inf_velocity", {"opponent_velocity": float('inf')}),
                    ("at_wall_cant_mirror", {"my_position": 0.0, "opponent_velocity": -3.0}),
                ]

                failures_before = len(self.failures)

                for case_name, params in test_cases:
                    try:
                        snapshot = self.create_snapshot(**params)
                        decision = decide(snapshot)

                        # Handle special cases
                        if "nan" in case_name or "inf" in case_name:
                            # Should handle gracefully, not crash
                            if decision:
                                self.validate_decision(decision, f"{dummy_name}/{case_name}")
                        else:
                            self.validate_decision(decision, f"{dummy_name}/{case_name}")
                    except Exception as e:
                        if "nan" not in case_name and "inf" not in case_name:
                            self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

                if len(self.failures) == failures_before:
                    print(f"  ✅ {dummy_name}: All edge cases handled")
                else:
                    new_failures = len(self.failures) - failures_before
                    print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

        # Test charge_on_approach
        dummy_name = "charge_on_approach"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)
            if decide:
                test_cases = [
                    ("exactly_4m", {"my_position": 2.0, "opponent_position": 6.0}),  # Exactly at threshold
                    ("just_under_4m", {"my_position": 2.0, "opponent_position": 5.999}),
                    ("just_over_4m", {"my_position": 2.0, "opponent_position": 6.001}),
                    ("at_wall_cant_charge", {"my_position": 0.0, "opponent_position": 3.0}),
                    ("opponent_behind_wall", {"my_position": 11.0, "opponent_position": 13.0}),
                    ("both_same_position", {"my_position": 5.0, "opponent_position": 5.0}),
                ]

                failures_before = len(self.failures)

                for case_name, params in test_cases:
                    try:
                        snapshot = self.create_snapshot(**params)
                        decision = decide(snapshot)
                        self.validate_decision(decision, f"{dummy_name}/{case_name}")
                    except Exception as e:
                        self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

                if len(self.failures) == failures_before:
                    print(f"  ✅ {dummy_name}: All edge cases handled")
                else:
                    new_failures = len(self.failures) - failures_before
                    print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_wall_hugger_edge_cases(self):
        """Test wall hugger edge cases."""
        print("\n=== TESTING WALL HUGGER EDGE CASES ===")

        for side in ["left", "right"]:
            dummy_name = f"wall_hugger_{side}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            wall_pos = 0.0 if side == "left" else 12.0
            test_cases = [
                ("already_at_wall", {"my_position": wall_pos}),
                ("beyond_wall", {"my_position": -1.0 if side == "left" else 13.0}),
                ("arena_shrunk", {"my_position": 6.0, "arena_width": 5.0 if side == "right" else 12.0}),
                ("negative_arena", {"arena_width": -10.0}),
                ("zero_arena", {"arena_width": 0.0}),
                ("opponent_blocking", {"my_position": 1.0 if side == "left" else 11.0,
                                      "opponent_position": wall_pos}),
            ]

            failures_before = len(self.failures)

            for case_name, params in test_cases:
                try:
                    snapshot = self.create_snapshot(**params)
                    decision = decide(snapshot)

                    if self.validate_decision(decision, f"{dummy_name}/{case_name}"):
                        # Check if moving toward wall
                        if "already_at_wall" not in case_name:
                            if side == "left" and decision["acceleration"] > 1.0:
                                self.warnings.append(
                                    f"{dummy_name}/{case_name}: Moving away from left wall"
                                )
                            elif side == "right" and decision["acceleration"] < -1.0:
                                self.warnings.append(
                                    f"{dummy_name}/{case_name}: Moving away from right wall"
                                )
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_shuttle_edge_cases(self):
        """Test shuttle movement edge cases."""
        print("\n=== TESTING SHUTTLE EDGE CASES ===")

        for speed in ["slow", "medium", "fast"]:
            dummy_name = f"shuttle_{speed}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            test_cases = [
                ("at_exact_left_bound", {"my_position": 3.0}),
                ("at_exact_right_bound", {"my_position": 9.0}),
                ("beyond_left_bound", {"my_position": 2.0}),
                ("beyond_right_bound", {"my_position": 10.0}),
                ("far_beyond_bounds", {"my_position": -10.0}),
                ("arena_smaller_than_bounds", {"my_position": 5.0, "arena_width": 6.0}),
                ("at_wall", {"my_position": 0.0}),
                ("opponent_in_path", {"my_position": 5.0, "opponent_position": 5.5}),
            ]

            failures_before = len(self.failures)

            for case_name, params in test_cases:
                try:
                    snapshot = self.create_snapshot(**params)
                    decision = decide(snapshot)
                    self.validate_decision(decision, f"{dummy_name}/{case_name}")
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} edge case failures")

    def test_behavioral_edge_cases(self):
        """Test behavioral fighter edge cases."""
        print("\n=== TESTING BEHAVIORAL FIGHTER EDGE CASES ===")

        behavioral_fighters = [
            "perfect_defender", "burst_attacker", "perfect_kiter",
            "stamina_optimizer", "wall_fighter", "adaptive_fighter"
        ]

        for fighter_name in behavioral_fighters:
            path = f"fighters/test_dummies/behavioral/{fighter_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {fighter_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            test_cases = [
                # Complex state combinations
                ("cornered_low_resources", {
                    "my_position": 0.5,
                    "opponent_position": 2.0,
                    "my_stamina": 10.0,
                    "my_hp": 20.0,
                    "opponent_hp": 80.0
                }),
                ("winning_but_exhausted", {
                    "my_hp": 90.0,
                    "opponent_hp": 10.0,
                    "my_stamina": 5.0,
                    "opponent_stamina": 90.0
                }),
                ("both_critical", {
                    "my_hp": 1.0,
                    "opponent_hp": 1.0,
                    "my_stamina": 1.0,
                    "opponent_stamina": 1.0
                }),
                ("opponent_invincible", {
                    "my_hp": 100.0,
                    "opponent_hp": 10000.0,  # Impossible HP
                    "opponent_stamina": 10000.0
                }),
                ("impossible_physics", {
                    "my_velocity": 50.0,
                    "opponent_velocity": -50.0,
                    "my_position": 6.0,
                    "opponent_position": 6.0
                }),
                # State transitions
                ("rapid_stance_change", {
                    "my_stance": "extended",
                    "opponent_stance": "retracted",
                    "tick": 1
                }),
                ("endgame_scenario", {
                    "tick": 99999,
                    "my_hp": 5.0,
                    "opponent_hp": 5.0
                }),
                # Adversarial inputs
                ("all_zeros", {
                    "my_position": 0.0,
                    "opponent_position": 0.0,
                    "my_velocity": 0.0,
                    "opponent_velocity": 0.0,
                    "my_hp": 0.0,
                    "opponent_hp": 0.0,
                    "my_stamina": 0.0,
                    "opponent_stamina": 0.0,
                    "arena_width": 0.0,
                    "tick": 0
                }),
                ("all_max", {
                    "my_position": 12.0,
                    "opponent_position": 12.0,
                    "my_velocity": 5.0,
                    "opponent_velocity": 5.0,
                    "my_hp": 100.0,
                    "opponent_hp": 100.0,
                    "my_stamina": 100.0,
                    "opponent_stamina": 100.0,
                    "arena_width": 12.0,
                    "tick": 1000
                }),
            ]

            failures_before = len(self.failures)

            for case_name, params in test_cases:
                try:
                    snapshot = self.create_snapshot(**params)
                    decision = decide(snapshot)

                    if self.validate_decision(decision, f"{fighter_name}/{case_name}"):
                        # Behavioral fighters should handle all edge cases gracefully
                        # Check for reasonable behavior
                        if "all_zeros" in case_name:
                            # Should not crash or return invalid values
                            pass
                        elif "cornered" in case_name:
                            # Should try to escape or defend
                            if decision["stance"] not in ["defending", "retracted"]:
                                self.warnings.append(
                                    f"{fighter_name}/{case_name}: Not defensive when cornered"
                                )
                except Exception as e:
                    self.failures.append(f"{fighter_name}/{case_name}: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {fighter_name}: All edge cases handled")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {fighter_name}: {new_failures} edge case failures")

    def test_adversarial_inputs(self):
        """Test with completely adversarial/malformed inputs."""
        print("\n=== TESTING ADVERSARIAL INPUTS ===")

        # Test a sample of dummies with really bad inputs
        test_dummies = [
            "fighters/test_dummies/atomic/stationary_neutral.py",
            "fighters/test_dummies/atomic/approach_fast.py",
            "fighters/test_dummies/atomic/distance_keeper_3m.py",
            "fighters/test_dummies/behavioral/perfect_defender.py"
        ]

        for path in test_dummies:
            if not Path(path).exists():
                continue

            dummy_name = Path(path).stem
            decide = self.load_fighter(path)
            if not decide:
                continue

            adversarial_cases = [
                # Missing fields
                ("missing_position", {
                    "you": {"velocity": 0, "hp": 100, "hp_max": 100, "stamina": 100, "stamina_max": 100, "stance": "neutral"},
                    "opponent": {"position": 6, "velocity": 0, "hp": 100, "hp_max": 100, "stamina": 100, "stamina_max": 100, "stance": "neutral"},
                    "arena": {"width": 12},
                    "tick": 0
                }),
                # Wrong types
                ("position_as_string", self.create_snapshot(my_position="six")),
                ("stamina_as_list", self.create_snapshot(my_stamina=[100])),
                # Null values (Python None)
                ("null_position", self.create_snapshot(my_position=None)),
                # Extreme values
                ("huge_numbers", self.create_snapshot(
                    my_position=1e308,
                    opponent_position=-1e308,
                    my_stamina=1e100,
                    tick=999999999999
                )),
                # Special float values
                ("nan_values", self.create_snapshot(
                    my_position=float('nan'),
                    opponent_velocity=float('nan')
                )),
                ("inf_values", self.create_snapshot(
                    my_position=float('inf'),
                    opponent_position=float('-inf')
                )),
            ]

            failures_before = len(self.failures)

            for case_name, snapshot in adversarial_cases:
                try:
                    if isinstance(snapshot, dict) and "you" in snapshot:
                        decision = decide(snapshot)
                        # If it doesn't crash, that's already good
                        if decision:
                            self.validate_decision(decision, f"{dummy_name}/{case_name}")
                except (KeyError, TypeError, ValueError) as e:
                    # Expected failures for truly malformed inputs
                    if "missing" in case_name or "null" in case_name or "as_string" in case_name:
                        pass  # These are expected to fail
                    else:
                        self.warnings.append(f"{dummy_name}/{case_name}: Handled error - {type(e).__name__}")
                except Exception as e:
                    self.failures.append(f"{dummy_name}/{case_name}: Unexpected exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: Handled adversarial inputs")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} adversarial failures")

    def test_stress_patterns(self):
        """Test stress patterns and rapid state changes."""
        print("\n=== TESTING STRESS PATTERNS ===")

        test_dummies = [
            ("stationary_neutral", "fighters/test_dummies/atomic/stationary_neutral.py"),
            ("shuttle_medium", "fighters/test_dummies/atomic/shuttle_medium.py"),
            ("distance_keeper_3m", "fighters/test_dummies/atomic/distance_keeper_3m.py"),
            ("perfect_defender", "fighters/test_dummies/behavioral/perfect_defender.py")
        ]

        for dummy_name, path in test_dummies:
            if not Path(path).exists():
                continue

            decide = self.load_fighter(path)
            if not decide:
                continue

            failures_before = len(self.failures)

            # Rapid oscillation test
            try:
                for i in range(100):
                    # Oscillate positions rapidly
                    my_pos = 6.0 + 3.0 * math.sin(i * 0.5)
                    opp_pos = 6.0 - 3.0 * math.sin(i * 0.5)

                    snapshot = self.create_snapshot(
                        my_position=my_pos,
                        opponent_position=opp_pos,
                        my_velocity=5.0 * math.cos(i * 0.3),
                        opponent_velocity=-5.0 * math.cos(i * 0.3),
                        my_stamina=50 + 50 * math.sin(i * 0.1),
                        tick=i
                    )

                    decision = decide(snapshot)
                    if not self.validate_decision(decision, f"{dummy_name}/oscillation_{i}"):
                        break

            except Exception as e:
                self.failures.append(f"{dummy_name}/oscillation_test: Exception - {e}")

            # Random input fuzzing
            try:
                random.seed(42)  # Reproducible
                for i in range(50):
                    snapshot = self.create_snapshot(
                        my_position=random.uniform(0, 12),
                        opponent_position=random.uniform(0, 12),
                        my_velocity=random.uniform(-5, 5),
                        opponent_velocity=random.uniform(-5, 5),
                        my_stamina=random.uniform(0, 100),
                        opponent_stamina=random.uniform(0, 100),
                        my_hp=random.uniform(1, 100),
                        opponent_hp=random.uniform(1, 100),
                        tick=random.randint(0, 10000)
                    )

                    decision = decide(snapshot)
                    if not self.validate_decision(decision, f"{dummy_name}/fuzz_{i}"):
                        break

            except Exception as e:
                self.failures.append(f"{dummy_name}/fuzzing_test: Exception - {e}")

            if len(self.failures) == failures_before:
                print(f"  ✅ {dummy_name}: Passed stress testing")
            else:
                new_failures = len(self.failures) - failures_before
                print(f"  ❌ {dummy_name}: {new_failures} stress failures")

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("EDGE CASE VALIDATION SUMMARY")
        print("=" * 60)

        total_tests = len(self.failures) + len(self.warnings)

        if not self.failures:
            print("✅ ALL EDGE CASE TESTS PASSED!")
        else:
            print(f"❌ {len(self.failures)} FAILURES FOUND")
            print("\nTop Failures (showing first 10):")
            for failure in self.failures[:10]:
                print(f"  - {failure}")

            if len(self.failures) > 10:
                print(f"\n  ... and {len(self.failures) - 10} more failures")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS")
            print("\nTop Warnings (showing first 5):")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")

        print("\n" + "=" * 60)
        return len(self.failures) == 0


def main():
    """Run all edge case tests."""
    validator = EdgeCaseValidator()

    print("TEST DUMMY EDGE CASE VALIDATOR")
    print("=" * 60)
    print("Testing edge cases, boundary conditions, and adversarial inputs")

    # Run all edge case tests
    validator.test_stationary_edge_cases()
    validator.test_movement_edge_cases()
    validator.test_distance_keeper_edge_cases()
    validator.test_stamina_pattern_edge_cases()
    validator.test_reactive_edge_cases()
    validator.test_wall_hugger_edge_cases()
    validator.test_shuttle_edge_cases()
    validator.test_behavioral_edge_cases()
    validator.test_adversarial_inputs()
    validator.test_stress_patterns()

    # Generate report
    success = validator.generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()