#!/usr/bin/env python3
"""
Test Dummy Validator

Unit tests for test dummy behaviors without running full fights.
Tests specific scenarios and validates expected responses.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class DummyValidator:
    """Validates test dummy behaviors against expected outputs."""

    def __init__(self):
        self.results = []
        self.failures = []

    def load_fighter(self, path: str):
        """Load a fighter module and return its decide function."""
        spec = importlib.util.spec_from_file_location("fighter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.decide

    def create_snapshot(self,
                       my_position: float = 6.0,
                       opponent_position: float = 6.0,
                       my_hp: float = 100.0,
                       opponent_hp: float = 100.0,
                       my_stamina: float = 100.0,
                       opponent_stamina: float = 100.0,
                       my_velocity: float = 0.0,
                       opponent_velocity: float = 0.0,
                       tick: int = 100,
                       arena_width: float = 12.0) -> Dict:
        """Create a test snapshot with specified parameters."""
        return {
            "you": {
                "position": my_position,
                "velocity": my_velocity,
                "hp": my_hp,
                "hp_max": 100.0,
                "stamina": my_stamina,
                "stamina_max": 100.0,
                "stance": "neutral"
            },
            "opponent": {
                "position": opponent_position,
                "velocity": opponent_velocity,
                "hp": opponent_hp,
                "hp_max": 100.0,
                "stamina": opponent_stamina,
                "stamina_max": 100.0,
                "stance": "neutral"
            },
            "arena": {
                "width": arena_width
            },
            "tick": tick
        }

    def validate_response(self, response: Dict, expected: Dict, test_name: str) -> bool:
        """Validate that response matches expected values."""
        passed = True
        errors = []

        # Check acceleration
        if "acceleration" in expected:
            if abs(response.get("acceleration", 0) - expected["acceleration"]) > 0.01:
                errors.append(f"Acceleration: expected {expected['acceleration']}, got {response.get('acceleration', 0)}")
                passed = False

        # Check stance
        if "stance" in expected:
            if response.get("stance") != expected["stance"]:
                errors.append(f"Stance: expected {expected['stance']}, got {response.get('stance')}")
                passed = False

        # Check acceleration bounds
        acc = response.get("acceleration", 0)
        if acc < -5.0 or acc > 5.0:
            errors.append(f"Acceleration out of bounds: {acc}")
            passed = False

        # Check stance validity
        valid_stances = ["neutral", "extended", "defending", "retracted"]
        if response.get("stance") not in valid_stances:
            errors.append(f"Invalid stance: {response.get('stance')}")
            passed = False

        if not passed:
            self.failures.append({
                "test": test_name,
                "errors": errors,
                "response": response,
                "expected": expected
            })

        return passed

    def test_stationary_dummies(self):
        """Test all stationary dummies maintain position and stance."""
        print("\n=== TESTING STATIONARY DUMMIES ===")

        stationary_tests = [
            ("stationary_neutral", {"acceleration": 0.0, "stance": "neutral"}),
            ("stationary_extended", {"acceleration": 0.0, "stance": "extended"}),
            ("stationary_defending", {"acceleration": 0.0, "stance": "defending"}),
            ("stationary_retracted", {"acceleration": 0.0, "stance": "retracted"})
        ]

        for dummy_name, expected in stationary_tests:
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Test various positions - should always be stationary
            test_cases = [
                ("center", self.create_snapshot(my_position=6.0)),
                ("left side", self.create_snapshot(my_position=2.0)),
                ("right side", self.create_snapshot(my_position=10.0)),
                ("opponent close", self.create_snapshot(opponent_position=5.0)),
                ("opponent far", self.create_snapshot(opponent_position=10.0))
            ]

            all_passed = True
            for case_name, snapshot in test_cases:
                response = decide(snapshot)
                if not self.validate_response(response, expected, f"{dummy_name}_{case_name}"):
                    all_passed = False

            if all_passed:
                print(f"  ✅ {dummy_name}: All tests passed")
            else:
                print(f"  ❌ {dummy_name}: Some tests failed")

    def test_movement_dummies(self):
        """Test movement dummies behave correctly."""
        print("\n=== TESTING MOVEMENT DUMMIES ===")

        # Test approach dummies
        for speed in ["slow", "fast"]:
            dummy_name = f"approach_{speed}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Test approaching from left
            snapshot = self.create_snapshot(my_position=3.0, opponent_position=9.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:
                print(f"  ✅ {dummy_name}: Approaches from left correctly")
            else:
                print(f"  ❌ {dummy_name}: Should move right when opponent is right")

            # Test approaching from right
            snapshot = self.create_snapshot(my_position=9.0, opponent_position=3.0)
            response = decide(snapshot)
            if response["acceleration"] < 0:
                print(f"  ✅ {dummy_name}: Approaches from right correctly")
            else:
                print(f"  ❌ {dummy_name}: Should move left when opponent is left")

        # Test flee_always
        dummy_name = "flee_always"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Test fleeing when opponent is right
            snapshot = self.create_snapshot(my_position=6.0, opponent_position=8.0)
            response = decide(snapshot)
            if response["acceleration"] < 0:
                print(f"  ✅ {dummy_name}: Flees left when opponent is right")
            else:
                print(f"  ❌ {dummy_name}: Should flee left, got {response['acceleration']}")

            # Test fleeing when opponent is left
            snapshot = self.create_snapshot(my_position=6.0, opponent_position=4.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:
                print(f"  ✅ {dummy_name}: Flees right when opponent is left")
            else:
                print(f"  ❌ {dummy_name}: Should flee right, got {response['acceleration']}")

    def test_distance_keepers(self):
        """Test distance keeper dummies maintain target distances."""
        print("\n=== TESTING DISTANCE KEEPERS ===")

        distances = [1, 3, 5]

        for dist in distances:
            dummy_name = f"distance_keeper_{dist}m"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Test at perfect distance - should stay still
            snapshot = self.create_snapshot(my_position=3.0, opponent_position=3.0 + dist)
            response = decide(snapshot)
            if abs(response["acceleration"]) < 1.0:
                print(f"  ✅ {dummy_name}: Maintains distance when at target")
            else:
                print(f"  ❌ {dummy_name}: Should stay still at target distance")

            # Test too close - should back away
            # Use dist - 0.5 to ensure we're closer than target but not at same position
            snapshot = self.create_snapshot(my_position=5.0, opponent_position=5.0 + dist - 0.5)
            response = decide(snapshot)
            if response["acceleration"] < 0:  # Opponent is right, so move left to increase distance
                print(f"  ✅ {dummy_name}: Backs away when too close")
            else:
                print(f"  ❌ {dummy_name}: Should back away when too close")

            # Test too far - should approach
            snapshot = self.create_snapshot(my_position=2.0, opponent_position=2.0 + dist + 2.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:  # Opponent is right, so move right to decrease distance
                print(f"  ✅ {dummy_name}: Approaches when too far")
            else:
                print(f"  ❌ {dummy_name}: Should approach when too far")

    def test_stamina_patterns(self):
        """Test stamina-based behavior changes."""
        print("\n=== TESTING STAMINA PATTERNS ===")

        # Test stamina_waster - should always be extended
        dummy_name = "stamina_waster"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Test at various stamina levels
            for stamina in [100, 50, 10, 0]:
                snapshot = self.create_snapshot(my_stamina=stamina)
                response = decide(snapshot)
                if response["stance"] == "extended":
                    print(f"  ✅ {dummy_name}: Extended at {stamina}% stamina")
                else:
                    print(f"  ❌ {dummy_name}: Should be extended, got {response['stance']}")

        # Test stamina_cycler - should change based on stamina
        dummy_name = "stamina_cycler"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # High stamina - should attack
            snapshot = self.create_snapshot(my_stamina=95)
            response = decide(snapshot)
            if response["stance"] == "extended":
                print(f"  ✅ {dummy_name}: Attacks at high stamina")
            else:
                print(f"  ❌ {dummy_name}: Should attack at 95% stamina")

            # Low stamina - should recover
            snapshot = self.create_snapshot(my_stamina=15)
            response = decide(snapshot)
            if response["stance"] == "retracted":
                print(f"  ✅ {dummy_name}: Recovers at low stamina")
            else:
                print(f"  ❌ {dummy_name}: Should recover at 15% stamina")

    def test_reactive_dummies(self):
        """Test reactive behavior dummies."""
        print("\n=== TESTING REACTIVE DUMMIES ===")

        # Test mirror_movement
        dummy_name = "mirror_movement"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Opponent moving right - should move right
            snapshot = self.create_snapshot(opponent_velocity=2.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:
                print(f"  ✅ {dummy_name}: Mirrors rightward movement")
            else:
                print(f"  ❌ {dummy_name}: Should move right when opponent moves right")

            # Opponent moving left - should move left
            snapshot = self.create_snapshot(opponent_velocity=-2.0)
            response = decide(snapshot)
            if response["acceleration"] < 0:
                print(f"  ✅ {dummy_name}: Mirrors leftward movement")
            else:
                print(f"  ❌ {dummy_name}: Should move left when opponent moves left")

        # Test counter_movement
        dummy_name = "counter_movement"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Opponent moving right - should move left
            snapshot = self.create_snapshot(opponent_velocity=2.0)
            response = decide(snapshot)
            if response["acceleration"] < 0:
                print(f"  ✅ {dummy_name}: Counters rightward movement")
            else:
                print(f"  ❌ {dummy_name}: Should move left when opponent moves right")

            # Opponent moving left - should move right
            snapshot = self.create_snapshot(opponent_velocity=-2.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:
                print(f"  ✅ {dummy_name}: Counters leftward movement")
            else:
                print(f"  ❌ {dummy_name}: Should move right when opponent moves left")

        # Test charge_on_approach
        dummy_name = "charge_on_approach"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Opponent far - should wait
            snapshot = self.create_snapshot(my_position=2.0, opponent_position=8.0)
            response = decide(snapshot)
            if abs(response["acceleration"]) < 0.1 and response["stance"] == "neutral":
                print(f"  ✅ {dummy_name}: Waits when opponent is far")
            else:
                print(f"  ❌ {dummy_name}: Should wait when opponent is far")

            # Opponent close - should charge
            snapshot = self.create_snapshot(my_position=5.0, opponent_position=7.5)
            response = decide(snapshot)
            if response["acceleration"] > 3.0 and response["stance"] == "extended":
                print(f"  ✅ {dummy_name}: Charges when opponent is close")
            else:
                print(f"  ❌ {dummy_name}: Should charge when opponent < 4m")

    def test_wall_behavior(self):
        """Test wall-related behaviors."""
        print("\n=== TESTING WALL BEHAVIORS ===")

        # Test wall huggers
        for side in ["left", "right"]:
            dummy_name = f"wall_hugger_{side}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            if side == "left":
                # At wall - should stay
                snapshot = self.create_snapshot(my_position=0.3)
                response = decide(snapshot)
                if response["acceleration"] <= 0:
                    print(f"  ✅ {dummy_name}: Stays at left wall")
                else:
                    print(f"  ❌ {dummy_name}: Should stay at left wall")

                # Away from wall - should move to wall
                snapshot = self.create_snapshot(my_position=6.0)
                response = decide(snapshot)
                if response["acceleration"] < -2.0:
                    print(f"  ✅ {dummy_name}: Moves to left wall from center")
                else:
                    print(f"  ❌ {dummy_name}: Should move left to wall")
            else:
                # At wall - should stay
                snapshot = self.create_snapshot(my_position=11.7)
                response = decide(snapshot)
                if response["acceleration"] >= 0:
                    print(f"  ✅ {dummy_name}: Stays at right wall")
                else:
                    print(f"  ❌ {dummy_name}: Should stay at right wall")

                # Away from wall - should move to wall
                snapshot = self.create_snapshot(my_position=6.0)
                response = decide(snapshot)
                if response["acceleration"] > 2.0:
                    print(f"  ✅ {dummy_name}: Moves to right wall from center")
                else:
                    print(f"  ❌ {dummy_name}: Should move right to wall")

    def test_shuttle_patterns(self):
        """Test shuttle movement patterns."""
        print("\n=== TESTING SHUTTLE PATTERNS ===")

        speeds = ["slow", "medium", "fast"]
        expected_speeds = {"slow": 1.0, "medium": 2.5, "fast": 4.0}

        for speed in speeds:
            dummy_name = f"shuttle_{speed}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Test at left bound - should move right
            snapshot = self.create_snapshot(my_position=3.0)
            response = decide(snapshot)
            if response["acceleration"] > 0:
                print(f"  ✅ {dummy_name}: Moves right from left bound")
            else:
                print(f"  ❌ {dummy_name}: Should move right from position 3.0")

            # Test at right bound - should move left
            snapshot = self.create_snapshot(my_position=9.0)
            response = decide(snapshot)
            if response["acceleration"] < 0:
                print(f"  ✅ {dummy_name}: Moves left from right bound")
            else:
                print(f"  ❌ {dummy_name}: Should move left from position 9.0")

    def generate_report(self):
        """Generate a summary report."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if not self.failures:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"❌ {len(self.failures)} TESTS FAILED")
            print("\nFailed Tests:")
            for failure in self.failures[:10]:  # Show first 10 failures
                print(f"\n  Test: {failure['test']}")
                for error in failure['errors']:
                    print(f"    - {error}")
                print(f"    Response: {failure['response']}")
                print(f"    Expected: {failure['expected']}")

            if len(self.failures) > 10:
                print(f"\n  ... and {len(self.failures) - 10} more failures")

        print("\n" + "=" * 60)
        return len(self.failures) == 0


def main():
    """Run all validation tests."""
    validator = DummyValidator()

    print("TEST DUMMY VALIDATOR")
    print("=" * 60)
    print("Validating test dummy behaviors without running full fights")

    # Run all test categories
    validator.test_stationary_dummies()
    validator.test_movement_dummies()
    validator.test_distance_keepers()
    validator.test_stamina_patterns()
    validator.test_reactive_dummies()
    validator.test_wall_behavior()
    validator.test_shuttle_patterns()

    # Generate report
    success = validator.generate_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()