#!/usr/bin/env python3
"""
Test Suite for Atomic Test Dummies

Verifies that atomic test dummies behave as specified.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple


class AtomicDummyTester:
    def __init__(self):
        self.dummy_dir = Path("fighters/test_dummies/atomic")
        self.results = []

    def run_test(self, dummy_name: str, opponent: str = "fighters/training_opponents/punching_bag.py",
                 duration: int = 100) -> Dict:
        """Run a single test against a dummy."""
        dummy_path = self.dummy_dir / f"{dummy_name}.py"

        # Run match and save telemetry
        output_file = f"outputs/test_{dummy_name}.json"
        cmd = [
            "python", "atom_fight.py",
            str(dummy_path),
            opponent,
            "--seed", "42",
            "--max-ticks", str(duration),
            "--save", output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Load telemetry
        try:
            with open(output_file, 'r') as f:
                telemetry_data = json.load(f)
                telemetry = telemetry_data['telemetry']
        except:
            return {"error": "Failed to load telemetry"}

        # Analyze behavior
        return self.analyze_dummy_behavior(dummy_name, telemetry)

    def analyze_dummy_behavior(self, dummy_name: str, telemetry: Dict) -> Dict:
        """Analyze if dummy behaved as expected."""
        ticks = telemetry['ticks']
        analysis = {
            "dummy": dummy_name,
            "total_ticks": len(ticks),
            "tests_passed": [],
            "tests_failed": []
        }

        if "stationary" in dummy_name:
            # Test: Should not move
            positions = [t['fighter_a']['position'] for t in ticks]
            pos_variance = max(positions) - min(positions)

            if pos_variance < 0.1:
                analysis["tests_passed"].append("Remains stationary")
            else:
                analysis["tests_failed"].append(f"Movement detected: variance {pos_variance:.2f}")

            # Test: Correct stance
            if "neutral" in dummy_name:
                expected_stance = "neutral"
            elif "extended" in dummy_name:
                expected_stance = "extended"
            elif "defending" in dummy_name:
                expected_stance = "defending"
            elif "retracted" in dummy_name:
                expected_stance = "retracted"
            else:
                expected_stance = None

            if expected_stance:
                stances = [t['fighter_a']['stance'] for t in ticks]
                stance_correct = all(s == expected_stance for s in stances)

                if stance_correct:
                    analysis["tests_passed"].append(f"Maintains {expected_stance} stance")
                else:
                    unique_stances = set(stances)
                    analysis["tests_failed"].append(f"Wrong stances: {unique_stances}")

        elif "shuttle" in dummy_name:
            # Test: Should oscillate between bounds
            positions = [t['fighter_a']['position'] for t in ticks if t['tick'] > 10]
            if positions:
                min_pos = min(positions)
                max_pos = max(positions)

                if max_pos - min_pos > 3.0:
                    analysis["tests_passed"].append(f"Shuttles properly: range {max_pos - min_pos:.1f}m")
                else:
                    analysis["tests_failed"].append(f"Insufficient shuttling: range {max_pos - min_pos:.1f}m")

                # Test: Speed appropriate
                velocities = [abs(t['fighter_a']['velocity']) for t in ticks if t['tick'] > 10]
                avg_vel = sum(velocities) / len(velocities) if velocities else 0

                if "slow" in dummy_name and avg_vel < 1.5:
                    analysis["tests_passed"].append(f"Slow speed confirmed: {avg_vel:.1f}")
                elif "medium" in dummy_name and 1.5 <= avg_vel <= 3.0:
                    analysis["tests_passed"].append(f"Medium speed confirmed: {avg_vel:.1f}")
                elif "fast" in dummy_name and avg_vel > 2.5:
                    analysis["tests_passed"].append(f"Fast speed confirmed: {avg_vel:.1f}")
                else:
                    analysis["tests_failed"].append(f"Unexpected velocity: {avg_vel:.1f}")

        elif "approach" in dummy_name:
            # Test: Should move toward opponent
            distances = [t['fighter_a']['position'] - t['fighter_b']['position'] for t in ticks]
            initial_dist = abs(distances[0])
            final_dist = abs(distances[-1])

            if final_dist < initial_dist - 1.0:
                analysis["tests_passed"].append(f"Approaches opponent: {initial_dist:.1f}m → {final_dist:.1f}m")
            else:
                analysis["tests_failed"].append(f"Failed to approach: {initial_dist:.1f}m → {final_dist:.1f}m")

        elif "flee" in dummy_name:
            # Test: Should move away from opponent
            distances = [abs(t['fighter_a']['position'] - t['fighter_b']['position']) for t in ticks]
            avg_distance = sum(distances) / len(distances)

            if avg_distance > 4.0:
                analysis["tests_passed"].append(f"Maintains distance: avg {avg_distance:.1f}m")
            else:
                analysis["tests_failed"].append(f"Too close: avg {avg_distance:.1f}m")

        elif "circle" in dummy_name:
            # Test: Should have consistent directional movement
            velocities = [t['fighter_a']['velocity'] for t in ticks if t['tick'] > 5]

            if "left" in dummy_name:
                left_movement = sum(1 for v in velocities if v < -0.5)
                if left_movement > len(velocities) * 0.7:
                    analysis["tests_passed"].append("Circles left consistently")
                else:
                    analysis["tests_failed"].append(f"Inconsistent left movement: {left_movement}/{len(velocities)}")

            elif "right" in dummy_name:
                right_movement = sum(1 for v in velocities if v > 0.5)
                if right_movement > len(velocities) * 0.7:
                    analysis["tests_passed"].append("Circles right consistently")
                else:
                    analysis["tests_failed"].append(f"Inconsistent right movement: {right_movement}/{len(velocities)}")

        elif "wall_hugger" in dummy_name:
            # Test: Should be at wall
            positions = [t['fighter_a']['position'] for t in ticks if t['tick'] > 20]

            if "left" in dummy_name:
                wall_time = sum(1 for p in positions if p < 1.0)
                if wall_time > len(positions) * 0.8:
                    analysis["tests_passed"].append("Hugs left wall")
                else:
                    analysis["tests_failed"].append(f"Not at left wall: {wall_time}/{len(positions)}")

            elif "right" in dummy_name:
                wall_time = sum(1 for p in positions if p > 11.0)
                if wall_time > len(positions) * 0.8:
                    analysis["tests_passed"].append("Hugs right wall")
                else:
                    analysis["tests_failed"].append(f"Not at right wall: {wall_time}/{len(positions)}")

        return analysis

    def run_all_tests(self):
        """Run tests for all atomic dummies."""
        dummies = [
            "stationary_neutral",
            "stationary_extended",
            "stationary_defending",
            "stationary_retracted",
            "shuttle_slow",
            "shuttle_medium",
            "shuttle_fast",
            "approach_slow",
            "approach_fast",
            "flee_always",
            "circle_left",
            "circle_right",
            "wall_hugger_left",
            "wall_hugger_right"
        ]

        print("ATOMIC DUMMY TEST SUITE")
        print("=" * 60)

        for dummy in dummies:
            if not (self.dummy_dir / f"{dummy}.py").exists():
                print(f"\n❌ {dummy}: FILE NOT FOUND")
                continue

            print(f"\nTesting {dummy}...")
            result = self.run_test(dummy)

            if "error" in result:
                print(f"  ❌ ERROR: {result['error']}")
            else:
                print(f"  Total ticks: {result['total_ticks']}")

                if result["tests_passed"]:
                    print("  ✅ Passed:")
                    for test in result["tests_passed"]:
                        print(f"    - {test}")

                if result["tests_failed"]:
                    print("  ❌ Failed:")
                    for test in result["tests_failed"]:
                        print(f"    - {test}")

                # Overall result
                if result["tests_failed"]:
                    print(f"  RESULT: ❌ FAILED ({len(result['tests_failed'])} issues)")
                else:
                    print(f"  RESULT: ✅ PASSED")

            self.results.append(result)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        passed = sum(1 for r in self.results if not r.get("tests_failed") and "error" not in r)
        failed = len(self.results) - passed
        print(f"✅ Passed: {passed}/{len(self.results)}")
        print(f"❌ Failed: {failed}/{len(self.results)}")


if __name__ == "__main__":
    tester = AtomicDummyTester()
    tester.run_all_tests()