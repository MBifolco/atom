#!/usr/bin/env python3
"""
Comprehensive Test Dummy Validation Suite

Runs all validation tests and generates a complete report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class ComprehensiveTestRunner:
    """Runs all test dummy validation suites."""

    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()

    def run_test(self, test_name: str, test_file: str) -> dict:
        """Run a single test suite and capture results."""
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                ["python", test_file],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output for results
            output = result.stdout
            passed = "ALL TESTS PASSED" in output or "ALL SEQUENCE TESTS PASSED" in output or "ALL EDGE CASE TESTS PASSED" in output

            # Count tests
            test_count = 0
            failure_count = 0
            warning_count = 0

            if "✅" in output:
                test_count = output.count("✅")
            if "❌" in output:
                failure_count = output.count("❌")
            if "⚠️" in output:
                warning_count = output.count("⚠️")

            return {
                "status": "PASSED" if passed else "FAILED",
                "return_code": result.returncode,
                "tests": test_count,
                "failures": failure_count,
                "warnings": warning_count,
                "output": output[-1000:] if not passed else ""  # Keep last 1000 chars if failed
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "return_code": -1,
                "tests": 0,
                "failures": 0,
                "warnings": 0,
                "output": "Test timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "return_code": -1,
                "tests": 0,
                "failures": 0,
                "warnings": 0,
                "output": str(e)
            }

    def run_all_tests(self):
        """Run all test suites."""
        test_suites = [
            ("Snapshot Validation", "test_dummy_validator.py"),
            ("Sequence Validation", "test_dummy_sequence_validator.py"),
            ("Edge Case Validation", "test_dummy_edge_cases.py"),
        ]

        # Optional: Include difficulty analyzer if desired
        # test_suites.append(("Difficulty Analysis", "test_dummy_difficulty_analyzer.py"))

        for test_name, test_file in test_suites:
            if Path(test_file).exists():
                self.results[test_name] = self.run_test(test_name, test_file)
            else:
                self.results[test_name] = {
                    "status": "NOT FOUND",
                    "return_code": -1,
                    "tests": 0,
                    "failures": 0,
                    "warnings": 0,
                    "output": f"Test file {test_file} not found"
                }

    def generate_report(self):
        """Generate comprehensive test report."""
        elapsed = datetime.now() - self.start_time

        print("\n" + "="*80)
        print("COMPREHENSIVE TEST DUMMY VALIDATION REPORT")
        print("="*80)
        print(f"Test Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {elapsed.total_seconds():.1f} seconds")
        print()

        # Summary table
        print("TEST SUITE SUMMARY")
        print("-"*80)
        print(f"{'Test Suite':<25} {'Status':<10} {'Tests':<10} {'Failures':<10} {'Warnings':<10}")
        print("-"*80)

        total_tests = 0
        total_failures = 0
        total_warnings = 0
        all_passed = True

        for test_name, result in self.results.items():
            status_symbol = {
                "PASSED": "✅",
                "FAILED": "❌",
                "TIMEOUT": "⏱️",
                "ERROR": "💥",
                "NOT FOUND": "❓"
            }.get(result["status"], "?")

            print(f"{test_name:<25} {status_symbol} {result['status']:<8} {result['tests']:<10} {result['failures']:<10} {result['warnings']:<10}")

            total_tests += result["tests"]
            total_failures += result["failures"]
            total_warnings += result["warnings"]

            if result["status"] != "PASSED":
                all_passed = False

        print("-"*80)
        print(f"{'TOTAL':<25} {'':10} {total_tests:<10} {total_failures:<10} {total_warnings:<10}")
        print()

        # Overall status
        if all_passed:
            print("🎉 OVERALL STATUS: ALL VALIDATION SUITES PASSED! 🎉")
            print()
            print("Test dummies are fully validated and production-ready.")
        else:
            print("❌ OVERALL STATUS: SOME TESTS FAILED")
            print()
            print("Failed tests:")
            for test_name, result in self.results.items():
                if result["status"] != "PASSED":
                    print(f"  - {test_name}: {result['status']}")
                    if result["output"]:
                        print(f"    Last output: ...{result['output'][-200:]}")

        # Test coverage analysis
        print("\n" + "="*80)
        print("TEST COVERAGE ANALYSIS")
        print("-"*80)

        dummy_categories = {
            "Atomic": [
                "stationary_*", "approach_*", "flee_*", "shuttle_*",
                "distance_keeper_*", "stamina_*", "wall_hugger_*",
                "mirror_*", "counter_*", "charge_*", "circle_*"
            ],
            "Behavioral": [
                "perfect_defender", "burst_attacker", "perfect_kiter",
                "stamina_optimizer", "wall_fighter", "adaptive_fighter"
            ]
        }

        for category, patterns in dummy_categories.items():
            print(f"\n{category} Dummies:")
            atomic_dir = Path(f"fighters/test_dummies/{category.lower()}")
            if atomic_dir.exists():
                dummy_count = len(list(atomic_dir.glob("*.py")))
                print(f"  Files found: {dummy_count}")
                print(f"  Expected patterns: {', '.join(patterns[:5])}...")
            else:
                print(f"  Directory not found: {atomic_dir}")

        # Save report to file
        report_file = f"test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")

        return all_passed

    def run_quick_tests(self):
        """Run only the fast tests (skip difficulty analyzer)."""
        print("Running quick validation tests (snapshot, sequence, edge cases)...")
        self.run_all_tests()
        return self.generate_report()


def main():
    """Run comprehensive test suite."""
    print("COMPREHENSIVE TEST DUMMY VALIDATION")
    print("="*80)
    print("This will run all validation tests for test dummies.")
    print("Expected duration: ~10-15 seconds")
    print()

    runner = ComprehensiveTestRunner()

    # Run all tests
    all_passed = runner.run_quick_tests()

    # Exit with appropriate code
    if all_passed:
        print("\n✅ All test dummies validated successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some validation tests failed. Please review and fix.")
        sys.exit(1)


if __name__ == "__main__":
    main()