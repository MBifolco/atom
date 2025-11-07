#!/usr/bin/env python3
"""
Regression Detector

Compares fighter performance across test runs to detect regressions.
Maintains baseline performance metrics and alerts on degradation.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import statistics


class RegressionDetector:
    def __init__(self, baseline_dir: str = None):
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.baseline_data = {}
        self.thresholds = {
            'win_rate_drop': 10.0,      # Alert if win rate drops by 10%
            'hp_diff_drop': 20.0,       # Alert if HP differential drops by 20
            'collision_drop': 0.5,       # Alert if collisions drop by 50%
            'wall_time_increase': 20.0, # Alert if wall time increases by 20%
        }

    def load_baseline(self, baseline_path: str = None):
        """Load baseline performance data."""
        path = Path(baseline_path) if baseline_path else self.find_latest_baseline()

        if not path or not path.exists():
            print(f"⚠️ No baseline found. First run will establish baseline.")
            return False

        # Load from CSV
        csv_path = path / "test_results.csv" if path.is_dir() else path

        if not csv_path.exists():
            print(f"⚠️ Baseline CSV not found at {csv_path}")
            return False

        self.baseline_data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['fighter']}_{row['category']}_{row['dummy']}"
                self.baseline_data[key] = {
                    'win_rate': float(row['win_rate']),
                    'hp_differential': float(row['hp_differential']),
                    'collisions': float(row['collisions']),
                    'avg_distance': float(row['avg_distance']),
                    'wall_time_pct': float(row['wall_time_pct'])
                }

        print(f"✅ Loaded baseline with {len(self.baseline_data)} test results")
        return True

    def find_latest_baseline(self) -> Optional[Path]:
        """Find the most recent test matrix output."""
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            return None

        test_dirs = sorted([
            d for d in outputs_dir.glob("test_matrix_*")
            if d.is_dir()
        ], reverse=True)

        return test_dirs[0] if test_dirs else None

    def compare_with_baseline(self, current_results_path: str) -> Dict:
        """Compare current results with baseline."""
        if not self.baseline_data:
            print("⚠️ No baseline loaded. Skipping comparison.")
            return {"status": "no_baseline"}

        current_path = Path(current_results_path)
        csv_path = current_path / "test_results.csv" if current_path.is_dir() else current_path

        if not csv_path.exists():
            return {"status": "error", "message": f"Current results not found at {csv_path}"}

        # Load current results
        current_data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['fighter']}_{row['category']}_{row['dummy']}"
                current_data[key] = {
                    'win_rate': float(row['win_rate']),
                    'hp_differential': float(row['hp_differential']),
                    'collisions': float(row['collisions']),
                    'avg_distance': float(row['avg_distance']),
                    'wall_time_pct': float(row['wall_time_pct'])
                }

        # Compare and detect regressions
        regressions = []
        improvements = []
        unchanged = []

        for key, baseline in self.baseline_data.items():
            if key not in current_data:
                regressions.append({
                    'test': key,
                    'issue': 'Test missing in current results'
                })
                continue

            current = current_data[key]
            regression_found = False

            # Check win rate
            win_rate_delta = current['win_rate'] - baseline['win_rate']
            if win_rate_delta < -self.thresholds['win_rate_drop']:
                regressions.append({
                    'test': key,
                    'metric': 'win_rate',
                    'baseline': baseline['win_rate'],
                    'current': current['win_rate'],
                    'delta': win_rate_delta,
                    'severity': 'high' if win_rate_delta < -20 else 'medium'
                })
                regression_found = True

            # Check HP differential
            hp_delta = current['hp_differential'] - baseline['hp_differential']
            if hp_delta < -self.thresholds['hp_diff_drop']:
                regressions.append({
                    'test': key,
                    'metric': 'hp_differential',
                    'baseline': baseline['hp_differential'],
                    'current': current['hp_differential'],
                    'delta': hp_delta,
                    'severity': 'medium'
                })
                regression_found = True

            # Check collisions (for aggressive fighters)
            if baseline['collisions'] > 10:  # Only check if baseline had meaningful collisions
                collision_ratio = current['collisions'] / baseline['collisions'] if baseline['collisions'] > 0 else 1
                if collision_ratio < self.thresholds['collision_drop']:
                    regressions.append({
                        'test': key,
                        'metric': 'collisions',
                        'baseline': baseline['collisions'],
                        'current': current['collisions'],
                        'delta': current['collisions'] - baseline['collisions'],
                        'severity': 'low'
                    })
                    regression_found = True

            # Check wall time
            wall_delta = current['wall_time_pct'] - baseline['wall_time_pct']
            if wall_delta > self.thresholds['wall_time_increase']:
                regressions.append({
                    'test': key,
                    'metric': 'wall_time',
                    'baseline': baseline['wall_time_pct'],
                    'current': current['wall_time_pct'],
                    'delta': wall_delta,
                    'severity': 'high'  # Wall grinding is critical
                })
                regression_found = True

            # Check for improvements
            if not regression_found:
                if win_rate_delta > 10 or hp_delta > 20:
                    improvements.append({
                        'test': key,
                        'win_rate_improvement': win_rate_delta if win_rate_delta > 10 else None,
                        'hp_improvement': hp_delta if hp_delta > 20 else None
                    })
                else:
                    unchanged.append(key)

        return {
            'status': 'complete',
            'regressions': regressions,
            'improvements': improvements,
            'unchanged': unchanged,
            'total_tests': len(self.baseline_data),
            'tests_run': len(current_data)
        }

    def generate_regression_report(self, comparison: Dict, output_path: str = None):
        """Generate a detailed regression report."""
        if comparison['status'] != 'complete':
            print(f"⚠️ Cannot generate report: {comparison.get('message', 'Comparison incomplete')}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_path) if output_path else Path(f"outputs/regression_report_{timestamp}.md")

        with open(report_path, 'w') as f:
            f.write("# Regression Detection Report\n")
            f.write(f"Generated: {timestamp}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests**: {comparison['total_tests']}\n")
            f.write(f"- **Tests Run**: {comparison['tests_run']}\n")
            f.write(f"- **Regressions Found**: {len(comparison['regressions'])}\n")
            f.write(f"- **Improvements Found**: {len(comparison['improvements'])}\n")
            f.write(f"- **Unchanged**: {len(comparison['unchanged'])}\n\n")

            # Critical regressions
            high_severity = [r for r in comparison['regressions'] if r.get('severity') == 'high']
            if high_severity:
                f.write("## 🚨 CRITICAL REGRESSIONS\n\n")
                for reg in high_severity:
                    f.write(f"### {reg['test']}\n")
                    f.write(f"- **Metric**: {reg.get('metric', 'unknown')}\n")
                    f.write(f"- **Baseline**: {reg.get('baseline', 'N/A'):.1f}\n")
                    f.write(f"- **Current**: {reg.get('current', 'N/A'):.1f}\n")
                    f.write(f"- **Change**: {reg.get('delta', 0):.1f}\n\n")

            # All regressions
            if comparison['regressions']:
                f.write("## All Regressions\n\n")
                f.write("| Test | Metric | Baseline | Current | Delta | Severity |\n")
                f.write("|------|--------|----------|---------|-------|----------|\n")

                for reg in sorted(comparison['regressions'], key=lambda x: x.get('severity', 'low'), reverse=True):
                    if 'metric' in reg:
                        f.write(f"| {reg['test']} | {reg['metric']} | ")
                        f.write(f"{reg.get('baseline', 0):.1f} | {reg.get('current', 0):.1f} | ")
                        f.write(f"{reg.get('delta', 0):+.1f} | {reg.get('severity', 'unknown')} |\n")
                    else:
                        f.write(f"| {reg['test']} | - | - | - | - | {reg.get('issue', 'unknown')} |\n")

                f.write("\n")

            # Improvements
            if comparison['improvements']:
                f.write("## ✅ Improvements\n\n")
                for imp in comparison['improvements']:
                    f.write(f"- **{imp['test']}**: ")
                    improvements = []
                    if imp.get('win_rate_improvement'):
                        improvements.append(f"Win rate +{imp['win_rate_improvement']:.1f}%")
                    if imp.get('hp_improvement'):
                        improvements.append(f"HP diff +{imp['hp_improvement']:.1f}")
                    f.write(", ".join(improvements) + "\n")
                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if high_severity:
                f.write("### Immediate Actions Required:\n\n")

                wall_issues = [r for r in high_severity if r.get('metric') == 'wall_time']
                if wall_issues:
                    f.write("1. **Wall Grinding Detected** - Review movement and wall detection logic\n")

                win_issues = [r for r in high_severity if r.get('metric') == 'win_rate']
                if win_issues:
                    f.write("2. **Significant Performance Drop** - Review recent fighter behavior changes\n")

                f.write("\n")

            f.write("### Testing Recommendations:\n\n")
            f.write("1. Re-run failed tests with detailed logging\n")
            f.write("2. Compare telemetry data between baseline and current\n")
            f.write("3. Review recent commits for behavior changes\n")
            f.write("4. Consider updating baseline if improvements are intentional\n\n")

        print(f"\n{'='*60}")
        if comparison['regressions']:
            print(f"🚨 REGRESSIONS DETECTED: {len(comparison['regressions'])} issues found")
            print(f"   High severity: {len(high_severity)}")
        else:
            print(f"✅ NO REGRESSIONS DETECTED")

        if comparison['improvements']:
            print(f"✅ Improvements found: {len(comparison['improvements'])}")

        print(f"\nReport saved to: {report_path}")
        print(f"{'='*60}\n")

        return report_path

    def update_baseline(self, new_baseline_path: str):
        """Update the baseline with new results."""
        new_path = Path(new_baseline_path)

        if not new_path.exists():
            print(f"❌ Cannot update baseline: {new_path} not found")
            return False

        # Create baseline directory if needed
        baseline_dir = Path("outputs/baseline")
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Copy the CSV file
        import shutil
        csv_source = new_path / "test_results.csv" if new_path.is_dir() else new_path
        csv_dest = baseline_dir / "test_results.csv"

        shutil.copy2(csv_source, csv_dest)

        # Save metadata
        metadata = {
            'updated': datetime.now().isoformat(),
            'source': str(new_path),
            'tests': len(self.baseline_data)
        }

        with open(baseline_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Baseline updated from {new_path}")
        return True


def main():
    """Run regression detection."""
    import sys

    detector = RegressionDetector()

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--update-baseline" and len(sys.argv) > 2:
            # Update baseline mode
            detector.update_baseline(sys.argv[2])
            return

        # Compare mode with specific path
        current_results = sys.argv[1]
    else:
        # Find latest test results
        current_results = detector.find_latest_baseline()
        if not current_results:
            print("❌ No test results found. Run test_matrix_runner.py first.")
            return

    # Load baseline
    detector.load_baseline()

    # Run comparison
    comparison = detector.compare_with_baseline(current_results)

    # Generate report
    if comparison['status'] == 'complete':
        detector.generate_regression_report(comparison)

        # Exit with error code if regressions found
        if comparison['regressions']:
            sys.exit(1)


if __name__ == "__main__":
    main()