#!/usr/bin/env python3
"""
Test Matrix Runner

Automated testing system that runs fighters against all test dummies
and generates comprehensive performance reports.
"""

import subprocess
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import statistics


class TestMatrixRunner:
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/test_matrix_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_all_test_dummies(self) -> Dict[str, List[str]]:
        """Get all test dummies organized by category."""
        dummies = {
            "atomic": [],
            "behavioral": [],
            "scenario": []
        }

        for category in dummies.keys():
            dummy_dir = Path(f"fighters/test_dummies/{category}")
            if dummy_dir.exists():
                dummies[category] = sorted([
                    f.stem for f in dummy_dir.glob("*.py")
                    if not f.name.startswith("__")
                ])

        return dummies

    def run_match(self, fighter_path: str, dummy_path: str,
                  seed: int = 42, max_ticks: int = 1000) -> Dict:
        """Run a single match and collect results."""
        output_file = self.output_dir / f"match_{Path(fighter_path).stem}_vs_{Path(dummy_path).stem}_seed{seed}.json"

        cmd = [
            "python", "atom_fight.py",
            fighter_path,
            dummy_path,
            "--seed", str(seed),
            "--max-ticks", str(max_ticks),
            "--save", str(output_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Load telemetry data
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    telemetry = data.get('telemetry', {})
                    result_data = data.get('result', {})

                    # Get winner from result data
                    winner_str = result_data.get('winner', '')
                    if 'draw' in winner_str.lower():
                        winner = 'draw'
                    elif telemetry.get('fighter_a_name', '') in winner_str:
                        winner = 'fighter_a'
                    elif telemetry.get('fighter_b_name', '') in winner_str:
                        winner = 'fighter_b'
                    else:
                        winner = None

                    # Extract key metrics
                    final_tick = telemetry['ticks'][-1] if telemetry.get('ticks') else {}

                    metrics = {
                        "winner": winner,
                        "duration": len(telemetry.get('ticks', [])),
                        "fighter_final_hp": final_tick.get('fighter_a', {}).get('hp', 0),
                        "dummy_final_hp": final_tick.get('fighter_b', {}).get('hp', 0),
                        "total_collisions": sum(
                            1 for t in telemetry.get('ticks', [])
                            if t.get('collision', {}).get('occurred', False)
                        )
                    }

                    # Analyze behavior patterns
                    metrics.update(self.analyze_behavior(telemetry))

                    return metrics

            else:
                return {"error": "Output file not created"}

        except subprocess.TimeoutExpired:
            return {"error": "Match timeout"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def analyze_behavior(self, telemetry: Dict) -> Dict:
        """Analyze fighter behavior from telemetry."""
        ticks = telemetry.get('ticks', [])
        if not ticks:
            return {}

        analysis = {}

        # Movement analysis
        positions_a = [t['fighter_a']['position'] for t in ticks]
        positions_b = [t['fighter_b']['position'] for t in ticks]

        analysis['fighter_avg_position'] = statistics.mean(positions_a)
        analysis['fighter_position_variance'] = statistics.variance(positions_a) if len(positions_a) > 1 else 0
        analysis['dummy_avg_position'] = statistics.mean(positions_b)
        analysis['dummy_position_variance'] = statistics.variance(positions_b) if len(positions_b) > 1 else 0

        # Distance management
        distances = [abs(t['fighter_a']['position'] - t['fighter_b']['position']) for t in ticks]
        analysis['avg_distance'] = statistics.mean(distances)
        analysis['min_distance'] = min(distances)
        analysis['max_distance'] = max(distances)

        # Stance usage
        stances_a = [t['fighter_a']['stance'] for t in ticks]
        analysis['fighter_extended_pct'] = stances_a.count('extended') / len(stances_a) * 100
        analysis['fighter_defending_pct'] = stances_a.count('defending') / len(stances_a) * 100
        analysis['fighter_neutral_pct'] = stances_a.count('neutral') / len(stances_a) * 100
        analysis['fighter_retracted_pct'] = stances_a.count('retracted') / len(stances_a) * 100

        # Wall time
        wall_threshold = 1.5
        arena_width = ticks[0]['arena']['width'] if ticks else 12.0

        fighter_wall_time = sum(
            1 for t in ticks
            if t['fighter_a']['position'] < wall_threshold or
               t['fighter_a']['position'] > arena_width - wall_threshold
        )
        analysis['fighter_wall_time_pct'] = fighter_wall_time / len(ticks) * 100

        # Stamina efficiency
        stamina_usage_a = []
        for i in range(1, len(ticks)):
            prev_stamina = ticks[i-1]['fighter_a']['stamina']
            curr_stamina = ticks[i]['fighter_a']['stamina']
            stamina_usage_a.append(prev_stamina - curr_stamina)

        if stamina_usage_a:
            analysis['fighter_avg_stamina_drain'] = statistics.mean([s for s in stamina_usage_a if s > 0] or [0])

        return analysis

    def run_test_matrix(self, fighter_paths: List[str], categories: List[str] = None):
        """Run fighters against all test dummies."""
        dummies = self.get_all_test_dummies()

        if categories is None:
            categories = list(dummies.keys())

        print(f"\n{'='*80}")
        print(f"TEST MATRIX RUNNER - {self.timestamp}")
        print(f"{'='*80}\n")

        for fighter_path in fighter_paths:
            fighter_name = Path(fighter_path).stem
            print(f"\nTesting: {fighter_name}")
            print("-" * 40)

            fighter_results = {
                "fighter": fighter_name,
                "results": {}
            }

            for category in categories:
                if category not in dummies:
                    continue

                print(f"\n  Category: {category.upper()}")
                category_results = {}

                for dummy in dummies[category]:
                    dummy_path = f"fighters/test_dummies/{category}/{dummy}.py"

                    # Run with multiple seeds for consistency
                    match_results = []
                    for seed in [42, 123, 999]:
                        result = self.run_match(fighter_path, dummy_path, seed=seed)
                        match_results.append(result)

                    # Aggregate results
                    aggregated = self.aggregate_results(match_results)
                    category_results[dummy] = aggregated

                    # Print summary
                    if "error" not in aggregated:
                        win_rate = aggregated.get('win_rate', 0)
                        avg_hp_diff = aggregated.get('avg_hp_differential', 0)
                        # More reasonable thresholds: 40% for pass (includes draws), 20% for warning
                        symbol = "✅" if win_rate >= 40 else "⚠️" if win_rate >= 20 else "❌"

                        print(f"    {symbol} {dummy:30} Win: {win_rate:5.1f}% HP±: {avg_hp_diff:+6.1f}")
                    else:
                        print(f"    ❌ {dummy:30} ERROR: {aggregated['error']}")

                fighter_results["results"][category] = category_results

            self.results.append(fighter_results)

        # Generate reports
        self.generate_summary_report()
        self.generate_detailed_report()
        self.generate_csv_export()

    def aggregate_results(self, match_results: List[Dict]) -> Dict:
        """Aggregate multiple match results."""
        valid_results = [r for r in match_results if "error" not in r]

        if not valid_results:
            return {"error": "All matches failed"}

        aggregated = {}

        # Win rate (count draws as 0.5 wins)
        wins = sum(1 for r in valid_results if r.get('winner') == 'fighter_a')
        draws = sum(1 for r in valid_results if r.get('winner') == 'draw')
        aggregated['win_rate'] = ((wins + draws * 0.5) / len(valid_results)) * 100

        # HP differential
        hp_diffs = [r['fighter_final_hp'] - r['dummy_final_hp'] for r in valid_results]
        aggregated['avg_hp_differential'] = statistics.mean(hp_diffs)

        # Average metrics
        metric_keys = [
            'duration', 'total_collisions', 'avg_distance',
            'fighter_extended_pct', 'fighter_defending_pct',
            'fighter_wall_time_pct', 'fighter_avg_stamina_drain'
        ]

        for key in metric_keys:
            values = [r.get(key, 0) for r in valid_results if key in r]
            if values:
                aggregated[f'avg_{key}'] = statistics.mean(values)

        return aggregated

    def generate_summary_report(self):
        """Generate a summary report."""
        report_path = self.output_dir / "SUMMARY_REPORT.md"

        with open(report_path, 'w') as f:
            f.write(f"# Test Matrix Summary Report\n")
            f.write(f"Generated: {self.timestamp}\n\n")

            for fighter_data in self.results:
                fighter = fighter_data['fighter']
                f.write(f"\n## {fighter}\n\n")

                for category, results in fighter_data['results'].items():
                    f.write(f"### {category.upper()} Tests\n\n")

                    # Calculate category statistics
                    win_rates = [r.get('win_rate', 0) for r in results.values() if 'win_rate' in r]
                    if win_rates:
                        avg_win_rate = statistics.mean(win_rates)
                        f.write(f"- **Average Win Rate**: {avg_win_rate:.1f}%\n")
                        f.write(f"- **Tests Passed**: {sum(1 for r in win_rates if r > 66)}/{len(win_rates)}\n\n")

                    f.write("| Test Dummy | Win Rate | HP Diff | Collisions | Avg Distance |\n")
                    f.write("|------------|----------|---------|------------|-------------|\n")

                    for dummy, result in results.items():
                        if "error" not in result:
                            f.write(f"| {dummy} | {result.get('win_rate', 0):.1f}% | ")
                            f.write(f"{result.get('avg_hp_differential', 0):+.1f} | ")
                            f.write(f"{result.get('avg_total_collisions', 0):.0f} | ")
                            f.write(f"{result.get('avg_avg_distance', 0):.1f}m |\n")
                        else:
                            f.write(f"| {dummy} | ERROR | - | - | - |\n")

                    f.write("\n")

        print(f"\n✅ Summary report saved to: {report_path}")

    def generate_detailed_report(self):
        """Generate a detailed analysis report."""
        report_path = self.output_dir / "DETAILED_REPORT.md"

        with open(report_path, 'w') as f:
            f.write(f"# Detailed Test Matrix Report\n")
            f.write(f"Generated: {self.timestamp}\n\n")

            for fighter_data in self.results:
                fighter = fighter_data['fighter']
                f.write(f"\n## {fighter}\n\n")

                for category, results in fighter_data['results'].items():
                    f.write(f"### {category.upper()} Category\n\n")

                    for dummy, result in results.items():
                        f.write(f"#### {dummy}\n\n")

                        if "error" in result:
                            f.write(f"❌ **ERROR**: {result['error']}\n\n")
                            continue

                        # Performance metrics
                        f.write("**Performance:**\n")
                        f.write(f"- Win Rate: {result.get('win_rate', 0):.1f}%\n")
                        f.write(f"- HP Differential: {result.get('avg_hp_differential', 0):+.1f}\n")
                        f.write(f"- Match Duration: {result.get('avg_duration', 0):.0f} ticks\n")
                        f.write(f"- Total Collisions: {result.get('avg_total_collisions', 0):.0f}\n\n")

                        # Behavior analysis
                        f.write("**Behavior:**\n")
                        f.write(f"- Average Distance: {result.get('avg_avg_distance', 0):.2f}m\n")
                        f.write(f"- Wall Time: {result.get('avg_fighter_wall_time_pct', 0):.1f}%\n")
                        f.write(f"- Position Variance: {result.get('avg_fighter_position_variance', 0):.2f}\n\n")

                        # Stance usage
                        f.write("**Stance Distribution:**\n")
                        f.write(f"- Extended: {result.get('avg_fighter_extended_pct', 0):.1f}%\n")
                        f.write(f"- Defending: {result.get('avg_fighter_defending_pct', 0):.1f}%\n")
                        f.write(f"- Neutral: {result.get('avg_fighter_neutral_pct', 0):.1f}%\n")
                        f.write(f"- Retracted: {result.get('avg_fighter_retracted_pct', 0):.1f}%\n\n")

                        f.write("---\n\n")

        print(f"✅ Detailed report saved to: {report_path}")

    def generate_csv_export(self):
        """Export results to CSV for further analysis."""
        csv_path = self.output_dir / "test_results.csv"

        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'fighter', 'category', 'dummy', 'win_rate', 'hp_differential',
                'duration', 'collisions', 'avg_distance', 'wall_time_pct',
                'extended_pct', 'defending_pct', 'neutral_pct', 'retracted_pct'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for fighter_data in self.results:
                fighter = fighter_data['fighter']

                for category, results in fighter_data['results'].items():
                    for dummy, result in results.items():
                        if "error" not in result:
                            row = {
                                'fighter': fighter,
                                'category': category,
                                'dummy': dummy,
                                'win_rate': result.get('win_rate', 0),
                                'hp_differential': result.get('avg_hp_differential', 0),
                                'duration': result.get('avg_duration', 0),
                                'collisions': result.get('avg_total_collisions', 0),
                                'avg_distance': result.get('avg_avg_distance', 0),
                                'wall_time_pct': result.get('avg_fighter_wall_time_pct', 0),
                                'extended_pct': result.get('avg_fighter_extended_pct', 0),
                                'defending_pct': result.get('avg_fighter_defending_pct', 0),
                                'neutral_pct': result.get('avg_fighter_neutral_pct', 0),
                                'retracted_pct': result.get('avg_fighter_retracted_pct', 0)
                            }
                            writer.writerow(row)

        print(f"✅ CSV export saved to: {csv_path}")


def main():
    """Run test matrix for specified fighters."""
    import sys

    # Default fighters to test
    default_fighters = [
        "fighters/examples/tank.py",
        "fighters/examples/rusher.py",
        "fighters/examples/balanced.py",
        "fighters/examples/grappler.py",
        "fighters/examples/berserker.py",
        "fighters/examples/zoner.py",
        "fighters/examples/dodger.py"
    ]

    # Get fighters from command line or use defaults
    fighters = sys.argv[1:] if len(sys.argv) > 1 else default_fighters

    # Run test matrix
    runner = TestMatrixRunner()
    runner.run_test_matrix(fighters, categories=["atomic", "behavioral"])

    print(f"\n{'='*80}")
    print(f"TEST MATRIX COMPLETE")
    print(f"Results saved to: {runner.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()