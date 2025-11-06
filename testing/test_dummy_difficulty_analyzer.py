#!/usr/bin/env python3
"""
Test Dummy Difficulty Analyzer

Analyzes the difficulty progression of test dummies to ensure
they provide appropriate training challenges.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


class DifficultyAnalyzer:
    """Analyzes difficulty levels and training value of test dummies."""

    def __init__(self):
        self.results = {}
        self.difficulty_scores = {}

    def run_match(self, fighter1: str, fighter2: str, seed: int = 42) -> Dict:
        """Run a match and collect basic metrics."""
        output_file = f"temp_match_{seed}.json"

        cmd = [
            "python", "atom_fight.py",
            fighter1, fighter2,
            "--seed", str(seed),
            "--max-ticks", "1000",
            "--save", output_file
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)

                Path(output_file).unlink()  # Clean up

                # Extract metrics
                result_data = data.get('result', {})
                telemetry = data.get('telemetry', {})
                ticks = telemetry.get('ticks', [])

                winner = result_data.get('winner', 'unknown')
                if 'draw' in winner.lower():
                    winner = 'draw'
                elif Path(fighter1).stem in winner:
                    winner = 'fighter1'
                elif Path(fighter2).stem in winner:
                    winner = 'fighter2'

                # Calculate engagement metrics
                collisions = sum(1 for t in ticks if t.get('collision', {}).get('occurred', False))

                # Calculate average distance
                distances = [abs(t['fighter_a']['position'] - t['fighter_b']['position']) for t in ticks]
                avg_distance = statistics.mean(distances) if distances else 0

                return {
                    'winner': winner,
                    'duration': len(ticks),
                    'collisions': collisions,
                    'avg_distance': avg_distance,
                    'final_hp_a': result_data.get('final_hp_a', 0),
                    'final_hp_b': result_data.get('final_hp_b', 0)
                }

            return {'error': 'No output file'}

        except Exception as e:
            return {'error': str(e)}

    def analyze_dummy_difficulty(self, dummy_path: str, reference_fighters: List[str]) -> Dict:
        """Analyze difficulty of a dummy against reference fighters."""
        dummy_name = Path(dummy_path).stem
        results = []

        for ref_fighter in reference_fighters:
            # Run 3 matches with different seeds
            for seed in [42, 123, 999]:
                match_result = self.run_match(ref_fighter, dummy_path, seed)
                if 'error' not in match_result:
                    results.append(match_result)

        if not results:
            return {'error': 'No valid matches'}

        # Calculate difficulty metrics
        win_rate = sum(1 for r in results if r['winner'] == 'fighter2') / len(results)
        draw_rate = sum(1 for r in results if r['winner'] == 'draw') / len(results)
        loss_rate = sum(1 for r in results if r['winner'] == 'fighter1') / len(results)

        avg_collisions = statistics.mean([r['collisions'] for r in results])
        avg_duration = statistics.mean([r['duration'] for r in results])
        avg_distance = statistics.mean([r['avg_distance'] for r in results])

        # Calculate difficulty score (0-100)
        # Higher score = harder dummy
        difficulty_score = (
            win_rate * 40 +                    # Winning = harder
            draw_rate * 20 +                    # Draws = moderate
            (1 - loss_rate) * 20 +             # Not losing = harder
            min(avg_collisions / 100, 1) * 10 + # More combat = harder
            (1 - avg_distance / 12) * 10       # Closer = harder
        ) * 100

        return {
            'dummy': dummy_name,
            'difficulty_score': difficulty_score,
            'win_rate': win_rate * 100,
            'draw_rate': draw_rate * 100,
            'loss_rate': loss_rate * 100,
            'avg_collisions': avg_collisions,
            'avg_duration': avg_duration,
            'avg_distance': avg_distance
        }

    def categorize_difficulty(self, score: float) -> str:
        """Categorize difficulty score into levels."""
        if score < 20:
            return "Trivial"
        elif score < 35:
            return "Easy"
        elif score < 50:
            return "Medium"
        elif score < 65:
            return "Hard"
        elif score < 80:
            return "Very Hard"
        else:
            return "Expert"

    def analyze_all_dummies(self):
        """Analyze difficulty of all test dummies."""
        print("\n=== ANALYZING DUMMY DIFFICULTY ===")
        print("Testing against reference AI (wanderer)\n")

        # Use wanderer as reference (weak AI for baseline)
        # Note: bumbler appears to be broken with 0-tick fights
        reference_fighters = ["fighters/training_opponents/wanderer.py"]

        categories = {
            "atomic": [],
            "behavioral": []
        }

        for category in categories.keys():
            dummy_dir = Path(f"fighters/test_dummies/{category}")
            if not dummy_dir.exists():
                continue

            print(f"\n{category.upper()} DUMMIES:")
            print("-" * 60)

            for dummy_file in sorted(dummy_dir.glob("*.py")):
                if dummy_file.name.startswith("__"):
                    continue

                result = self.analyze_dummy_difficulty(
                    str(dummy_file),
                    reference_fighters
                )

                if 'error' not in result:
                    difficulty = self.categorize_difficulty(result['difficulty_score'])
                    symbol = {
                        "Trivial": "⚪",
                        "Easy": "🟢",
                        "Medium": "🟡",
                        "Hard": "🟠",
                        "Very Hard": "🔴",
                        "Expert": "🟣"
                    }.get(difficulty, "❓")

                    print(f"{symbol} {result['dummy']:25} Score: {result['difficulty_score']:5.1f} ({difficulty})")
                    print(f"   Win: {result['win_rate']:.0f}% Draw: {result['draw_rate']:.0f}% Loss: {result['loss_rate']:.0f}%")
                    print(f"   Collisions: {result['avg_collisions']:.0f} Distance: {result['avg_distance']:.1f}m")

                    categories[category].append(result)
                else:
                    print(f"❌ {dummy_file.stem:25} ERROR: {result['error']}")

        self.results = categories
        return categories

    def generate_curriculum(self):
        """Generate a training curriculum based on difficulty."""
        print("\n" + "=" * 60)
        print("SUGGESTED TRAINING CURRICULUM")
        print("=" * 60)

        # Flatten and sort all dummies by difficulty
        all_dummies = []
        for category, dummies in self.results.items():
            for dummy in dummies:
                dummy['category'] = category
                all_dummies.append(dummy)

        all_dummies.sort(key=lambda x: x['difficulty_score'])

        # Group into curriculum levels
        levels = {
            "Level 1 - Fundamentals": [],
            "Level 2 - Basic Skills": [],
            "Level 3 - Intermediate": [],
            "Level 4 - Advanced": [],
            "Level 5 - Expert": []
        }

        for dummy in all_dummies:
            score = dummy['difficulty_score']
            if score < 20:
                levels["Level 1 - Fundamentals"].append(dummy)
            elif score < 35:
                levels["Level 2 - Basic Skills"].append(dummy)
            elif score < 50:
                levels["Level 3 - Intermediate"].append(dummy)
            elif score < 65:
                levels["Level 4 - Advanced"].append(dummy)
            else:
                levels["Level 5 - Expert"].append(dummy)

        # Print curriculum
        for level_name, dummies in levels.items():
            if dummies:
                print(f"\n{level_name}:")
                print("-" * 40)
                for dummy in dummies:
                    category_tag = "[A]" if dummy['category'] == "atomic" else "[B]"
                    print(f"  {category_tag} {dummy['dummy']:30} (Score: {dummy['difficulty_score']:.1f})")

        # Generate progression requirements
        print("\n\nPROGRESSION REQUIREMENTS:")
        print("-" * 40)
        print("Level 1 → 2: Win rate > 80% against all Level 1 dummies")
        print("Level 2 → 3: Win rate > 70% against all Level 2 dummies")
        print("Level 3 → 4: Win rate > 60% against all Level 3 dummies")
        print("Level 4 → 5: Win rate > 50% against all Level 4 dummies")
        print("Graduate:    Win rate > 40% against all Level 5 dummies")

    def analyze_training_value(self):
        """Analyze what skills each dummy teaches."""
        print("\n" + "=" * 60)
        print("TRAINING VALUE ANALYSIS")
        print("=" * 60)

        skill_map = {
            "stationary": "Basic attack timing and stance usage",
            "approach": "Pursuit and closing distance",
            "flee": "Chase mechanics and speed management",
            "shuttle": "Predictive movement and interception",
            "distance_keeper": "Spacing control and range management",
            "stamina": "Resource management and efficiency",
            "wall_hugger": "Wall combat and position recovery",
            "mirror": "Reactive movement and adaptation",
            "counter": "Evasion and anti-patterns",
            "charge": "Reaction timing and counter-charging",
            "perfect_defender": "Breaking strong defenses",
            "burst_attacker": "Defending against burst damage",
            "perfect_kiter": "Dealing with hit-and-run tactics",
            "adaptive": "Dynamic strategy adjustment",
            "wall_fighter": "Avoiding position traps",
            "stamina_optimizer": "Efficient resource battles"
        }

        print("\nSKILLS TAUGHT BY DUMMIES:")
        print("-" * 40)

        for category, dummies in self.results.items():
            print(f"\n{category.upper()}:")
            for dummy in sorted(dummies, key=lambda x: x['dummy']):
                name = dummy['dummy']
                # Find matching skill
                skill = "General combat"
                for key, value in skill_map.items():
                    if key in name:
                        skill = value
                        break

                difficulty = self.categorize_difficulty(dummy['difficulty_score'])
                print(f"  {name:30} → {skill}")
                print(f"    Difficulty: {difficulty}, Best for: {self.get_training_stage(difficulty)}")

    def get_training_stage(self, difficulty: str) -> str:
        """Get appropriate training stage for difficulty."""
        stages = {
            "Trivial": "Initial learning, understanding mechanics",
            "Easy": "Building confidence, basic strategies",
            "Medium": "Developing tactics, improving execution",
            "Hard": "Refining skills, handling challenges",
            "Very Hard": "Mastery preparation, edge cases",
            "Expert": "Final validation, tournament ready"
        }
        return stages.get(difficulty, "General training")


def main():
    """Run difficulty analysis."""
    print("TEST DUMMY DIFFICULTY ANALYZER")
    print("=" * 60)
    print("Analyzing difficulty levels and training progression")

    analyzer = DifficultyAnalyzer()

    # Analyze all dummies
    analyzer.analyze_all_dummies()

    # Generate curriculum
    analyzer.generate_curriculum()

    # Analyze training value
    analyzer.analyze_training_value()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()