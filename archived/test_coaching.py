#!/usr/bin/env python3
"""
Test script for the coaching system.

This script measures the impact of different coaching strategies on fight outcomes.
"""

import sys
from pathlib import Path
import importlib.util
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.arena.match_orchestrator import MatchOrchestrator
from src.arena.arena import Arena
from src.coaching.coaching_wrapper import CoachingWrapper, AdaptiveCoachingWrapper


def load_fighter(fighter_path: str):
    """Load a fighter from a Python file."""
    spec = importlib.util.spec_from_file_location("fighter", fighter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Fighter()


class CoachingTester:
    """Test different coaching strategies."""

    def __init__(self, fighter_a_path: str, fighter_b_path: str, episodes: int = 100):
        self.fighter_a_path = fighter_a_path
        self.fighter_b_path = fighter_b_path
        self.episodes = episodes
        self.results = {}

    def run_match(self, coaching_strategy: str = None) -> Dict:
        """Run a single match with optional coaching."""

        # Load fresh fighters
        fighter_a = load_fighter(self.fighter_a_path)
        fighter_b = load_fighter(self.fighter_b_path)

        # Apply coaching if specified
        if coaching_strategy:
            fighter_a = CoachingWrapper(fighter_a)
            self.apply_coaching_strategy(fighter_a, coaching_strategy)

        # Create arena and orchestrator
        arena = Arena()
        orchestrator = MatchOrchestrator(
            arena=arena,
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            max_ticks=500
        )

        # Run match
        while orchestrator.current_tick < orchestrator.max_ticks:
            orchestrator.step()

            # Apply dynamic coaching adjustments
            if coaching_strategy == "smart" and isinstance(fighter_a, CoachingWrapper):
                self.apply_smart_coaching(fighter_a, orchestrator)

        # Determine outcome
        if orchestrator.fighter_a_hp <= 0:
            winner = "B"
        elif orchestrator.fighter_b_hp <= 0:
            winner = "A"
        elif orchestrator.fighter_a_hp > orchestrator.fighter_b_hp:
            winner = "A"
        elif orchestrator.fighter_b_hp > orchestrator.fighter_a_hp:
            winner = "B"
        else:
            winner = "draw"

        return {
            "winner": winner,
            "fighter_a_hp": orchestrator.fighter_a_hp,
            "fighter_b_hp": orchestrator.fighter_b_hp,
            "duration": orchestrator.current_tick,
            "damage_dealt": max(0, 100 - orchestrator.fighter_b_hp),
            "damage_taken": max(0, 100 - orchestrator.fighter_a_hp)
        }

    def apply_coaching_strategy(self, coached_fighter: CoachingWrapper, strategy: str):
        """Apply a predefined coaching strategy."""

        if strategy == "aggressive":
            coached_fighter.receive_coaching("AGGRESSIVE")

        elif strategy == "defensive":
            coached_fighter.receive_coaching("DEFENSIVE")

        elif strategy == "balanced":
            coached_fighter.receive_coaching("BALANCED")

        # Smart strategy is applied dynamically during the match

    def apply_smart_coaching(self, coached_fighter: CoachingWrapper, orchestrator: MatchOrchestrator):
        """Apply intelligent coaching based on game state."""

        tick = orchestrator.current_tick
        distance = abs(orchestrator.fighter_a_position - orchestrator.fighter_b_position)
        stamina_a = orchestrator.fighter_a_stamina
        stamina_b = orchestrator.fighter_b_stamina
        hp_a = orchestrator.fighter_a_hp
        hp_b = orchestrator.fighter_b_hp

        # Smart coaching decisions
        if tick % 50 == 0:  # Adjust every 50 ticks
            if hp_a > hp_b * 1.5 and stamina_a > 30:
                # We're winning with energy - press advantage
                coached_fighter.receive_coaching("AGGRESSIVE")

            elif hp_a < hp_b * 0.7:
                # We're losing badly - defensive mode
                coached_fighter.receive_coaching("DEFENSIVE")

            elif stamina_a < 20 and stamina_b > 40:
                # Low stamina, opponent has energy - retreat
                coached_fighter.receive_coaching("RETREAT")

            elif stamina_b < 20 and stamina_a > 40:
                # Opponent is tired - rush!
                coached_fighter.receive_coaching("RUSH")

            elif distance > 4 and hp_a > hp_b:
                # We're ahead and far away - maintain distance
                coached_fighter.receive_coaching("DEFENSIVE")

            else:
                # Default to balanced
                coached_fighter.receive_coaching("BALANCED")

    def test_strategy(self, strategy: str) -> Dict:
        """Test a specific coaching strategy over multiple episodes."""

        wins = 0
        draws = 0
        losses = 0
        total_damage_dealt = 0
        total_damage_taken = 0
        total_duration = 0

        print(f"Testing {strategy:15s}: ", end="", flush=True)

        for i in range(self.episodes):
            result = self.run_match(strategy)

            if result["winner"] == "A":
                wins += 1
            elif result["winner"] == "B":
                losses += 1
            else:
                draws += 1

            total_damage_dealt += result["damage_dealt"]
            total_damage_taken += result["damage_taken"]
            total_duration += result["duration"]

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        win_rate = wins / self.episodes * 100
        avg_damage_dealt = total_damage_dealt / self.episodes
        avg_damage_taken = total_damage_taken / self.episodes
        avg_duration = total_duration / self.episodes

        print(f" Win rate: {win_rate:5.1f}%")

        return {
            "strategy": strategy,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "avg_damage_dealt": avg_damage_dealt,
            "avg_damage_taken": avg_damage_taken,
            "avg_duration": avg_duration,
            "damage_ratio": avg_damage_dealt / max(avg_damage_taken, 1)
        }

    def run_all_tests(self) -> List[Dict]:
        """Test all coaching strategies."""

        strategies = ["none", "aggressive", "defensive", "balanced", "smart"]
        results = []

        print(f"\n{'='*60}")
        print(f"  COACHING IMPACT ANALYSIS - {self.episodes} episodes each")
        print(f"{'='*60}\n")

        for strategy in strategies:
            result = self.test_strategy(strategy)
            results.append(result)

        return results

    def display_results(self, results: List[Dict]):
        """Display test results in a formatted table."""

        print(f"\n{'='*60}")
        print("                    RESULTS SUMMARY")
        print(f"{'='*60}\n")

        # Header
        print(f"{'Strategy':<15} {'Win%':>6} {'Wins':>5} {'Losses':>6} {'Draws':>5} "
              f"{'Dmg Dealt':>10} {'Dmg Taken':>10} {'Ratio':>6}")
        print("-" * 80)

        # Find baseline (no coaching)
        baseline = next((r for r in results if r["strategy"] == "none"), None)

        # Results
        for result in results:
            # Calculate improvement over baseline
            improvement = ""
            if baseline and result["strategy"] != "none":
                delta = result["win_rate"] - baseline["win_rate"]
                improvement = f" ({delta:+.1f}%)"

            print(f"{result['strategy']:<15} "
                  f"{result['win_rate']:>5.1f}% "
                  f"{result['wins']:>5} "
                  f"{result['losses']:>6} "
                  f"{result['draws']:>5} "
                  f"{result['avg_damage_dealt']:>10.1f} "
                  f"{result['avg_damage_taken']:>10.1f} "
                  f"{result['damage_ratio']:>6.2f}"
                  f"{improvement}")

        # Best strategy
        best = max(results, key=lambda x: x["win_rate"])
        print(f"\n{'='*60}")
        print(f"BEST STRATEGY: {best['strategy']} ({best['win_rate']:.1f}% win rate)")

        if baseline:
            improvement = best["win_rate"] - baseline["win_rate"]
            print(f"Improvement over no coaching: {improvement:+.1f}%")

        print(f"{'='*60}\n")

    def test_adaptive_coaching(self):
        """Test advanced adaptive coaching strategies."""

        print(f"\n{'='*60}")
        print("         TESTING ADAPTIVE COACHING STRATEGIES")
        print(f"{'='*60}\n")

        # Load fighters
        fighter_a = load_fighter(self.fighter_a_path)
        fighter_b = load_fighter(self.fighter_b_path)

        # Create adaptive coaching wrapper
        coached_fighter = AdaptiveCoachingWrapper(fighter_a)

        strategies = ["ROPE_A_DOPE", "PEEK_A_BOO", "OUTBOXER"]

        for strategy in strategies:
            wins = 0

            print(f"Testing {strategy:15s}: ", end="", flush=True)

            for episode in range(20):  # Fewer episodes for adaptive strategies
                # Reset fighters
                fighter_a = load_fighter(self.fighter_a_path)
                fighter_b = load_fighter(self.fighter_b_path)
                coached_fighter = AdaptiveCoachingWrapper(fighter_a)

                # Create arena and orchestrator
                arena = Arena()
                orchestrator = MatchOrchestrator(
                    arena=arena,
                    fighter_a=coached_fighter,
                    fighter_b=fighter_b,
                    max_ticks=500
                )

                # Run match with boxing strategy
                while orchestrator.current_tick < orchestrator.max_ticks:
                    # Apply boxing strategy based on game state
                    snapshot = {
                        "stamina": orchestrator.fighter_a_stamina,
                        "opponent_stamina": orchestrator.fighter_b_stamina,
                        "distance": abs(orchestrator.fighter_a_position - orchestrator.fighter_b_position)
                    }
                    coached_fighter.apply_boxing_strategy(strategy, snapshot)
                    orchestrator.step()

                # Check winner
                if orchestrator.fighter_a_hp > orchestrator.fighter_b_hp:
                    wins += 1

                print(".", end="", flush=True)

            win_rate = wins / 20 * 100
            print(f" Win rate: {win_rate:5.1f}%")

        print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the impact of coaching on fight outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--fighter-a",
        default="fighters/examples/balanced.py",
        help="Path to fighter A (default: fighters/examples/balanced.py)"
    )
    parser.add_argument(
        "--fighter-b",
        default="fighters/examples/tank.py",
        help="Path to fighter B (default: fighters/examples/tank.py)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per test (default: 100)"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Also test adaptive boxing strategies"
    )

    args = parser.parse_args()

    # Verify fighter files exist
    if not Path(args.fighter_a).exists():
        print(f"Error: Fighter A not found: {args.fighter_a}")
        sys.exit(1)

    if not Path(args.fighter_b).exists():
        print(f"Error: Fighter B not found: {args.fighter_b}")
        sys.exit(1)

    # Run tests
    tester = CoachingTester(args.fighter_a, args.fighter_b, args.episodes)

    # Basic coaching strategies
    results = tester.run_all_tests()
    tester.display_results(results)

    # Adaptive strategies (if requested)
    if args.adaptive:
        tester.test_adaptive_coaching()

    print("\nTesting complete!")


if __name__ == "__main__":
    main()