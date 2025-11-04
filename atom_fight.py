#!/usr/bin/env python3
"""
Atom Combat - Simple Fight Runner

Usage:
    python atom_fight.py fighter_a.py fighter_b.py [options]

Examples:
    # Run a fight with default settings
    python atom_fight.py fighters/rusher.py fighters/tank.py

    # Generate HTML replay
    python atom_fight.py fighters/rusher.py fighters/tank.py --html replay.html

    # Show ASCII animation
    python atom_fight.py fighters/rusher.py fighters/tank.py --watch

    # Custom fighter masses
    python atom_fight.py fighters/rusher.py fighters/tank.py --mass-a 65 --mass-b 80

    # Save telemetry
    python atom_fight.py fighters/rusher.py fighters/tank.py --save match.json.gz
"""

import argparse
import sys
import importlib.util
from pathlib import Path

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import AsciiRenderer, HtmlRenderer
from src.telemetry import ReplayStore


def load_fighter_function(filepath: str):
    """
    Load a fighter decision function from a Python file.

    The file must define a function called 'decide' with signature:
        def decide(snapshot: dict) -> dict
    """
    path = Path(filepath)

    if not path.exists():
        print(f"Error: Fighter file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    # Load module from file
    spec = importlib.util.spec_from_file_location("fighter_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the decide function
    if not hasattr(module, 'decide'):
        print(f"Error: Fighter file must define a 'decide' function: {filepath}", file=sys.stderr)
        sys.exit(1)

    return module.decide


def main():
    parser = argparse.ArgumentParser(
        description='Run an Atom Combat match between two fighters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('fighter_a', help='Path to fighter A Python file')
    parser.add_argument('fighter_b', help='Path to fighter B Python file')

    # Fighter configuration
    parser.add_argument('--name-a', default=None, help='Name for fighter A (default: filename)')
    parser.add_argument('--name-b', default=None, help='Name for fighter B (default: filename)')
    parser.add_argument('--mass-a', type=float, default=70.0, help='Mass for fighter A in kg (default: 70)')
    parser.add_argument('--mass-b', type=float, default=70.0, help='Mass for fighter B in kg (default: 75)')
    parser.add_argument('--pos-a', type=float, default=2.0, help='Starting position for fighter A (default: 2.0)')
    parser.add_argument('--pos-b', type=float, default=10.0, help='Starting position for fighter B (default: 10.0)')

    # Match configuration
    parser.add_argument('--max-ticks', type=int, default=1000, help='Maximum ticks before timeout (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    # Output options
    parser.add_argument('--watch', action='store_true', help='Show ASCII animation in terminal')
    parser.add_argument('--html', metavar='FILE', help='Generate HTML replay file')
    parser.add_argument('--save', metavar='FILE', help='Save telemetry to file (JSON or .json.gz)')
    parser.add_argument('--speed', type=float, default=5.0, help='Playback speed for --watch (default: 5.0)')

    # Parse arguments
    args = parser.parse_args()

    # Load fighters
    print("Loading fighters...")
    try:
        decide_a = load_fighter_function(args.fighter_a)
        decide_b = load_fighter_function(args.fighter_b)
    except Exception as e:
        print(f"Error loading fighters: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate names if not provided
    name_a = args.name_a or Path(args.fighter_a).stem
    name_b = args.name_b or Path(args.fighter_b).stem

    print(f"  ✓ Loaded: {name_a} ({args.mass_a}kg)")
    print(f"  ✓ Loaded: {name_b} ({args.mass_b}kg)")

    # Create world config
    config = WorldConfig()

    # Create orchestrator
    orchestrator = MatchOrchestrator(config, max_ticks=args.max_ticks, record_telemetry=True)

    # Define fighter specs
    fighter_a_spec = {"name": name_a, "mass": args.mass_a, "position": args.pos_a}
    fighter_b_spec = {"name": name_b, "mass": args.mass_b, "position": args.pos_b}

    # Run match
    print(f"\nRunning match (max {args.max_ticks} ticks)...")
    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        decide_a,
        decide_b,
        seed=args.seed
    )

    # Evaluate spectacle
    evaluator = SpectacleEvaluator()
    score = evaluator.evaluate(result.telemetry, result)

    # Print results
    print("\n" + "=" * 60)
    print("MATCH RESULTS".center(60))
    print("=" * 60)
    print(f"Winner: {result.winner}")
    print(f"Duration: {result.total_ticks} ticks ({result.total_ticks * config.dt:.1f}s)")
    print(f"Final HP: {result.final_hp_a:.1f} vs {result.final_hp_b:.1f}")

    collision_count = len([e for e in result.events if e["type"] == "COLLISION"])
    print(f"Collisions: {collision_count}")

    print(f"\nSpectacle Score: {score.overall:.3f}")
    if score.overall >= 0.8:
        print("  Rating: EXCELLENT ⭐⭐⭐⭐⭐")
    elif score.overall >= 0.6:
        print("  Rating: GOOD ⭐⭐⭐⭐")
    elif score.overall >= 0.4:
        print("  Rating: FAIR ⭐⭐⭐")
    else:
        print("  Rating: POOR ⭐⭐")

    # Save telemetry if requested
    if args.save:
        print(f"\nSaving telemetry to {args.save}...")
        store = ReplayStore()
        store.replay_dir = Path(args.save).parent
        store.save(
            result.telemetry,
            result,
            compress=args.save.endswith('.gz'),
            filename=Path(args.save).name
        )
        print("  ✓ Saved")

    # Generate HTML if requested
    if args.html:
        print(f"\nGenerating HTML replay: {args.html}...")
        renderer = HtmlRenderer()
        renderer.generate_replay_html(
            result.telemetry,
            result,
            args.html,
            spectacle_score=score
        )
        print(f"  ✓ Generated: {args.html}")
        print(f"  → Open in browser to watch replay")

    # Show ASCII animation if requested
    if args.watch:
        print("\n" + "=" * 60)
        print("REPLAY".center(60))
        print("=" * 60)
        renderer = AsciiRenderer(arena_width=config.arena_width)
        renderer.play_replay(
            result.telemetry,
            result,
            spectacle_score=score,
            playback_speed=args.speed,
            skip_ticks=5
        )

    print()


if __name__ == "__main__":
    main()
