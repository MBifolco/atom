#!/usr/bin/env python3
"""Atom Combat fight runner application."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-detect GPU issues and fallback to CPU.
try:
    import jax

    devices = jax.devices()
    if any('gpu' in str(device).lower() or 'rocm' in str(device).lower() for device in devices):
        try:
            test = jax.numpy.array([1.0])
            _ = test + 1
        except Exception:
            print("⚠️ GPU initialization failed, using CPU mode")
            os.environ["ATOM_FORCE_CPU"] = "1"
except Exception:
    pass

from src.atom.runtime.arena import WorldConfig
from src.atom.runtime.evaluator import SpectacleEvaluator
from src.atom.runtime.orchestrator import MatchOrchestrator
from src.atom.runtime.renderer import AsciiRenderer, HtmlRenderer
from src.atom.runtime.telemetry import ReplayStore


def load_fighter_function(filepath: str) -> Callable[[dict], dict]:
    """Load a fighter decision function from a Python file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: Fighter file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("fighter_module", path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load fighter file: {filepath}", file=sys.stderr)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "decide"):
        print(f"Error: Fighter file must define a 'decide' function: {filepath}", file=sys.stderr)
        sys.exit(1)

    return module.decide


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run an Atom Combat match between two fighters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py
    python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --html replay.html
    python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --watch
        """,
    )

    parser.add_argument("fighter_a", help="Path to fighter A Python file")
    parser.add_argument("fighter_b", help="Path to fighter B Python file")

    parser.add_argument("--name-a", default=None, help="Name for fighter A (default: filename)")
    parser.add_argument("--name-b", default=None, help="Name for fighter B (default: filename)")
    parser.add_argument("--mass-a", type=float, default=70.0, help="Mass for fighter A in kg (default: 70)")
    parser.add_argument("--mass-b", type=float, default=70.0, help="Mass for fighter B in kg (default: 75)")
    parser.add_argument("--pos-a", type=float, default=2.0, help="Starting position for fighter A (default: 2.0)")
    parser.add_argument("--pos-b", type=float, default=10.0, help="Starting position for fighter B (default: 10.0)")

    parser.add_argument("--max-ticks", type=int, default=1000, help="Maximum ticks before timeout (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    parser.add_argument("--watch", action="store_true", help="Show ASCII animation in terminal")
    parser.add_argument("--html", metavar="FILE", help="Generate HTML replay file")
    parser.add_argument("--save", metavar="FILE", help="Save telemetry to file (JSON or .json.gz)")
    parser.add_argument("--speed", type=float, default=5.0, help="Playback speed for --watch (default: 5.0)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the fight CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Loading fighters...")
    try:
        decide_a = load_fighter_function(args.fighter_a)
        decide_b = load_fighter_function(args.fighter_b)
    except Exception as exc:
        print(f"Error loading fighters: {exc}", file=sys.stderr)
        sys.exit(1)

    name_a = args.name_a or Path(args.fighter_a).stem
    name_b = args.name_b or Path(args.fighter_b).stem

    print(f"  ✓ Loaded: {name_a} ({args.mass_a}kg)")
    print(f"  ✓ Loaded: {name_b} ({args.mass_b}kg)")

    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=args.max_ticks, record_telemetry=True)

    fighter_a_spec = {"name": name_a, "mass": args.mass_a, "position": args.pos_a}
    fighter_b_spec = {"name": name_b, "mass": args.mass_b, "position": args.pos_b}

    print(f"\nRunning match (max {args.max_ticks} ticks)...")
    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        decide_a,
        decide_b,
        seed=args.seed,
    )

    evaluator = SpectacleEvaluator()
    score = evaluator.evaluate(result.telemetry, result)

    print("\n" + "=" * 60)
    print("MATCH RESULTS".center(60))
    print("=" * 60)
    print(f"Winner: {result.winner}")
    print(f"Duration: {result.total_ticks} ticks ({result.total_ticks * config.dt:.1f}s)")
    print(f"Final HP: {result.final_hp_a:.1f} vs {result.final_hp_b:.1f}")

    hit_count = len([event for event in result.events if event["type"] == "HIT"])
    print(f"Hits: {hit_count}")

    print(f"\nSpectacle Score: {score.overall:.3f}")
    if score.overall >= 0.8:
        print("  Rating: EXCELLENT ⭐⭐⭐⭐⭐")
    elif score.overall >= 0.6:
        print("  Rating: GOOD ⭐⭐⭐⭐")
    elif score.overall >= 0.4:
        print("  Rating: FAIR ⭐⭐⭐")
    else:
        print("  Rating: POOR ⭐⭐")

    if args.save:
        print(f"\nSaving telemetry to {args.save}...")
        store = ReplayStore()
        store.replay_dir = Path(args.save).parent
        store.save(
            result.telemetry,
            result,
            compress=args.save.endswith(".gz"),
            filename=Path(args.save).name,
        )
        print("  ✓ Saved")

    if args.html:
        print(f"\nGenerating HTML replay: {args.html}...")
        renderer = HtmlRenderer()
        renderer.generate_replay_html(
            result.telemetry,
            result,
            args.html,
            spectacle_score=score,
        )
        print(f"  ✓ Generated: {args.html}")
        print("  → Open in browser to watch replay")

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
            skip_ticks=5,
        )

    print()


if __name__ == "__main__":
    main()
