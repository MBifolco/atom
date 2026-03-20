#!/usr/bin/env python3
"""
Run a local deterministic baseline training command.

Use this during refactor work to validate behavior locally before Colab gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.utils.baseline_harness import BaselineRunConfig, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local deterministic progressive-training baseline."
    )
    parser.add_argument("--mode", default="curriculum", choices=["quick", "curriculum", "population", "complete"])
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--cores", type=int, default=1, help="Parallel env count for local runs")
    parser.add_argument("--use-vmap", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=250)
    parser.add_argument("--override-episodes-per-level", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/local_baselines")
    parser.add_argument("--quiet", action="store_true", help="Do not stream subprocess output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BaselineRunConfig(
        output_dir=args.output_dir,
        mode=args.mode,
        timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        cores=args.cores,
        use_vmap=args.use_vmap,
        max_ticks=args.max_ticks,
        override_episodes_per_level=args.override_episodes_per_level,
    )

    result = run_baseline(
        config=config,
        stream_output=not args.quiet,
        check=False,
    )

    print("\nBaseline run summary")
    print(f"  returncode: {result.returncode}")
    print(f"  duration:   {result.duration_seconds:.1f}s")
    print(f"  log:        {result.log_path}")
    print(f"  metadata:   {result.metadata_path}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
