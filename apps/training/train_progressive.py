#!/usr/bin/env python3
"""Progressive training application for Atom Combat."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atom.training.pipelines import ProgressiveTrainer


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for progressive training."""
    parser = argparse.ArgumentParser(
        description="Progressive training for Atom Combat fighters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_progressive.py --mode quick --algorithm ppo
  python train_progressive.py --mode curriculum --timesteps 1000000
  python train_progressive.py --mode complete --timesteps 2000000 --population 8 --generations 10
        """,
    )

    parser.add_argument("--mode", choices=["quick", "curriculum", "population", "complete"], default="complete", help="Training mode")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo", help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Timesteps for curriculum training")
    parser.add_argument("--seed", type=int, default=1337, help="Training seed for reproducible runs")
    parser.add_argument("--population", type=int, default=8, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Population training generations")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: outputs/progressive_TIMESTAMP)")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use for PPO parallel environments (default: 8)")
    parser.add_argument("--n-parallel-fighters", type=int, default=None, help="Number of fighters to train in parallel in population mode (default: 2 for GPU, cpu_count-1 for CPU)")
    parser.add_argument("--episodes-per-gen", type=int, default=2000, help="Training episodes per generation for population training (default: 2000)")
    parser.add_argument("--max-ticks", type=int, default=250, help="Maximum ticks per episode (default: 250, ~21 seconds)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="cuda", help="Device to use for training: cpu, cuda (GPU), or auto")
    parser.add_argument("--use-vmap", action="store_true", help="Enable JAX vmap for GPU-accelerated training (CUDA or ROCm)")
    parser.add_argument("--population-cpu-only", action="store_true", help="Force CPU-only training for population phase (avoids GPU OOM issues)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to see detailed fight information")
    parser.add_argument("--record-replays", action="store_true", help="Record fight replays for creating a training montage (samples bottom/middle/top spectacle)")
    parser.add_argument("--replay-frequency", type=int, default=5, help="Record replays every N generations for population training (default: 5)")
    parser.add_argument("--override-episodes-per-level", type=int, default=None, help="Force graduation after N episodes (for testing). None for normal graduation.")
    parser.add_argument("--resume-curriculum", action="store_true", help="Resume curriculum training from latest checkpoint bundle in output dir.")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Save curriculum checkpoint bundle every N timesteps (default: 100000).")
    parser.add_argument("--keep-top", type=float, default=0.5, help="Fraction of population to keep during evolution (default: 0.5 = top 50%%)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation strength for evolved fighters (default: 0.1 = 10%% weight noise)")
    parser.add_argument("--evolution-frequency", type=int, default=2, help="Evolve population every N generations (default: 2)")
    return parser


def resolve_output_dir(output_dir: str | None) -> str:
    """Resolve the output directory, applying the timestamp default when needed."""
    if output_dir is not None:
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/progressive_{timestamp}"


def main(argv: list[str] | None = None) -> None:
    """Run the progressive training CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    trainer = ProgressiveTrainer(
        algorithm=args.algorithm,
        output_dir=resolve_output_dir(args.output_dir),
        verbose=True,
        seed=args.seed,
        n_parallel_fighters=args.n_parallel_fighters,
        n_envs=args.cores,
        max_ticks=args.max_ticks,
        device=args.device,
        use_vmap=args.use_vmap,
        population_cpu_only=args.population_cpu_only,
        debug=args.debug,
        record_replays=args.record_replays,
        replay_frequency=args.replay_frequency,
        override_episodes_per_level=args.override_episodes_per_level,
        checkpoint_interval=args.checkpoint_interval,
    )

    if args.mode == "quick":
        trainer.run_complete_pipeline(
            curriculum_timesteps=10_000,
            population_generations=2,
            population_size=4,
            episodes_per_generation=500,
            resume_curriculum=args.resume_curriculum,
        )
        return

    if args.mode == "curriculum":
        trainer.run_curriculum_training(
            timesteps=args.timesteps,
            resume_from_latest=args.resume_curriculum,
        )
        return

    if args.mode == "population":
        print("Note: Population mode requires an existing curriculum graduate model.")
        trainer.run_population_training(
            generations=args.generations,
            episodes_per_generation=args.episodes_per_gen,
            population_size=args.population,
            keep_top=args.keep_top,
            evolution_frequency=args.evolution_frequency,
            mutation_rate=args.mutation_rate,
        )
        return

    trainer.run_complete_pipeline(
        curriculum_timesteps=args.timesteps,
        population_generations=args.generations,
        population_size=args.population,
        episodes_per_generation=args.episodes_per_gen,
        keep_top=args.keep_top,
        evolution_frequency=args.evolution_frequency,
        mutation_rate=args.mutation_rate,
        resume_curriculum=args.resume_curriculum,
    )


if __name__ == "__main__":
    main()
