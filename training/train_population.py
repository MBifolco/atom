#!/usr/bin/env python3
"""
Population-Based Training for Atom Combat

Train a population of fighters that learn from each other, creating
diverse strategies through competition and evolution.
"""

# Add parent directory to path FIRST before any imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add training directory too

import argparse
from src.trainers.population import PopulationTrainer
from src.arena import WorldConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train a population of Atom Combat fighters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a small population for testing
  python train_population.py --population 4 --generations 2 --episodes 100

  # Standard population training
  python train_population.py --population 8 --generations 10 --episodes 500

  # Large diverse population
  python train_population.py --population 16 --generations 20 --episodes 1000 --mass-range 50 90

  # Fast evolution with SAC
  python train_population.py --algorithm sac --evolution-freq 1 --keep-top 0.3
        """
    )

    # Core parameters
    parser.add_argument(
        "--population",
        type=int,
        default=8,
        help="Size of the fighter population (default: 8)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to evolve (default: 10)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Training episodes per generation (default: 500)"
    )

    # Training configuration
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use (default: ppo)"
    )
    parser.add_argument(
        "--envs-per-fighter",
        type=int,
        default=2,
        help="Parallel environments per fighter (default: 2)"
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=1000,
        help="Maximum ticks per episode (default: 1000)"
    )

    # Fighter configuration
    parser.add_argument(
        "--mass-range",
        nargs=2,
        type=float,
        default=[60.0, 85.0],
        help="Min and max mass for fighters (default: 60 85)"
    )

    # Evolution parameters
    parser.add_argument(
        "--evolution-freq",
        type=int,
        default=2,
        help="Evolve population every N generations (default: 2)"
    )
    parser.add_argument(
        "--keep-top",
        type=float,
        default=0.5,
        help="Fraction of population to keep during evolution (default: 0.5)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="Mutation rate for evolved fighters (default: 0.1)"
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/population",
        help="Output directory for models and logs (default: outputs/population)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.population < 2:
        print("Error: Population size must be at least 2")
        sys.exit(1)

    if args.keep_top <= 0 or args.keep_top > 1:
        print("Error: --keep-top must be between 0 and 1")
        sys.exit(1)

    if args.mass_range[0] >= args.mass_range[1]:
        print("Error: Invalid mass range")
        sys.exit(1)

    # Create trainer
    print("\n" + "="*80)
    print("ATOM COMBAT - POPULATION TRAINING")
    print("="*80)
    print(f"Population: {args.population} fighters")
    print(f"Generations: {args.generations}")
    print(f"Episodes/gen: {args.episodes}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Mass range: {args.mass_range[0]:.0f}-{args.mass_range[1]:.0f}kg")
    print(f"Evolution: Every {args.evolution_freq} generations (keep top {args.keep_top:.0%})")
    print(f"Output: {args.output}")
    print("="*80)

    # Initialize trainer
    trainer = PopulationTrainer(
        population_size=args.population,
        config=WorldConfig(),
        algorithm=args.algorithm,
        output_dir=args.output,
        n_envs_per_fighter=args.envs_per_fighter,
        max_ticks=args.max_ticks,
        mass_range=tuple(args.mass_range),
        verbose=not args.quiet
    )

    # Run training
    try:
        trainer.train(
            generations=args.generations,
            episodes_per_generation=args.episodes,
            evolution_frequency=args.evolution_freq
        )

        print("\n✅ Population training complete!")
        print(f"Models saved to: {args.output}/models/")
        print(f"Logs saved to: {args.output}/logs/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current generation...")
        trainer.save_population()
        print(f"Models saved to: {args.output}/models/")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()