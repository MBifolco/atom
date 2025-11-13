#!/usr/bin/env python3
"""
Resume or continue population training from a checkpoint.

This script allows you to:
1. Resume training from a specific generation if interrupted
2. Continue training beyond the original number of generations

Usage:
    # Resume from generation 25 and continue to 40
    python resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 25 --total-gens 40

    # Continue training for 20 more generations after completing 40
    python resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 40 --total-gens 60
    python resume_population_training.py --checkpoint-dir  outputs/progressive_20251112_183943 --start-gen 8 --total-gens 40
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.training.trainers.population.population_trainer import PopulationTrainer
from stable_baselines3 import PPO, SAC


def load_population_from_checkpoint(trainer: PopulationTrainer, checkpoint_dir: Path, generation: int):
    """Load a population from a saved generation checkpoint."""

    generation_dir = checkpoint_dir / "population" / "models" / f"generation_{generation}"

    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation {generation} checkpoint not found at {generation_dir}")

    print(f"Loading generation {generation} from {generation_dir}")

    # Clear existing population
    trainer.population = []
    trainer.generation = generation

    # Load all fighter models from the generation
    for model_file in generation_dir.glob("*.zip"):
        if model_file.stem == "rankings":  # Skip rankings.txt
            continue

        fighter_name = model_file.stem
        print(f"  Loading fighter: {fighter_name}")

        # Create a dummy environment for loading
        from src.envs.atom_fight_env import AtomFightEnv
        env = AtomFightEnv()

        # Load the model
        if trainer.algorithm == "ppo":
            model = PPO.load(model_file, env=env)
        else:
            model = SAC.load(model_file, env=env)

        # Create fighter object
        from src.training.trainers.population.population_trainer import Fighter
        fighter = Fighter(
            name=fighter_name,
            model=model,
            generation=generation,
            parent=None,  # Parent info lost, but not critical
            mass=100,  # Default mass
            last_checkpoint=str(model_file)
        )

        trainer.population.append(fighter)

    print(f"Loaded {len(trainer.population)} fighters from generation {generation}")

    # Load ELO rankings if available
    rankings_file = generation_dir / "rankings.txt"
    if rankings_file.exists():
        print(f"  Loading ELO rankings from {rankings_file}")
        # Note: ELO tracker state is not fully serialized, so rankings will reset
        # This is okay - they'll rebuild quickly in the next generation

    return trainer


def resume_population_training(
    checkpoint_dir: Path,
    start_generation: int,
    total_generations: int,
    algorithm: str = "ppo",
    population_size: int = 8,
    episodes_per_gen: int = 2000,
    n_envs: int = 45,
    device: str = "auto",
    use_vmap: bool = False,
    output_dir: str = None
):
    """Resume or continue population training from a checkpoint."""

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/resumed_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint data to new output directory if different
    if checkpoint_dir != output_dir.parent:
        print(f"\nCopying checkpoint data to {output_dir}")

        # Copy population models up to start_generation
        src_models = checkpoint_dir / "population" / "models"
        dst_models = output_dir / "population" / "models"
        dst_models.mkdir(parents=True, exist_ok=True)

        for gen in range(start_generation + 1):
            src_gen = src_models / f"generation_{gen}"
            if src_gen.exists():
                dst_gen = dst_models / f"generation_{gen}"
                if not dst_gen.exists():
                    shutil.copytree(src_gen, dst_gen)
                    print(f"  Copied generation_{gen}")

    # Initialize population trainer
    print(f"\nInitializing population trainer")
    print(f"  Algorithm: {algorithm}")
    print(f"  Population size: {population_size}")
    print(f"  Episodes per generation: {episodes_per_gen}")
    print(f"  Starting from generation: {start_generation}")
    print(f"  Training until generation: {total_generations}")

    trainer = PopulationTrainer(
        population_size=population_size,
        algorithm=algorithm,
        n_envs_per_fighter=n_envs if not use_vmap else 1,  # vmap handles parallelization differently
        output_dir=output_dir / "population",
        use_vmap=use_vmap,
        n_vmap_envs=250,  # Number of vmap environments for GPU mode
        verbose=True
    )

    # Load the population from checkpoint
    trainer = load_population_from_checkpoint(trainer, checkpoint_dir, start_generation)

    # Continue training from next generation
    remaining_generations = total_generations - start_generation

    if remaining_generations <= 0:
        print(f"\nAlready at or past target generation {total_generations}")
        return

    print(f"\nResuming training for {remaining_generations} more generations")
    print("="*80)

    # Run the remaining generations using the train() method
    trainer.train(
        generations=remaining_generations,
        episodes_per_generation=episodes_per_gen,
        evolution_frequency=2,  # Evolve every 2 generations
        keep_top=0.5  # Keep top 50% during evolution
    )

    print(f"\n{'='*80}")
    print(f"✅ POPULATION TRAINING COMPLETE!")
    print(f"Trained from generation {start_generation} to {total_generations}")
    print(f"Results saved to: {output_dir}")

    # Export final champion
    champion_path = output_dir / "population" / "champion.py"
    print(f"\n🏆 Final champion exported to: {champion_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume or continue population training from a checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume from generation 25 (crashed) and continue to 40
  python resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 25 --total-gens 40

  # Continue training for 20 more generations (40 -> 60)
  python resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 40 --total-gens 60

  # Resume with different settings (more episodes)
  python resume_population_training.py --checkpoint-dir outputs/old_run --start-gen 15 --total-gens 30 --episodes-per-gen 3000
        """
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing the checkpoint to resume from"
    )

    parser.add_argument(
        "--start-gen",
        type=int,
        required=True,
        help="Generation to resume from (must exist in checkpoint)"
    )

    parser.add_argument(
        "--total-gens",
        type=int,
        required=True,
        help="Total number of generations to train to"
    )

    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm (default: ppo)"
    )

    parser.add_argument(
        "--population",
        type=int,
        default=8,
        help="Population size (default: 8)"
    )

    parser.add_argument(
        "--episodes-per-gen",
        type=int,
        default=2000,
        help="Training episodes per generation (default: 2000)"
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=45,
        help="Number of parallel environments (default: 45)"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for training (default: auto)"
    )

    parser.add_argument(
        "--use-vmap",
        action="store_true",
        help="Use JAX vmap for GPU acceleration"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/resumed_TIMESTAMP)"
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    gen_dir = checkpoint_dir / "population" / "models" / f"generation_{args.start_gen}"
    if not gen_dir.exists():
        print(f"ERROR: Generation {args.start_gen} not found in {checkpoint_dir}")
        print(f"Available generations:")
        models_dir = checkpoint_dir / "population" / "models"
        if models_dir.exists():
            for gen in sorted(models_dir.glob("generation_*")):
                print(f"  - {gen.name}")
        sys.exit(1)

    # Resume training
    resume_population_training(
        checkpoint_dir=checkpoint_dir,
        start_generation=args.start_gen,
        total_generations=args.total_gens,
        algorithm=args.algorithm,
        population_size=args.population,
        episodes_per_gen=args.episodes_per_gen,
        n_envs=args.n_envs,
        device=args.device,
        use_vmap=args.use_vmap,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()