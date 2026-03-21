#!/usr/bin/env python3
"""
Resume or continue population training from a checkpoint.

This script allows you to:
1. Resume training from a specific generation if interrupted
2. Continue training beyond the original number of generations

Usage:
    # Resume from generation 25 and continue to 40
    python scripts/training/resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 25 --total-gens 40

    # Continue training for 20 more generations after completing 40
    python scripts/training/resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 40 --total-gens 60
    python scripts/training/resume_population_training.py --checkpoint-dir  outputs/progressive_20251112_183943 --start-gen 8 --total-gens 40
"""

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atom.training.pipelines import ProgressiveTrainer
from src.atom.training.trainers.population.population_trainer import PopulationTrainer
from stable_baselines3 import PPO, SAC


def detect_original_gpu_mode(checkpoint_dir: Path) -> tuple[bool, int]:
    """Detect if original training used GPU/vmap by checking log files.

    Returns:
        (use_vmap, n_vmap_envs): True if GPU was used, and number of vmap envs
    """
    # Check population training log
    log_files = list((checkpoint_dir / "population" / "logs").glob("*.log"))

    if not log_files:
        return False, 0

    # Read the most recent log file
    log_file = sorted(log_files)[-1]

    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "GPU Acceleration: ENABLED" in line:
                    # Extract vmap env count from line like "GPU Acceleration: ENABLED (vmap with 45 envs)"
                    import re
                    match = re.search(r'vmap with (\d+) envs', line)
                    n_envs = int(match.group(1)) if match else 250
                    return True, n_envs
                elif "GPU Acceleration: DISABLED" in line or "Parallel Fighters:" in line:
                    # If we see parallel fighters line without GPU acceleration, it's CPU mode
                    continue
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")

    return False, 0


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
        from src.atom.training.gym_env import AtomCombatEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
            opponent_decision_func=lambda _: {"acceleration": 0, "stance": "neutral"},
            max_ticks=250
        ))])

        # Load the model
        if trainer.algorithm == "ppo":
            model = PPO.load(model_file, env=env)
        else:
            model = SAC.load(model_file, env=env)

        # Create fighter object
        from src.atom.training.trainers.population.population_trainer import PopulationFighter
        fighter = PopulationFighter(
            name=fighter_name,
            model=model,
            generation=generation,
            lineage=f"G{generation}",  # Lineage from generation
            mass=70.0,  # Default mass
            last_checkpoint=str(model_file)
        )

        trainer.population.append(fighter)
        # Also add to ELO tracker to avoid duplicates later
        trainer.elo_tracker.add_fighter(fighter_name)

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
    seed: int = 1337,
    population_size: int = 8,
    episodes_per_gen: int = 2000,
    n_envs: int = 45,
    use_vmap: bool = False,
    force_cpu: bool = False,
    n_parallel_fighters: int = None,
    output_dir: str = None,
    keep_top: float = 0.5,
    mutation_rate: float = 0.1,
    evolution_frequency: int = 2,
    record_replays: bool = False,
    replay_frequency: int = 5
):
    """Resume or continue population training from a checkpoint.

    Uses ProgressiveTrainer to ensure same code paths as train_progressive.py
    """

    # Detect original training configuration
    original_use_vmap, original_n_vmap_envs = detect_original_gpu_mode(checkpoint_dir)

    # Handle force_cpu override
    if force_cpu:
        print("\n" + "="*80)
        print("🚫 FORCING CPU MODE (--force-cpu flag)")
        print("="*80)
        if original_use_vmap:
            print(f"Original training used GPU (vmap with {original_n_vmap_envs} envs)")
            print("Overriding to CPU mode - this will be ~77x slower!")
        use_vmap = False
        print("="*80 + "\n")

    # Warn if there's a mismatch in GPU mode (but only if not force_cpu)
    elif original_use_vmap and not use_vmap:
        print("\n" + "="*80)
        print("⚠️  WARNING: GPU MODE MISMATCH DETECTED!")
        print("="*80)
        print(f"Original training used GPU acceleration (vmap with {original_n_vmap_envs} envs)")
        print("Current settings: CPU mode (no --use-vmap flag)")
        print("\nThis will be MUCH SLOWER (~77x slower)!")
        print("\nTo resume with GPU acceleration, add --use-vmap flag:")
        print(f"  python scripts/training/resume_population_training.py --checkpoint-dir {checkpoint_dir} \\")
        print(f"         --start-gen {start_generation} --total-gens {total_generations} --use-vmap")
        print("\nAuto-enabling GPU mode to match original training...")
        print("="*80 + "\n")
        use_vmap = True
        if original_n_vmap_envs > 0:
            n_envs = original_n_vmap_envs
    elif not original_use_vmap and use_vmap:
        print("\n" + "="*80)
        print("ℹ️  INFO: Original training used CPU mode, but --use-vmap was specified")
        print("="*80)
        print("Continuing with GPU mode as requested (this may give different results)")
        print("="*80 + "\n")

    # Use existing checkpoint directory or create new output directory
    if output_dir is None:
        output_dir = checkpoint_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint data if using new directory
        if checkpoint_dir != output_dir:
            print(f"\nCopying checkpoint data to {output_dir}")
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

    # Create ProgressiveTrainer instance (reuses same code as train_progressive.py)
    print(f"\nInitializing ProgressiveTrainer...")
    print(f"  Algorithm: {algorithm}")
    print(f"  Seed: {seed}")
    print(f"  GPU acceleration: {'ENABLED' if use_vmap else 'DISABLED'}")
    if use_vmap:
        print(f"  Vmap environments: {n_envs}")

    progressive_trainer = ProgressiveTrainer(
        output_dir=str(output_dir),
        algorithm=algorithm,
        seed=seed,
        use_vmap=use_vmap,
        n_parallel_fighters=n_parallel_fighters,
        verbose=True
    )

    # Create population trainer using same method as train_progressive.py
    progressive_trainer.population_trainer = PopulationTrainer(
        population_size=population_size,
        algorithm=algorithm,
        output_dir=str(output_dir / "population"),
        verbose=True,
        seed=seed,
        n_parallel_fighters=n_parallel_fighters,
        use_vmap=use_vmap,
        n_vmap_envs=n_envs if use_vmap else 45,
        n_envs_per_fighter=n_envs if not use_vmap else 1,  # Critical: vmap handles parallelization
        record_replays=record_replays,
        replay_recording_frequency=replay_frequency
    )

    # Load the population from checkpoint
    progressive_trainer.population_trainer = load_population_from_checkpoint(
        progressive_trainer.population_trainer,
        checkpoint_dir,
        start_generation
    )

    # Continue training from next generation
    remaining_generations = total_generations - start_generation

    if remaining_generations <= 0:
        print(f"\nAlready at or past target generation {total_generations}")
        return

    print(f"\nResuming training for {remaining_generations} more generations")
    print(f"Evolution Settings:")
    print(f"  Keep top: {keep_top*100:.0f}% of population")
    print(f"  Mutation rate: {mutation_rate} (weight noise level)")
    print(f"  Evolution frequency: Every {evolution_frequency} generations")
    print("="*80)

    # Run remaining generations using same method as train_progressive.py
    # This ensures we use the exact same training logic and parameters
    progressive_trainer.run_population_training(
        generations=remaining_generations,
        episodes_per_generation=episodes_per_gen,
        population_size=population_size,
        keep_top=keep_top,
        evolution_frequency=evolution_frequency,
        mutation_rate=mutation_rate
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
  python scripts/training/resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 25 --total-gens 40

  # Continue training for 20 more generations (40 -> 60)
  python scripts/training/resume_population_training.py --checkpoint-dir outputs/progressive_20251112_085705 --start-gen 40 --total-gens 60

  # Resume with different settings (more episodes)
  python scripts/training/resume_population_training.py --checkpoint-dir outputs/old_run --start-gen 15 --total-gens 30 --episodes-per-gen 3000
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
        "--seed",
        type=int,
        default=1337,
        help="Global training seed (default: 1337)"
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
        help="Number of parallel environments / vmap batch size (default: 45)"
    )

    parser.add_argument(
        "--use-vmap",
        action="store_true",
        help="Use JAX vmap for GPU acceleration"
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode even if checkpoint was trained with GPU (disables auto-detection)"
    )

    parser.add_argument(
        "--n-parallel-fighters",
        type=int,
        default=None,
        help="Number of fighters to train in parallel (default: auto - 2 for GPU, cpu_count-1 for CPU)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/resumed_TIMESTAMP)"
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
        help="Mutation strength for evolved fighters (default: 0.1)"
    )

    parser.add_argument(
        "--evolution-frequency",
        type=int,
        default=2,
        help="Evolve population every N generations (default: 2)"
    )

    parser.add_argument(
        "--record-replays",
        action="store_true",
        help="Record fight replays for creating a training montage (samples bottom/middle/top spectacle)"
    )

    parser.add_argument(
        "--replay-frequency",
        type=int,
        default=5,
        help="Record replays every N generations for population training (default: 5)"
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
        seed=args.seed,
        population_size=args.population,
        episodes_per_gen=args.episodes_per_gen,
        n_envs=args.n_envs,
        use_vmap=args.use_vmap,
        force_cpu=args.force_cpu,
        n_parallel_fighters=args.n_parallel_fighters,
        output_dir=args.output_dir,
        keep_top=args.keep_top,
        mutation_rate=args.mutation_rate,
        evolution_frequency=args.evolution_frequency,
        record_replays=args.record_replays,
        replay_frequency=args.replay_frequency
    )


if __name__ == "__main__":
    main()
