#!/usr/bin/env python3
"""
Progressive Training System for Atom Combat

Run this from the project root directory.
Combines curriculum learning with population-based training.

Usage:
    python train_progressive.py --mode quick
    python train_progressive.py --mode complete --timesteps 1000000
"""

import sys
from pathlib import Path

# Ensure we're running from project root
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import cleanly from src
from src.training.trainers.curriculum_trainer import CurriculumTrainer
from src.training.trainers.population.population_trainer import PopulationTrainer
from src.arena import WorldConfig

# Phase 2: Try SBX (JAX) first, fall back to SB3 (PyTorch)
# SBX requires JAX < 0.7.0, incompatible with ROCm JAX 0.7.1
try:
    from sbx import PPO, SAC  # JAX-accelerated (if available)
    _using_sbx = True
except ImportError:
    from stable_baselines3 import PPO, SAC  # PyTorch fallback
    _using_sbx = False

import argparse
import json
from datetime import datetime
import shutil


class ProgressiveTrainer:
    """
    Manages the complete progressive training pipeline.

    Training Stages:
    1. Curriculum Learning: Train against test dummies with increasing difficulty
    2. Population Initialization: Create diverse population from curriculum graduates
    3. Population Evolution: Population-based training with self-play
    """

    def __init__(self,
                 algorithm: str = "ppo",
                 output_dir: str = "outputs/progressive",
                 verbose: bool = True,
                 n_parallel_fighters: int = None,
                 max_ticks: int = 250,
                 device: str = "auto",
                 use_vmap: bool = False,
                 debug: bool = False):
        """
        Initialize the progressive trainer.

        Args:
            algorithm: RL algorithm to use ("ppo" or "sac")
            output_dir: Directory for all outputs
            verbose: Whether to print progress
            n_parallel_fighters: Number of fighters to train in parallel (default: cpu_count - 1)
            max_ticks: Maximum ticks per episode (default: 250)
            device: Device to use for training ("cpu", "cuda", or "auto")
            use_vmap: Use JAX vmap for GPU-accelerated training (Level 3/4)
        """
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.n_parallel_fighters = n_parallel_fighters
        self.max_ticks = max_ticks
        self.device = device
        self.use_vmap = use_vmap
        self.debug = debug

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_dir = self.output_dir / "curriculum"
        self.population_dir = self.output_dir / "population"

        # Training components
        self.curriculum_trainer = None
        self.population_trainer = None

        # Timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_curriculum_training(self,
                              timesteps: int = 500_000,
                              n_envs: int = None) -> Path:
        """
        Run curriculum training phase.

        Args:
            timesteps: Total training timesteps
            n_envs: Number of parallel environments (default: 8 for CPU, 250 for GPU)

        Returns:
            Path to the trained model
        """
        # Set default n_envs based on vmap usage
        if n_envs is None:
            n_envs = 250 if self.use_vmap else 8

        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 1: CURRICULUM TRAINING")
            print("="*80)
            print("Training fighter through progressive difficulty levels...")
            print(f"- Level 1: Stationary targets (fundamentals)")
            print(f"- Level 2: Simple movements (basic skills)")
            print(f"- Level 3: Distance/stamina management (intermediate)")
            print(f"- Level 4: Behavioral fighters (advanced)")
            print(f"- Level 5: Hardcoded fighters (expert)")
            if self.use_vmap:
                print(f"\n⚡ GPU Acceleration: ENABLED (vmap with {n_envs} parallel environments)")
            else:
                print(f"\n💻 CPU Training: {n_envs} parallel environments")
            print()

        # Create curriculum trainer
        self.curriculum_trainer = CurriculumTrainer(
            algorithm=self.algorithm,
            output_dir=str(self.curriculum_dir),
            n_envs=n_envs,
            max_ticks=self.max_ticks,
            verbose=self.verbose,
            device=self.device,
            use_vmap=self.use_vmap,
            debug=self.debug
        )

        # Train through curriculum
        self.curriculum_trainer.train(total_timesteps=timesteps)

        # Get the trained model path
        model_path = self.curriculum_dir / "models" / "curriculum_graduate.zip"

        if self.verbose:
            print(f"\n✅ Curriculum training complete!")
            print(f"Model saved to: {model_path}")

        return model_path

    def initialize_population_from_curriculum(self,
                                            curriculum_model_path: Path,
                                            population_size: int = 8,
                                            variation_factor: float = 0.1) -> None:
        """
        Initialize a population from a curriculum-trained model.

        Creates variations of the trained model to seed population diversity.

        Args:
            curriculum_model_path: Path to the curriculum graduate model
            population_size: Number of fighters in population
            variation_factor: How much to vary the initial models (0-1)
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 2: POPULATION INITIALIZATION")
            print("="*80)
            print(f"Creating population of {population_size} fighters from curriculum graduate...")

        # Create population trainer
        self.population_trainer = PopulationTrainer(
            population_size=population_size,
            algorithm=self.algorithm,
            output_dir=str(self.population_dir),
            max_ticks=self.max_ticks,
            verbose=self.verbose,
            n_parallel_fighters=self.n_parallel_fighters
        )

        # Initialize population with the curriculum model as base
        self.population_trainer.initialize_population(
            base_model_path=str(curriculum_model_path),
            variation_factor=variation_factor
        )

        if self.verbose:
            print(f"\n✅ Population initialized with {population_size} fighters!")

    def run_population_training(self,
                              generations: int = 10,
                              episodes_per_generation: int = 500,
                              population_size: int = 8,
                              keep_top: float = 0.5):
        """
        Run population-based training phase.

        Args:
            generations: Number of generations to evolve
            episodes_per_generation: Training episodes per generation
            population_size: Size of the population
            keep_top: Fraction to keep during evolution
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 3: POPULATION TRAINING")
            print("="*80)
            print(f"Evolving population through {generations} generations...")
            print(f"Population size: {population_size}")
            print(f"Episodes per generation: {episodes_per_generation}")
            print(f"Selection pressure: Keep top {keep_top*100:.0f}%")
            print()

        if not self.population_trainer:
            # Create population trainer if not already created
            self.population_trainer = PopulationTrainer(
                population_size=population_size,
                algorithm=self.algorithm,
                output_dir=str(self.population_dir),
                max_ticks=self.max_ticks,
                verbose=self.verbose,
                n_parallel_fighters=self.n_parallel_fighters
            )

        # Check if we have a base model from curriculum training
        base_model = None
        curriculum_model = self.curriculum_dir / "models" / "curriculum_graduate.zip"
        if curriculum_model.exists():
            base_model = str(curriculum_model)

        # Run population training
        self.population_trainer.train(
            generations=generations,
            episodes_per_generation=episodes_per_generation,
            keep_top=keep_top,
            evolution_frequency=2,  # Evolve every 2 generations
            base_model_path=base_model
        )

        if self.verbose:
            print("\n✅ Population training complete!")

    def run_complete_pipeline(self,
                             curriculum_timesteps: int = 500_000,
                             population_generations: int = 10,
                             population_size: int = 8,
                             episodes_per_generation: int = 2000):
        """
        Run the complete progressive training pipeline.

        Args:
            curriculum_timesteps: Timesteps for curriculum training
            population_generations: Generations for population training
            population_size: Size of the population
            episodes_per_generation: Training episodes per generation
        """
        if self.verbose:
            print("\n" + "🚀"*40)
            print("STARTING PROGRESSIVE TRAINING PIPELINE")
            print("🚀"*40)
            print(f"\nConfiguration:")
            print(f"  Training Backend: {'SBX (JAX)' if _using_sbx else 'SB3 (PyTorch)'}")
            print(f"  GPU Acceleration: {'Enabled (vmap)' if self.use_vmap else 'Disabled'}")
            print(f"\nOutput directory: {self.output_dir}")
            print(f"Logs will be saved to:")
            print(f"  - {self.curriculum_dir / 'logs'}")
            print(f"  - {self.population_dir / 'logs'}")
            print()

        # Phase 1: Curriculum Training
        model_path = self.run_curriculum_training(
            timesteps=curriculum_timesteps
            # n_envs defaults to 8 for CPU or 250 for GPU (auto-configured)
        )

        # Phase 2: Initialize Population
        self.initialize_population_from_curriculum(
            curriculum_model_path=model_path,
            population_size=population_size,
            variation_factor=0.1
        )

        # Phase 3: Population Training
        self.run_population_training(
            generations=population_generations,
            episodes_per_generation=episodes_per_generation,
            population_size=population_size,
            keep_top=0.5
        )

        if self.verbose:
            print("\n" + "🏆"*40)
            print("PROGRESSIVE TRAINING COMPLETE!")
            print("🏆"*40)
            print(f"\nResults saved to: {self.output_dir}")
            print("\nLog Files:")
            print(f"  Curriculum: {self.curriculum_dir / 'logs'}")
            print(f"  Population: {self.population_dir / 'logs'}")
            print("\nTrained Models:")
            print(f"  Curriculum graduate: {self.curriculum_dir / 'models' / 'curriculum_graduate.zip'}")
            print(f"  Population models: {self.population_dir / 'models'}")
            print("\nTo review training progress:")
            print(f"  tail -f {self.curriculum_dir / 'logs'}/*.log")
            print(f"  tail -f {self.population_dir / 'logs'}/*.log")


def main():
    """Main entry point for progressive training."""
    parser = argparse.ArgumentParser(
        description="Progressive training for Atom Combat fighters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train_progressive.py --mode quick --algorithm ppo

  # Full curriculum only
  python train_progressive.py --mode curriculum --timesteps 1000000

  # Complete pipeline
  python train_progressive.py --mode complete --timesteps 2000000 --population 8 --generations 10
        """
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "curriculum", "population", "complete"],
        default="complete",
        help="Training mode"
    )
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Timesteps for curriculum training"
    )
    parser.add_argument(
        "--population",
        
        type=int,
        default=8,
        help="Population size"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Population training generations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/progressive_TIMESTAMP)"
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel fighter training (default: cpu_count - 1)"
    )
    parser.add_argument(
        "--episodes-per-gen",
        type=int,
        default=2000,
        help="Training episodes per generation for population training (default: 2000)"
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=250,
        help="Maximum ticks per episode (default: 250, ~21 seconds)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Device to use for training: cpu, cuda (GPU), or auto (default: cuda for ROCm)"
    )
    parser.add_argument(
        "--use-vmap",
        action="store_true",
        help="Enable JAX vmap for GPU-accelerated training (77x speedup with GPU). Requires: source setup_gpu.sh"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to see detailed fight information"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/progressive_{timestamp}"
    else:
        output_dir = args.output_dir

    # Create trainer
    trainer = ProgressiveTrainer(
        algorithm=args.algorithm,
        output_dir=output_dir,
        verbose=True,
        n_parallel_fighters=args.cores,
        max_ticks=args.max_ticks,
        device=args.device,
        use_vmap=args.use_vmap,
        debug=args.debug
    )

    # Run based on mode
    if args.mode == "quick":
        # Quick test run
        trainer.run_complete_pipeline(
            curriculum_timesteps=10_000,
            population_generations=2,
            population_size=4,
            episodes_per_generation=500  # Keep low for quick mode
        )
    elif args.mode == "curriculum":
        # Curriculum only
        trainer.run_curriculum_training(timesteps=args.timesteps)
    elif args.mode == "population":
        # Population only
        print("Note: Population mode requires an existing curriculum graduate model.")
        trainer.run_population_training(
            generations=args.generations,
            episodes_per_generation=args.episodes_per_gen,
            population_size=args.population
        )
    else:  # complete
        # Full pipeline
        trainer.run_complete_pipeline(
            curriculum_timesteps=args.timesteps,
            population_generations=args.generations,
            population_size=args.population,
            episodes_per_generation=args.episodes_per_gen
        )


if __name__ == "__main__":
    main()
