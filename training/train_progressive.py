#!/usr/bin/env python3
"""
Progressive Training System for Atom Combat

Combines curriculum learning with population-based training for optimal results.
Fighters first learn through test dummies, then compete in population training.
"""

import sys
from pathlib import Path
# Set paths first before any project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
from datetime import datetime
import shutil
from stable_baselines3 import PPO, SAC

# First import the package to trigger path setup
import src.trainers
# Now import project modules
from src.trainers.curriculum_trainer import CurriculumTrainer
from src.trainers.population.population_trainer import PopulationTrainer
from src.arena import WorldConfig


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
                 verbose: bool = True):
        """
        Initialize the progressive trainer.

        Args:
            algorithm: RL algorithm to use ("ppo" or "sac")
            output_dir: Directory for all outputs
            verbose: Whether to print progress
        """
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.verbose = verbose

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
                              n_envs: int = 4) -> Path:
        """
        Run curriculum training phase.

        Args:
            timesteps: Total training timesteps
            n_envs: Number of parallel environments

        Returns:
            Path to the trained model
        """
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
            print()

        # Create curriculum trainer
        self.curriculum_trainer = CurriculumTrainer(
            algorithm=self.algorithm,
            output_dir=str(self.curriculum_dir),
            n_envs=n_envs,
            verbose=self.verbose
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
            verbose=self.verbose
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
                verbose=self.verbose
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

    def export_best_fighters(self, top_n: int = 3):
        """
        Export the best fighters from population training.

        Args:
            top_n: Number of top fighters to export
        """
        if self.verbose:
            print("\n" + "="*80)
            print("EXPORTING BEST FIGHTERS")
            print("="*80)

        # Find the best models from population training
        models_dir = self.population_dir / "models"
        if not models_dir.exists():
            print("No models found to export.")
            return

        # Get all model files
        model_files = list(models_dir.glob("*.zip"))

        # Sort by generation number (assuming format: fighter_XX_gen_YY.zip)
        def get_generation(path):
            try:
                parts = path.stem.split("_")
                if len(parts) >= 4 and parts[2] == "gen":
                    return int(parts[3])
            except:
                pass
            return 0

        model_files.sort(key=get_generation, reverse=True)

        # Export top models
        export_dir = self.output_dir / "champions"
        export_dir.mkdir(exist_ok=True)

        exported = 0
        for model_path in model_files[:top_n]:
            if exported >= top_n:
                break

            # Copy to export directory
            export_name = f"champion_{exported+1}_{self.timestamp}.zip"
            export_path = export_dir / export_name
            shutil.copy2(model_path, export_path)

            if self.verbose:
                print(f"Exported: {export_name}")

            # Also create a Python wrapper for the champion
            self._create_champion_wrapper(export_path, exported + 1)

            exported += 1

        if self.verbose:
            print(f"\n✅ Exported {exported} champion fighters!")
            print(f"Champions saved to: {export_dir}")

    def _create_champion_wrapper(self, model_path: Path, rank: int):
        """Create a Python wrapper for using the champion fighter."""
        wrapper_code = f'''"""
Champion Fighter #{rank}
Trained using progressive curriculum + population training
Generated: {datetime.now().isoformat()}
"""

from stable_baselines3 import {self.algorithm.upper()}
import numpy as np
from pathlib import Path

# Load the trained model
model_path = Path(__file__).parent / "{model_path.name}"
model = {self.algorithm.upper()}.load(model_path)

def decide(snapshot):
    """
    Decision function for the champion fighter.

    Args:
        snapshot: Game state snapshot

    Returns:
        dict: Decision with acceleration and stance
    """
    # Extract observation from snapshot
    obs = np.array([
        snapshot["you"]["position"],
        snapshot["you"]["velocity"],
        snapshot["opponent"]["position"] - snapshot["you"]["position"],  # relative position
        snapshot["opponent"]["velocity"],
        snapshot["you"]["hp"] / 100.0,
        snapshot["you"]["stamina"] / 100.0,
        snapshot["opponent"]["hp"] / 100.0,
        snapshot["opponent"]["stamina"] / 100.0,
    ])

    # Get action from model
    action, _ = model.predict(obs, deterministic=True)

    # Convert action to game format
    # Assuming action[0] is acceleration, action[1] is stance
    stances = ["neutral", "extended", "defending", "retracted"]

    return {{
        "acceleration": float(np.clip(action[0] * 5.0, -5.0, 5.0)),
        "stance": stances[int(np.clip(action[1], 0, 3))]
    }}
'''
        wrapper_path = model_path.parent / f"champion_{rank}.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)

    def save_training_report(self):
        """Save a comprehensive training report."""
        report = {
            "training_type": "progressive",
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "phases": {
                "curriculum": {
                    "completed": self.curriculum_trainer is not None,
                    "levels_graduated": self.curriculum_trainer.progress.graduated_levels if self.curriculum_trainer else []
                },
                "population": {
                    "completed": self.population_trainer is not None,
                    "generations": self.population_trainer.generation if self.population_trainer else 0,
                    "population_size": self.population_trainer.population_size if self.population_trainer else 0
                }
            },
            "output_dir": str(self.output_dir)
        }

        report_path = self.output_dir / f"training_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\nTraining report saved to: {report_path}")

    def run_complete_pipeline(self,
                             curriculum_timesteps: int = 500_000,
                             population_generations: int = 10,
                             population_size: int = 8):
        """
        Run the complete progressive training pipeline.

        Args:
            curriculum_timesteps: Timesteps for curriculum training
            population_generations: Generations for population training
            population_size: Size of the population
        """
        if self.verbose:
            print("\n" + "🚀"*40)
            print("STARTING PROGRESSIVE TRAINING PIPELINE")
            print("🚀"*40)

        # Phase 1: Curriculum Training
        model_path = self.run_curriculum_training(
            timesteps=curriculum_timesteps,
            n_envs=4
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
            episodes_per_generation=500,
            population_size=population_size,
            keep_top=0.5
        )

        # Export Best Fighters
        self.export_best_fighters(top_n=3)

        # Save Training Report
        self.save_training_report()

        if self.verbose:
            print("\n" + "🏆"*40)
            print("PROGRESSIVE TRAINING COMPLETE!")
            print("🏆"*40)
            print(f"\nResults saved to: {self.output_dir}")


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

  # Population only (requires existing curriculum graduate)
  python train_progressive.py --mode population --population 16 --generations 20

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
        verbose=True
    )

    # Run based on mode
    if args.mode == "quick":
        # Quick test run
        trainer.run_complete_pipeline(
            curriculum_timesteps=10_000,
            population_generations=2,
            population_size=4
        )
    elif args.mode == "curriculum":
        # Curriculum only
        trainer.run_curriculum_training(timesteps=args.timesteps)
    elif args.mode == "population":
        # Population only (assumes curriculum model exists)
        print("Note: This mode requires an existing curriculum graduate model.")
        print("Please ensure you have run curriculum training first.")
        # You would need to specify the model path here
        trainer.run_population_training(
            generations=args.generations,
            episodes_per_generation=500,
            population_size=args.population
        )
    else:  # complete
        # Full pipeline
        trainer.run_complete_pipeline(
            curriculum_timesteps=args.timesteps,
            population_generations=args.generations,
            population_size=args.population
        )


if __name__ == "__main__":
    main()