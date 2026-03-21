"""Progressive training pipeline orchestration."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path for script-style execution contexts.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atom.training.utils.determinism import set_global_seeds
from src.atom.training.utils.runtime_platform import configure_runtime_gpu_env

# Configure CUDA/ROCm/CPU defaults before importing modules that may touch JAX.
DETECTED_RUNTIME_PLATFORM = configure_runtime_gpu_env(enable_gpu=True, memory_fraction=0.75)

from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer
from src.atom.training.trainers.population.population_trainer import PopulationTrainer

# Phase 2: Try SBX (JAX) first, fall back to SB3 (PyTorch).
# SBX requires JAX < 0.7.0, incompatible with ROCm JAX 0.7.1.
try:
    from sbx import PPO, SAC  # noqa: F401
    USING_SBX = True
except ImportError:
    from stable_baselines3 import PPO, SAC  # noqa: F401
    USING_SBX = False

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
                 n_envs: int = None,
                 max_ticks: int = 250,
                 device: str = "auto",
                 use_vmap: bool = False,
                 population_cpu_only: bool = False,
                 debug: bool = False,
                 record_replays: bool = False,
                 replay_frequency: int = 5,
                 override_episodes_per_level: int = None,
                 checkpoint_interval: int = 100000,
                 seed: int = 1337):
        """
        Initialize the progressive trainer.

        Args:
            algorithm: RL algorithm to use ("ppo" or "sac")
            output_dir: Directory for all outputs
            verbose: Whether to print progress
            n_parallel_fighters: Number of fighters to train in parallel in population mode (default: 2 for GPU, cpu_count-1 for CPU)
            n_envs: Number of parallel environments for PPO/SAC training (default: 8 for CPU, 250 for GPU)
            max_ticks: Maximum ticks per episode (default: 250)
            device: Device to use for training ("cpu", "cuda", or "auto")
            use_vmap: Use JAX vmap for GPU-accelerated training (Level 3/4)
            record_replays: Whether to record fight replays for montage
            replay_frequency: Record replays every N generations
            seed: Training seed for reproducible runs
        """
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.n_parallel_fighters = n_parallel_fighters
        self.n_envs = n_envs
        self.max_ticks = max_ticks
        self.device = device
        self.use_vmap = use_vmap
        self.population_cpu_only = population_cpu_only
        self.debug = debug
        self.record_replays = record_replays
        self.replay_frequency = replay_frequency
        self.override_episodes_per_level = override_episodes_per_level
        self.checkpoint_interval = checkpoint_interval
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        self.seed_report = set_global_seeds(self.seed)

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
                              n_envs: int = None,
                              resume_from_latest: bool = False) -> Path:
        """
        Run curriculum training phase.

        Args:
            timesteps: Total training timesteps
            n_envs: Number of parallel environments (default: from init or 8 for CPU, 250 for GPU)
            resume_from_latest: Resume from latest curriculum checkpoint bundle.

        Returns:
            Path to the trained model
        """
        # Use instance n_envs if not overridden, otherwise set default based on vmap usage
        if n_envs is None:
            if self.n_envs is not None:
                n_envs = self.n_envs
            else:
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
            debug=self.debug,
            record_replays=self.record_replays,
            override_episodes_per_level=self.override_episodes_per_level,
            checkpoint_interval=self.checkpoint_interval,
            seed=self.seed,
        )

        # Train through curriculum
        self.curriculum_trainer.train(
            total_timesteps=timesteps,
            resume_from_latest=resume_from_latest,
        )

        # Get the trained model path
        model_path = self.curriculum_dir / "models" / "curriculum_graduate.zip"

        # COMPREHENSIVE CLEANUP - Free all GPU/CPU memory from curriculum training
        print("\n💾 Cleaning up curriculum trainer resources...")

        # 1. Close environments properly (this now calls vmap_env.close())
        if hasattr(self.curriculum_trainer, 'envs') and self.curriculum_trainer.envs is not None:
            try:
                self.curriculum_trainer.envs.close()
                print("  ✓ Closed curriculum environments")
            except Exception as e:
                print(f"  ⚠ Error closing environments: {e}")
            finally:
                del self.curriculum_trainer.envs
                self.curriculum_trainer.envs = None

        # 2. Delete the PPO model and its components
        if hasattr(self.curriculum_trainer, 'model') and self.curriculum_trainer.model is not None:
            # Delete policy network
            if hasattr(self.curriculum_trainer.model, 'policy'):
                del self.curriculum_trainer.model.policy

            # Delete value network
            if hasattr(self.curriculum_trainer.model, 'value_net'):
                del self.curriculum_trainer.model.value_net

            # Delete rollout buffer
            if hasattr(self.curriculum_trainer.model, 'rollout_buffer'):
                del self.curriculum_trainer.model.rollout_buffer

            # Delete the model itself
            del self.curriculum_trainer.model
            self.curriculum_trainer.model = None
            print("  ✓ Deleted PPO model")

        # 3. Delete progress tracking and statistics
        if hasattr(self.curriculum_trainer, 'progress'):
            del self.curriculum_trainer.progress
            self.curriculum_trainer.progress = None

        if hasattr(self.curriculum_trainer, 'stats'):
            del self.curriculum_trainer.stats
            self.curriculum_trainer.stats = None

        # 4. Delete the entire curriculum trainer
        del self.curriculum_trainer
        self.curriculum_trainer = None
        print("  ✓ Deleted curriculum trainer")

        # 5. Force garbage collection (Python)
        import gc
        gc.collect()
        gc.collect()  # Run twice to ensure cleanup
        print("  ✓ Ran garbage collection")

        # 6. Clear PyTorch GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all GPU operations to complete

                # Get memory stats
                if self.verbose:
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    print(f"  ✓ Cleared PyTorch GPU cache")
                    print(f"    GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ PyTorch GPU cleanup: {e}")

        # 7. Clear JAX resources
        try:
            import jax
            jax.clear_caches()

            # Also clear JAX GPU memory if using GPU
            try:
                devices = jax.devices()
                if any('gpu' in str(d).lower() or 'rocm' in str(d).lower() for d in devices):
                    # Force JAX to release GPU memory
                    jax.clear_backends()
                    print("  ✓ Cleared JAX GPU resources")
            except:
                pass

            print("  ✓ Cleared JAX compilation cache")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ JAX cleanup: {e}")

        # 8. ROCm/AMD GPU specific cleanup
        try:
            # Try to reset ROCm if available
            import subprocess
            result = subprocess.run(['rocm-smi', '--resetclocks'],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                print("  ✓ Reset ROCm GPU clocks")
        except:
            pass  # ROCm tools may not be available

        # 9. Brief pause to ensure all resources are released
        import time
        time.sleep(1)
        print("  ✓ Resource cleanup complete\n")

        if self.verbose:
            print(f"✅ Curriculum training complete!")
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
        # NOTE: Population training uses GPU with reduced envs to fit memory
        # 8 fighters × 45 envs = 360 total envs (~7.2 GB VRAM)
        self.population_trainer = PopulationTrainer(
            population_size=population_size,
            algorithm=self.algorithm,
            output_dir=str(self.population_dir),
            max_ticks=self.max_ticks,
            verbose=self.verbose,
            n_parallel_fighters=self.n_parallel_fighters,
            use_vmap=self.use_vmap,  # Use GPU if enabled
            n_vmap_envs=45,  # Reduced from 250 to fit 8 parallel fighters in 8GB VRAM
            record_replays=self.record_replays,
            replay_recording_frequency=self.replay_frequency,
            seed=self.seed,
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
                              keep_top: float = 0.5,
                              evolution_frequency: int = 2,
                              mutation_rate: float = 0.1):
        """
        Run population-based training phase.

        Args:
            generations: Number of generations to evolve
            episodes_per_generation: Training episodes per generation
            population_size: Size of the population
            keep_top: Fraction to keep during evolution
            evolution_frequency: Evolve every N generations
            mutation_rate: Strength of mutations (noise level for weights)
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 3: POPULATION TRAINING")
            print("="*80)
            print(f"Evolving population through {generations} generations...")
            print(f"Population size: {population_size}")
            print(f"Episodes per generation: {episodes_per_generation}")
            print(f"Selection pressure: Keep top {keep_top*100:.0f}%")
            print(f"Mutation rate: {mutation_rate} (weight noise level)")
            print(f"Evolution frequency: Every {evolution_frequency} generations")
            print()

        if not self.population_trainer:
            # Create population trainer if not already created
            # Determine whether to use GPU for population training
            population_use_vmap = self.use_vmap and not self.population_cpu_only

            if self.population_cpu_only and self.use_vmap and self.verbose:
                print("⚠️  Forcing CPU mode for population training (--population-cpu-only)")
                print("   (This avoids GPU out-of-memory issues)")

            self.population_trainer = PopulationTrainer(
                population_size=population_size,
                algorithm=self.algorithm,
                output_dir=str(self.population_dir),
                max_ticks=self.max_ticks,
                verbose=self.verbose,
                n_parallel_fighters=self.n_parallel_fighters,
                use_vmap=population_use_vmap,  # May be overridden by population_cpu_only
                n_vmap_envs=45,  # Reduced from 250 to fit 8 parallel fighters in 8GB VRAM
                record_replays=self.record_replays,
                replay_recording_frequency=self.replay_frequency,
                seed=self.seed,
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
            evolution_frequency=evolution_frequency,
            mutation_rate=mutation_rate,
            base_model_path=base_model
        )

        if self.verbose:
            print("\n✅ Population training complete!")

    def run_complete_pipeline(self,
                             curriculum_timesteps: int = 500_000,
                             population_generations: int = 10,
                             population_size: int = 8,
                             episodes_per_generation: int = 2000,
                             keep_top: float = 0.5,
                             evolution_frequency: int = 2,
                             mutation_rate: float = 0.1,
                             resume_curriculum: bool = False):
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
            print(f"  Training Backend: {'SBX (JAX)' if USING_SBX else 'SB3 (PyTorch)'}")
            print(f"  GPU Acceleration: {'Enabled (vmap)' if self.use_vmap else 'Disabled'}")
            print(f"  Runtime Platform: {DETECTED_RUNTIME_PLATFORM}")
            print(f"  Seed: {self.seed}")
            print(f"\nOutput directory: {self.output_dir}")
            print(f"Logs will be saved to:")
            print(f"  - {self.curriculum_dir / 'logs'}")
            print(f"  - {self.population_dir / 'logs'}")
            print()

        # Phase 1: Curriculum Training
        model_path = self.run_curriculum_training(
            timesteps=curriculum_timesteps,
            resume_from_latest=resume_curriculum,
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
            keep_top=keep_top,
            evolution_frequency=evolution_frequency,
            mutation_rate=mutation_rate
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

