#!/usr/bin/env python3
"""
Multi-core population training using the real Arena.
Trains fighters in parallel across CPU cores with proper logging.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging
import multiprocessing as mp
from typing import List, Dict, Optional
import numpy as np
import random
import time

# Add parent directories to path
atom_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(atom_root))
sys.path.insert(0, str(Path(__file__).parent))

# Import gym environment and training utilities
from training.src.gym_env import AtomCombatEnv
from src.protocol.combat_protocol import generate_snapshot
from src.training.signal_engine import build_observation_from_snapshot

# Import fighter loading utility
from fighter_loader import load_hardcoded_fighters

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogger(BaseCallback):
    """Callback to log training progress."""

    def __init__(self, logger, fighter_name, verbose=0):
        super().__init__(verbose)
        self.logger = logger
        self.fighter_name = fighter_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []

    def _on_step(self):
        # Check for completed episodes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                # Check if it was a win
                if "won" in info:
                    self.episode_wins.append(1 if info["won"] else 0)

                # Log every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    win_rate = np.mean(self.episode_wins[-10:]) if self.episode_wins else 0

                    self.logger.info(
                        f"{self.fighter_name} - Episodes: {len(self.episode_rewards)}, "
                        f"Avg Reward: {avg_reward:.2f}, "
                        f"Avg Length: {avg_length:.1f}, "
                        f"Win Rate: {win_rate:.1%}"
                    )
        return True


class PopulationFighter:
    """A fighter in the population."""

    def __init__(self, name: str, model: Optional[PPO] = None):
        self.name = name
        self.model = model
        self.elo = 1500  # Starting ELO
        self.wins = 0
        self.losses = 0
        self.generation = 0
        self.test_results = {}
        self.env_for_decision = None  # Will store an env instance for observation conversion

    def decide(self, snapshot: Dict):
        """Decision function compatible with Arena."""
        if self.model is None:
            # Random actions if no model
            return {
                "acceleration": random.uniform(-4.5, 4.5),
                "stance": random.choice(["neutral", "extended", "defending"])
            }

        # Create a temporary environment if needed for observation conversion
        if self.env_for_decision is None:
            self.env_for_decision = AtomCombatEnv(opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"})
            self.env_for_decision.reset()

        obs = build_observation_from_snapshot(snapshot, recent_damage=0.0)

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=False)

        # Convert action to decision dict (matching gym_env.step logic)
        acceleration_normalized = float(np.clip(action[0], -1.0, 1.0))
        acceleration = acceleration_normalized * 4.5  # max_acceleration

        stance_idx = int(np.clip(action[1], 0, 2))
        stances = ["neutral", "extended", "defending"]
        stance = stances[stance_idx]

        return {"acceleration": acceleration, "stance": stance}


def make_env(opponent_func=None, seed=None):
    """Create environment factory for parallel training."""

    def _init():
        env = AtomCombatEnv(opponent_decision_func=opponent_func)
        env = Monitor(env)
        if seed is not None:
            env.seed(seed)
        return env

    return _init


def test_against_hardcoded(fighter: PopulationFighter, hardcoded_fighters: Dict,
                          matches_per_opponent: int = 5, logger=None):
    """Test a population fighter against hardcoded opponents."""
    results = {}

    for opp_name, opp_func in hardcoded_fighters.items():
        wins = 0

        for match in range(matches_per_opponent):
            # Create environment with hardcoded opponent
            env = AtomCombatEnv(opponent_decision_func=opp_func)
            obs, _ = env.reset()

            done = False
            truncated = False
            while not done and not truncated:
                # Fighter model directly acts on observation
                action, _ = fighter.model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, done, truncated, info = env.step(action)

            if info.get("won", False):
                wins += 1

        win_rate = wins / matches_per_opponent
        results[opp_name] = win_rate

        if logger:
            logger.debug(f"{fighter.name} vs {opp_name}: {win_rate:.1%} ({wins}/{matches_per_opponent})")

    fighter.test_results = results
    return results


def create_output_directory():
    """Create timestamped output directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/home/biff/eng/atom/training/outputs/population_run_{timestamp}")

    # Create subdirectories
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "stats").mkdir(parents=True, exist_ok=True)

    return output_dir


def setup_logging(output_dir: Path):
    """Set up logging to file and console."""
    log_file = output_dir / "logs" / "training.log"

    # Create logger
    logger = logging.getLogger("population_training")
    logger.setLevel(logging.DEBUG)

    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler for important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_config(output_dir: Path, config: Dict):
    """Save training configuration."""
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def main():
    """Main training loop."""
    # Set multiprocessing start method to 'spawn' for compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    print("\n" + "=" * 80)
    print("POPULATION TRAINING WITH REAL ARENA")
    print("=" * 80)

    # Create output directory
    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir}")

    # Set up logging
    logger = setup_logging(output_dir)
    logger.info("Starting population training with real Arena")

    # Configuration
    cpu_count = mp.cpu_count()
    MAX_CORES = 12  # Limit to 12 cores
    N_ENVS = min(MAX_CORES, cpu_count - 4, 8)
    N_ENVS = max(N_ENVS, 1)

    config = {
        "population_size": 8,
        "generations": 20,
        "steps_per_generation": 50000,
        "parallel_envs": N_ENVS,
        "target_win_rate": 0.7,
        "learning_rate": 5e-4,
        "cpu_cores": cpu_count,
        "max_cores_used": MAX_CORES,
        "timestamp": datetime.now().isoformat()
    }

    # Save configuration
    save_config(output_dir, config)

    logger.info(f"System has {cpu_count} CPU cores")
    logger.info(f"Using {N_ENVS} parallel environments per fighter")
    logger.info(f"Population size: {config['population_size']}")
    logger.info(f"Generations: {config['generations']}")
    logger.info(f"Steps per generation: {config['steps_per_generation']}")

    # Load hardcoded fighters for testing
    logger.info("\nLoading hardcoded fighters...")
    try:
        hardcoded_fighters = load_hardcoded_fighters(atom_root, verbose=False)
        logger.info(f"Loaded fighters: {list(hardcoded_fighters.keys())}")
    except Exception as e:
        logger.error(f"Failed to load hardcoded fighters: {e}")
        hardcoded_fighters = {}

    if not hardcoded_fighters:
        logger.warning("No hardcoded fighters loaded - will train with self-play only")

    # Initialize population
    logger.info("\nInitializing population...")
    fighter_names = ["Alpha", "Beta", "Gamma", "Delta", "Echo", "Zeta", "Eta", "Theta"]
    population = []

    for i in range(config["population_size"]):
        name = fighter_names[i] if i < len(fighter_names) else f"Fighter{i}"
        fighter = PopulationFighter(name)

        # Create initial model
        if N_ENVS > 1:
            env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
        else:
            env = DummyVecEnv([make_env()])

        fighter.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=config["learning_rate"],
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            tensorboard_log=str(output_dir / "tensorboard")
        )

        env.close()
        population.append(fighter)
        logger.info(f"  Created {name}")

    # Training loop
    best_overall_score = 0
    start_time = time.time()

    for generation in range(1, config["generations"] + 1):
        gen_start = time.time()
        gen_dir = output_dir / "models" / f"gen_{generation:03d}"
        gen_dir.mkdir(exist_ok=True)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"GENERATION {generation}")
        logger.info(f"{'=' * 60}")

        # Training phase
        logger.info("Training fighters...")

        for fighter in population:
            logger.info(f"  Training {fighter.name}...")

            # Create training environments with mixed opponents
            envs = []

            if hardcoded_fighters and generation > 2:
                # After generation 2, mix in hardcoded opponents
                for i in range(N_ENVS // 2):
                    # Half with hardcoded opponents
                    opp_name = random.choice(list(hardcoded_fighters.keys()))
                    envs.append(make_env(hardcoded_fighters[opp_name]))

                # Half with population opponents
                others = [f for f in population if f != fighter]
                for i in range(N_ENVS - len(envs)):
                    if others:
                        opponent = random.choice(others)
                        envs.append(make_env(opponent.decide))
                    else:
                        envs.append(make_env())  # Random if no others
            else:
                # Early generations: self-play and random
                others = [f for f in population if f != fighter]
                for i in range(N_ENVS):
                    if others and random.random() > 0.3:
                        opponent = random.choice(others)
                        envs.append(make_env(opponent.decide))
                    else:
                        envs.append(make_env())  # Random opponent

            # Create vectorized environment
            if len(envs) > 1:
                vec_env = SubprocVecEnv(envs)
            else:
                vec_env = DummyVecEnv(envs)

            # Train with callback for logging
            callback = TrainingLogger(logger, fighter.name)
            fighter.model.learn(
                total_timesteps=config["steps_per_generation"],
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=False
            )

            vec_env.close()

            # Save model
            model_path = gen_dir / f"{fighter.name.lower()}.zip"
            fighter.model.save(model_path)
            fighter.generation = generation

        # Testing phase
        if hardcoded_fighters:
            logger.info("\nTesting against hardcoded fighters...")

            for fighter in population:
                results = test_against_hardcoded(fighter, hardcoded_fighters, matches_per_opponent=5, logger=logger)

                # Calculate average win rate
                avg_win_rate = np.mean(list(results.values()))
                fighter.elo = 1500 + (avg_win_rate * 1000)  # Simple ELO approximation

                # Log results
                result_str = ", ".join([f"{k[0].upper()}: {v:.0%}" for k, v in results.items()])
                logger.info(f"  {fighter.name}: {result_str} → Average: {avg_win_rate:.0%}")

            # Find best fighter
            best_fighter = max(population, key=lambda f: np.mean(list(f.test_results.values())))
            best_score = np.mean(list(best_fighter.test_results.values()))

            if best_score > best_overall_score:
                best_overall_score = best_score
                # Save as champion
                champion_path = output_dir / "models" / "champion.zip"
                best_fighter.model.save(champion_path)
                logger.info(f"\n✓ New champion: {best_fighter.name} with {best_score:.0%} win rate")

            logger.info(f"\nGeneration {generation} best: {best_score:.0%} (overall best: {best_overall_score:.0%})")

            # Check if target reached
            if best_score >= config["target_win_rate"]:
                logger.info(f"\n🎉 SUCCESS! Reached {config['target_win_rate']:.0%}+ win rate!")
                break

        # Evolution (every 2 generations)
        if generation % 2 == 0 and generation < config["generations"]:
            logger.info("\nEvolution phase...")

            # Sort by performance
            population.sort(key=lambda f: np.mean(list(f.test_results.values())) if f.test_results else 0, reverse=True)

            # Keep top half
            survivors = population[:len(population) // 2]
            logger.info(f"  Survivors: {', '.join([s.name for s in survivors])}")

            # Replace bottom half with mutated versions of top performers
            for i in range(len(population) // 2, len(population)):
                parent = survivors[i % len(survivors)]
                child = population[i]

                # Transfer learned parameters with small mutation
                child.model.set_parameters(parent.model.get_parameters())

                # Small learning rate adjustment as mutation
                new_lr = config["learning_rate"] * random.uniform(0.8, 1.2)
                child.model.learning_rate = new_lr

                logger.info(f"  {child.name} ← {parent.name} (lr: {new_lr:.2e})")

        # Generation summary
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        logger.info(f"\nGeneration {generation} completed in {gen_time:.1f}s (total: {total_time:.1f}s)")

        # Save generation statistics
        stats = {
            "generation": generation,
            "fighters": [
                {
                    "name": f.name,
                    "elo": f.elo,
                    "test_results": f.test_results,
                    "avg_win_rate": np.mean(list(f.test_results.values())) if f.test_results else 0
                }
                for f in population
            ],
            "best_score": best_score if hardcoded_fighters else 0,
            "time": gen_time
        }

        stats_file = output_dir / "stats" / f"gen_{generation:03d}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Best overall score: {best_overall_score:.0%}")
    logger.info(f"Output directory: {output_dir}")

    if best_overall_score >= config["target_win_rate"]:
        logger.info("\n✅ Training successful!")
    else:
        logger.info(f"\n⚠️ Did not reach target win rate of {config['target_win_rate']:.0%}")


if __name__ == "__main__":
    main()
