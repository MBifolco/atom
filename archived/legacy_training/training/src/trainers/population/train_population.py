#!/usr/bin/env python3
"""
Population training using real Arena - stable single-process version.
Trains fighters using the actual game physics.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import random
import time
import funkybob

# Add parent directories to path
atom_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(atom_root))
sys.path.insert(0, str(Path(__file__).parent))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import gym environment and training utilities
from training.src.gym_env import AtomCombatEnv
from src.protocol.combat_protocol import generate_snapshot

# Import fighter loading utility
from fighter_loader import load_hardcoded_fighters

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


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

    def decide(self, snapshot: Dict):
        """Decision function compatible with Arena."""
        if self.model is None:
            # Random actions if no model
            return {
                "acceleration": random.uniform(-4.5, 4.5),
                "stance": random.choice(["neutral", "extended", "retracted", "defending"])
            }

        # Recreate observation from snapshot
        you_hp_norm = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
        you_stamina_norm = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
        opp_hp_norm = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
        opp_stamina_norm = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

        obs = np.array([
            snapshot["you"]["position"],
            snapshot["you"]["velocity"],
            you_hp_norm,
            you_stamina_norm,
            snapshot["opponent"]["distance"],
            snapshot["opponent"]["velocity"],
            opp_hp_norm,
            opp_stamina_norm,
            snapshot["arena"]["width"]
        ], dtype=np.float32)

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=False)

        # Convert action to decision dict
        acceleration_normalized = float(np.clip(action[0], -1.0, 1.0))
        acceleration = acceleration_normalized * 4.5  # max_acceleration

        stance_idx = int(np.clip(action[1], 0, 3))
        stances = ["neutral", "extended", "retracted", "defending"]
        stance = stances[stance_idx]

        return {"acceleration": acceleration, "stance": stance}


def test_against_hardcoded(fighter: PopulationFighter, hardcoded_fighters: Dict,
                          matches_per_opponent: int = 3):
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

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    """Main training loop."""
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
    config = {
        "population_size": 20,
        "generations": 50,
        "steps_per_generation": 100000,
        "target_win_rate": 0.8,
        "learning_rate": 1e-4,
        "timestamp": datetime.now().isoformat()
    }

    # Save configuration
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

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

    # Generate creative fighter names
    def generate_fighter_names(count: int) -> List[str]:
        """Generate unique fighter names using funkybob."""
        name_generator = funkybob.RandomNameGenerator(members=2, separator='_')
        name_iter = iter(name_generator)
        names = []
        seen = set()
        while len(names) < count:
            name = next(name_iter)
            display_name = ' '.join(word.capitalize() for word in name.split('_'))
            if name not in seen:
                names.append(display_name)
                seen.add(name)
        return names

    # Initialize population
    logger.info("\nInitializing population...")
    fighter_names = generate_fighter_names(config["population_size"])
    logger.info(f"Fighter roster: {', '.join(fighter_names)}")
    population = []

    for i in range(config["population_size"]):
        name = fighter_names[i]
        fighter = PopulationFighter(name)

        # Create initial model with single environment
        def make_env():
            return Monitor(AtomCombatEnv(
                opponent_decision_func=lambda _: {"acceleration": random.uniform(-2, 2), "stance": "neutral"}
            ))

        env = DummyVecEnv([make_env])

        fighter.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=config["learning_rate"],
            n_steps=512,
            batch_size=64,
            n_epochs=4
        )

        env.close()
        population.append(fighter)
        logger.info(f"  Created {name}")

    # Training loop
    best_overall_score = 0
    start_time = time.time()
    stats = []  # Initialize stats list to track performance over generations

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

            # Multi-opponent training: Each fighter trains against multiple opponents per generation
            # This provides more diverse experience and faster learning

            # Determine opponent mix based on generation and performance
            if stats:
                # Get best win rate from previous generation
                last_gen_stats = stats[-1]
                best_win_rate = max([f["avg_win_rate"] for f in last_gen_stats["fighters"]] + [0])
            else:
                best_win_rate = 0

            # Calculate steps per opponent (distribute training across multiple opponents)
            if generation <= 5:
                num_opponents = 3  # Train against 3 different opponents
            elif generation <= 10:
                num_opponents = 4  # Train against 4 different opponents
            else:
                num_opponents = 5  # Train against 5 different opponents

            steps_per_opponent = config["steps_per_generation"] // num_opponents

            # Build opponent pool based on curriculum and performance
            opponent_pool = []

            if generation <= 2:
                # Early generations: Easy opponents for basic skills
                easy_opponents = ["dodger", "stamina_manager"]
                available_easy = [op for op in easy_opponents if op in hardcoded_fighters]
                if available_easy:
                    opponent_name = random.choice(available_easy)
                    opponent_func = hardcoded_fighters[opponent_name]
                    logger.debug(f"    Against {opponent_name} (easy)")
                else:
                    # Fallback to random
                    opponent_func = lambda _: {"acceleration": random.uniform(-2, 2), "stance": random.choice(["neutral", "defending"])}
                    logger.debug(f"    Against random (easy)")

            elif generation <= 5:
                # Mid generations: Medium difficulty opponents
                medium_opponents = ["balanced", "counter_puncher", "hit_and_run"]
                available_medium = [op for op in medium_opponents if op in hardcoded_fighters]
                if available_medium:
                    opponent_name = random.choice(available_medium)
                    opponent_func = hardcoded_fighters[opponent_name]
                    logger.debug(f"    Against {opponent_name} (medium)")
                else:
                    # Fallback to self-play
                    others = [f for f in population if f != fighter]
                    if others:
                        opponent = random.choice(others)
                        opponent_func = opponent.decide
                        logger.debug(f"    Against {opponent.name} (self-play)")
                    else:
                        opponent_func = lambda _: {"acceleration": random.uniform(-3, 3), "stance": random.choice(["neutral", "extended"])}
                        logger.debug(f"    Against random (medium)")

            elif generation <= 8:
                # Later generations: Hard opponents
                hard_opponents = ["tank", "rusher", "berserker", "grappler"]
                available_hard = [op for op in hard_opponents if op in hardcoded_fighters]
                if available_hard:
                    opponent_name = random.choice(available_hard)
                    opponent_func = hardcoded_fighters[opponent_name]
                    logger.debug(f"    Against {opponent_name} (hard)")
                elif hardcoded_fighters:
                    opponent_name = random.choice(list(hardcoded_fighters.keys()))
                    opponent_func = hardcoded_fighters[opponent_name]
                    logger.debug(f"    Against {opponent_name} (fallback hard)")
                else:
                    opponent_func = lambda _: {"acceleration": random.uniform(-3, 3), "stance": random.choice(["neutral", "extended"])}
                    logger.debug(f"    Against random (hard fallback)")

            else:
                # Final generations: Mixed training with all opponents
                # 40% hard, 40% from all types, 20% self-play
                rand_val = random.random()
                if rand_val < 0.4 and hardcoded_fighters:
                    # Hard opponents
                    hard_opponents = ["tank", "rusher", "berserker", "grappler", "zoner"]
                    available_hard = [op for op in hard_opponents if op in hardcoded_fighters]
                    if available_hard:
                        opponent_name = random.choice(available_hard)
                        opponent_func = hardcoded_fighters[opponent_name]
                        logger.debug(f"    Against {opponent_name} (mixed-hard)")
                    else:
                        opponent_func = hardcoded_fighters[random.choice(list(hardcoded_fighters.keys()))]
                        logger.debug(f"    Against random hardcoded")
                elif rand_val < 0.8 and hardcoded_fighters:
                    # Any hardcoded opponent
                    opponent_name = random.choice(list(hardcoded_fighters.keys()))
                    opponent_func = hardcoded_fighters[opponent_name]
                    logger.debug(f"    Against {opponent_name} (mixed-any)")
                else:
                    # Self-play for diversity
                    others = [f for f in population if f != fighter]
                    if others:
                        opponent = random.choice(others)
                        opponent_func = opponent.decide
                        logger.debug(f"    Against {opponent.name} (self-play)")
                    else:
                        opponent_func = lambda _: {"acceleration": random.uniform(-4, 4), "stance": random.choice(["neutral", "extended", "retracted"])}
                        logger.debug(f"    Against random (fallback)")

            # Create environment
            def make_env():
                return Monitor(AtomCombatEnv(opponent_decision_func=opponent_func))

            env = DummyVecEnv([make_env])

            # Train
            try:
                fighter.model.learn(
                    total_timesteps=config["steps_per_generation"],
                    progress_bar=False,
                    reset_num_timesteps=False
                )
                logger.info(f"    ✓ Completed {config['steps_per_generation']} steps")
            except Exception as e:
                logger.error(f"    ✗ Training error: {e}")

            env.close()

            # Save model
            model_path = gen_dir / f"{fighter.name.lower()}.zip"
            fighter.model.save(model_path)

        # Testing phase
        if hardcoded_fighters:
            logger.info("\nTesting against hardcoded fighters...")

            for fighter in population:
                results = test_against_hardcoded(fighter, hardcoded_fighters, matches_per_opponent=3)

                # Calculate average win rate
                avg_win_rate = np.mean(list(results.values()))

                # Log results
                result_str = ", ".join([f"{k[0].upper()}: {v:.0%}" for k, v in results.items()])
                logger.info(f"  {fighter.name}: {result_str} → {avg_win_rate:.0%}")

            # Find best fighter
            best_fighter = max(population, key=lambda f: np.mean(list(f.test_results.values())) if f.test_results else 0)
            best_score = np.mean(list(best_fighter.test_results.values())) if best_fighter.test_results else 0

            if best_score > best_overall_score:
                best_overall_score = best_score
                # Save as champion
                champion_path = output_dir / "models" / "champion.zip"
                best_fighter.model.save(champion_path)
                logger.info(f"\n✓ New champion: {best_fighter.name} with {best_score:.0%} win rate")

            logger.info(f"\nBest this generation: {best_score:.0%} (overall best: {best_overall_score:.0%})")

            # Check if target reached
            if best_score >= config["target_win_rate"]:
                logger.info(f"\n🎉 SUCCESS! Reached {config['target_win_rate']:.0%}+ win rate!")
                break

        # Evolution (every 3 generations)
        if generation % 3 == 0 and generation < config["generations"]:
            logger.info("\nEvolution phase...")

            # Sort by performance
            if hardcoded_fighters:
                population.sort(key=lambda f: np.mean(list(f.test_results.values())) if f.test_results else 0, reverse=True)
            else:
                # Random shuffle if no test results
                random.shuffle(population)

            # Keep top half
            survivors = population[:len(population) // 2]
            logger.info(f"  Survivors: {', '.join([s.name for s in survivors])}")

            # Replace bottom half
            for i in range(len(population) // 2, len(population)):
                parent = survivors[i % len(survivors)]
                child = population[i]

                # Transfer learned parameters
                child.model.set_parameters(parent.model.get_parameters())
                logger.info(f"  {child.name} ← {parent.name}")

        # Generation summary
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        logger.info(f"\nGeneration {generation} completed in {gen_time:.1f}s (total: {total_time/60:.1f}m)")

        # Save generation statistics
        gen_stats = {
            "generation": generation,
            "fighters": [
                {
                    "name": f.name,
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
            json.dump(gen_stats, f, indent=2)

        # Track best performer for curriculum decisions
        stats.append(gen_stats)

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

        # Export champion to ONNX and create fighter wrapper
        try:
            logger.info("\nExporting champion to ONNX...")
            from training.src.onnx_fighter import export_to_onnx, create_fighter_wrapper

            champion_sb3_path = output_dir / "models" / "champion.zip"
            onnx_path = output_dir / "champion.onnx"
            wrapper_path = output_dir / "champion.py"

            # Export to ONNX
            export_to_onnx(str(champion_sb3_path), str(onnx_path))

            # Create fighter wrapper
            create_fighter_wrapper(str(onnx_path), str(wrapper_path))

            logger.info(f"✓ Champion exported to ONNX: {onnx_path}")
            logger.info(f"✓ Fighter wrapper created: {wrapper_path}")
            logger.info(f"\nTo use the champion:")
            logger.info(f"  cp {wrapper_path} fighters/")
            logger.info(f"  cp {onnx_path} fighters/")
            logger.info(f"  python atom_fight.py fighters/champion.py fighters/examples/tank.py")

        except Exception as e:
            logger.error(f"Failed to export champion: {e}")
    else:
        logger.info(f"\n⚠️ Did not reach target win rate of {config['target_win_rate']:.0%}")


if __name__ == "__main__":
    main()