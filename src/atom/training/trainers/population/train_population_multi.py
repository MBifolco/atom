#!/usr/bin/env python3
"""
Population-based training for Atom Combat with multi-opponent training and self-play emphasis.

This version implements:
- Multi-opponent training per generation (3-5 opponents)
- Progressive transition from hardcoded opponents to self-play
- Emphasis on self-play once fundamentals are learned
- Creative fighter names using funkybob
"""

import sys
import random
import time
import json
from pathlib import Path
from typing import List, Optional, Dict
import logging

# Add parent directory if not already in path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Training imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Atom imports
from src.atom.training.gym_env import AtomCombatEnv
from src.atom.training.trainers.population.population_fighter import PopulationFighter
from src.atom.training.trainers.population.fighter_loader import load_hardcoded_fighters, FighterLoadError

# Name generation
import funkybob

# Configuration
DEFAULT_CONFIG = {
    "population_size": 8,            # Increased for more diversity
    "generations": 30,                # More generations for deep learning
    "steps_per_generation": 30000,   # More steps per generation
    "target_win_rate": 0.75,         # Higher target for excellence
    "learning_rate": 0.0001,          # Lower for stability
    "self_play_threshold": 0.5,      # Start heavy self-play at 50% win rate
}

def generate_fighter_names(count: int) -> List[str]:
    """Generate unique fighter names using funkybob."""
    name_generator = funkybob.RandomNameGenerator(members=2, separator='_')
    name_iter = iter(name_generator)

    names = []
    seen = set()

    while len(names) < count:
        name = next(name_iter)
        # Capitalize first letter of each word and replace underscores with spaces for display
        display_name = ' '.join(word.capitalize() for word in name.split('_'))
        # Keep internal name with underscores for file paths
        if name not in seen:
            names.append(display_name)
            seen.add(name)

    return names


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    log_file = output_dir / "logs" / "training.log"
    log_file.parent.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("population_training")
    logger.setLevel(logging.DEBUG)

    # File handler - logs everything
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler - only important stuff
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Simple formatter for console
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def test_fighter(fighter, hardcoded_fighters, test_matches=1):
    """Test a fighter against all hardcoded opponents."""
    results = {}

    for opponent_name, opponent_func in hardcoded_fighters.items():
        wins = 0

        for _ in range(test_matches):
            env = AtomCombatEnv(opponent_decision_func=opponent_func)
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = fighter.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if info.get("won"):
                wins += 1

        results[opponent_name] = wins / test_matches

    return results


def select_training_opponents(
    generation: int,
    best_win_rate: float,
    fighter: PopulationFighter,
    population: List[PopulationFighter],
    hardcoded_fighters: Dict,
    num_opponents: int
) -> List[tuple]:
    """
    Select diverse opponents for multi-opponent training.
    Returns list of (opponent_func, opponent_name) tuples.
    """
    opponents = []

    # Phase 1: Learn fundamentals (generations 1-5)
    if generation <= 5:
        # Focus on hardcoded opponents to learn basics
        if generation <= 2:
            # Start with easier opponents
            preference_order = ["dodger", "stamina_manager", "balanced", "hit_and_run"]
        else:
            # Add harder opponents
            preference_order = ["balanced", "tank", "rusher", "counter_puncher", "berserker"]

        # Select from available hardcoded fighters
        available = [name for name in preference_order if name in hardcoded_fighters]

        for i in range(num_opponents):
            if i < len(available):
                name = available[i % len(available)]
                opponents.append((hardcoded_fighters[name], f"hardcoded:{name}"))
            else:
                # Fallback to random if not enough hardcoded
                opponents.append((
                    lambda _: {"acceleration": random.uniform(-3, 3), "stance": random.choice(["neutral", "extended"])},
                    "random"
                ))

    # Phase 2: Transition phase (generations 6-10 or until 50% win rate)
    elif generation <= 10 or best_win_rate < DEFAULT_CONFIG["self_play_threshold"]:
        # Mix of hardcoded and self-play
        others = [f for f in population if f != fighter]

        # 60% hardcoded, 40% self-play
        num_hardcoded = int(num_opponents * 0.6)
        num_selfplay = num_opponents - num_hardcoded

        # Add hardcoded opponents (focus on ones we struggle with)
        all_hardcoded = list(hardcoded_fighters.keys())
        for i in range(num_hardcoded):
            name = random.choice(all_hardcoded)
            opponents.append((hardcoded_fighters[name], f"hardcoded:{name}"))

        # Add self-play opponents
        for i in range(num_selfplay):
            if others:
                opponent = random.choice(others)
                opponents.append((opponent.decide, f"self-play:{opponent.name}"))
            else:
                # No other fighters yet, use random
                opponents.append((
                    lambda _: {"acceleration": random.uniform(-3, 3), "stance": random.choice(["neutral", "extended", "retracted"])},
                    "random"
                ))

    # Phase 3: Advanced training through self-play (after beating hardcoded)
    else:
        # Heavy self-play with occasional hardcoded for grounding
        others = [f for f in population if f != fighter]

        if best_win_rate >= 0.7:
            # 80% self-play, 20% hardcoded when doing well
            num_selfplay = int(num_opponents * 0.8)
            num_hardcoded = num_opponents - num_selfplay
        else:
            # 60% self-play, 40% hardcoded when struggling
            num_selfplay = int(num_opponents * 0.6)
            num_hardcoded = num_opponents - num_selfplay

        # Add self-play opponents (prioritize strong fighters)
        if others:
            # Sort by performance if we have past stats
            strong_others = sorted(others, key=lambda f: random.random())  # Random for diversity

            for i in range(num_selfplay):
                opponent = strong_others[i % len(strong_others)]
                opponents.append((opponent.decide, f"self-play:{opponent.name}"))

        # Add hardcoded opponents for grounding
        if num_hardcoded > 0 and hardcoded_fighters:
            # Focus on diverse fighter types
            diverse_fighters = ["tank", "dodger", "counter_puncher", "berserker", "zoner"]
            available_diverse = [name for name in diverse_fighters if name in hardcoded_fighters]

            for i in range(num_hardcoded):
                if available_diverse:
                    name = available_diverse[i % len(available_diverse)]
                else:
                    name = random.choice(list(hardcoded_fighters.keys()))
                opponents.append((hardcoded_fighters[name], f"hardcoded:{name}"))

    # Add past champion self-play (mirror match) for highest skill development
    if generation > 15 and best_win_rate > 0.6:
        # Replace one opponent with mirror match (playing against itself)
        if opponents:
            opponents[-1] = (fighter.decide, f"mirror:{fighter.name}")

    return opponents


def main():
    """Main training loop with multi-opponent training."""
    config = DEFAULT_CONFIG

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"training/outputs/population_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        config["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(config, f, indent=2)

    # Setup logging
    logger = setup_logging(output_dir)

    # Banner
    logger.info("\n" + "=" * 80)
    logger.info("ATOM COMBAT - POPULATION TRAINING WITH MULTI-OPPONENT & SELF-PLAY")
    logger.info("=" * 80)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: {config}")
    logger.info("")

    # Load hardcoded fighters
    try:
        hardcoded_fighters = load_hardcoded_fighters(".", verbose=False)
        logger.info(f"Loaded {len(hardcoded_fighters)} hardcoded fighters: {list(hardcoded_fighters.keys())}")
    except Exception as e:
        logger.warning(f"Failed to load some hardcoded fighters: {e}")
        hardcoded_fighters = {}

    # Generate creative fighter names
    fighter_names = generate_fighter_names(config["population_size"])
    logger.info(f"\nGenerated fighter names: {', '.join(fighter_names)}")

    # Initialize population
    logger.info("\nInitializing population...")
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
            learning_rate=config["learning_rate"],
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            verbose=0,
            tensorboard_log=str(output_dir / "tensorboard" / name)
        )

        population.append(fighter)
        logger.info(f"  Created {name}")

    # Training loop
    stats = []
    best_overall = 0
    start_time = time.time()

    for generation in range(1, config["generations"] + 1):
        gen_start = time.time()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"GENERATION {generation}")
        logger.info(f"{'=' * 60}")

        # Get best performance from previous generation
        best_win_rate = max([s.avg_win_rate for s in stats] + [0]) if stats else 0
        logger.info(f"Previous best win rate: {best_win_rate:.1%}")

        # Training phase with multi-opponent
        logger.info("\nTraining fighters with multi-opponent curriculum...")

        for fighter in population:
            # Determine number of opponents for this generation
            if generation <= 5:
                num_opponents = 3
            elif generation <= 10:
                num_opponents = 4
            else:
                num_opponents = 5

            steps_per_opponent = config["steps_per_generation"] // num_opponents

            # Select diverse opponents
            opponents = select_training_opponents(
                generation, best_win_rate, fighter,
                population, hardcoded_fighters, num_opponents
            )

            logger.info(f"\n  {fighter.name} training plan ({num_opponents} opponents, {steps_per_opponent} steps each):")
            for _, opp_name in opponents:
                logger.info(f"    - {opp_name}")

            # Train against each opponent
            for opponent_func, opponent_name in opponents:
                logger.debug(f"    Training against {opponent_name}...")

                # Create environment for this opponent
                def make_env():
                    return Monitor(AtomCombatEnv(opponent_decision_func=opponent_func))

                env = DummyVecEnv([make_env])

                # Train
                try:
                    fighter.model.learn(
                        total_timesteps=steps_per_opponent,
                        progress_bar=False,
                        reset_num_timesteps=False
                    )
                except Exception as e:
                    logger.error(f"    Training error: {e}")

            logger.info(f"    ✓ Completed {config['steps_per_generation']} total steps")

        # Testing phase
        logger.info("\nTesting against hardcoded fighters...")
        gen_stats = []

        for fighter in population:
            # Test against all hardcoded fighters
            results = test_fighter(fighter, hardcoded_fighters)
            avg_win_rate = sum(results.values()) / len(results) if results else 0

            # Create fighter stats
            fighter_stats = type('obj', (object,), {
                'name': fighter.name,
                'test_results': results,
                'avg_win_rate': avg_win_rate
            })()

            gen_stats.append(fighter_stats)

            # Log results
            result_str = ", ".join([f"{k[0].upper()}: {int(v*100)}%" for k, v in results.items()])
            logger.info(f"  {fighter.name}: {result_str} → {int(avg_win_rate*100)}%")

        # Find champion
        champion = max(gen_stats, key=lambda x: x.avg_win_rate)
        champion_fighter = next(f for f in population if f.name == champion.name)

        if champion.avg_win_rate > best_overall:
            best_overall = champion.avg_win_rate
            logger.info(f"\n✓ New champion: {champion.name} with {int(champion.avg_win_rate*100)}% win rate")

            # Save champion
            model_dir = output_dir / "models" / f"gen_{generation:03d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            champion_fighter.model.save(str(model_dir / "champion"))

        # Save generation stats
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(exist_ok=True)

        with open(stats_dir / f"gen_{generation:03d}_stats.json", "w") as f:
            json.dump({
                "generation": generation,
                "fighters": [
                    {
                        "name": s.name,
                        "test_results": s.test_results,
                        "avg_win_rate": s.avg_win_rate
                    }
                    for s in gen_stats
                ],
                "best_score": champion.avg_win_rate,
                "time": time.time() - gen_start
            }, f, indent=2)

        stats.append(champion)

        # Progress update
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start
        logger.info(f"\nBest this generation: {int(champion.avg_win_rate*100)}% (overall best: {int(best_overall*100)}%)")
        logger.info(f"Generation {generation} completed in {gen_time:.1f}s (total: {elapsed/60:.1f}m)")

        # Check graduation condition
        if champion.avg_win_rate >= config["target_win_rate"]:
            logger.info(f"\n🎉 SUCCESS! Reached {int(config['target_win_rate']*100)}%+ win rate!")

            # Export champion
            logger.info("\nExporting champion to ONNX...")
            from training.src.onnx_fighter import export_to_onnx, create_fighter_wrapper

            champion_sb3_path = output_dir / "models" / f"gen_{generation:03d}" / "champion.zip"
            onnx_path = output_dir / "champion.onnx"
            wrapper_path = output_dir / "champion.py"

            export_to_onnx(str(champion_sb3_path), str(onnx_path))
            create_fighter_wrapper(str(onnx_path), str(wrapper_path))

            logger.info(f"✓ Champion exported to ONNX: {onnx_path}")
            logger.info(f"✓ Fighter wrapper created: {wrapper_path}")
            break

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Best overall score: {int(best_overall*100)}%")
    logger.info(f"Output directory: {output_dir}")

    if best_overall >= config["target_win_rate"]:
        logger.info("\n✅ Training successful!")
    else:
        logger.info(f"\n⚠️ Did not reach target of {int(config['target_win_rate']*100)}% (got {int(best_overall*100)}%)")

    logger.info("\nTo use the champion:")
    logger.info(f"  cp {output_dir}/champion.py fighters/")
    logger.info(f"  cp {output_dir}/champion.onnx fighters/")
    logger.info("  python atom_fight.py fighters/champion.py fighters/examples/boxer.py")


if __name__ == "__main__":
    main()
