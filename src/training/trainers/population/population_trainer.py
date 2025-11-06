"""
Population-Based Training for Atom Combat

Trains a population of fighters simultaneously, allowing them to learn
from each other and evolve diverse strategies.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import logging
from datetime import datetime
import time
import random
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Clean relative imports within src package structure
from ....arena import WorldConfig
from ...gym_env import AtomCombatEnv
from .elo_tracker import EloTracker


@dataclass
class PopulationFighter:
    """Represents a fighter in the population."""
    name: str
    model: PPO  # or SAC
    generation: int = 0
    lineage: str = "founder"
    mass: float = 70.0
    training_episodes: int = 0
    last_checkpoint: Optional[str] = None


class PopulationCallback(BaseCallback):
    """Callback for tracking population training progress."""

    def __init__(self, fighter_name: str, elo_tracker: EloTracker, verbose: int = 0):
        super().__init__(verbose)
        self.fighter_name = fighter_name
        self.elo_tracker = elo_tracker
        self.episode_count = 0
        self.recent_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                reward = info["episode"]["r"]
                self.recent_rewards.append(reward)

                # Keep only last 100 episodes
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)

        return True


class PopulationTrainer:
    """
    Manages population-based training for Atom Combat fighters.

    Key features:
    - Trains multiple fighters simultaneously
    - Fighters learn by competing against each other
    - Tracks performance with ELO ratings
    - Implements evolution/selection mechanics
    - Supports parallel training across CPU cores
    """

    def __init__(self,
                 population_size: int = 8,
                 config: WorldConfig = None,
                 algorithm: str = "ppo",
                 output_dir: str = "outputs/population",
                 n_envs_per_fighter: int = 2,
                 max_ticks: int = 1000,
                 mass_range: Tuple[float, float] = (60.0, 85.0),
                 verbose: bool = True,
                 export_threshold: float = 0.5):
        """
        Initialize the population trainer.

        Args:
            population_size: Number of fighters in the population
            config: World configuration
            algorithm: "ppo" or "sac"
            output_dir: Directory for saving models and logs
            n_envs_per_fighter: Parallel environments per fighter
            max_ticks: Maximum ticks per episode
            mass_range: Range of masses for fighter variety
            verbose: Whether to print progress
            export_threshold: Minimum win rate to export fighters to AIs directory
        """
        self.population_size = population_size
        self.config = config or WorldConfig()
        self.algorithm = algorithm.lower()
        self.output_dir = Path(output_dir)
        self.n_envs_per_fighter = n_envs_per_fighter
        self.max_ticks = max_ticks
        self.mass_range = mass_range
        self.verbose = verbose
        self.export_threshold = export_threshold

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize population
        self.population: List[PopulationFighter] = []
        self.elo_tracker = EloTracker()

        # Training state
        self.generation = 0
        self.total_matches = 0

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for population training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"population_training_{timestamp}.log"

        self.logger = logging.getLogger('population_trainer')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.propagate = False

        self.logger.info("="*80)
        self.logger.info("POPULATION TRAINING INITIALIZED")
        self.logger.info(f"Population Size: {self.population_size}")
        self.logger.info(f"Algorithm: {self.algorithm}")
        self.logger.info(f"Mass Range: {self.mass_range}")
        self.logger.info("="*80)

    def _create_fighter_name(self, index: int, generation: int = 0) -> str:
        """Generate a unique name for a fighter."""
        prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Echo", "Zeta", "Eta", "Theta",
                   "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"]

        if index < len(prefixes):
            base_name = prefixes[index]
        else:
            base_name = f"Fighter{index}"

        if generation > 0:
            return f"{base_name}_G{generation}"
        return base_name

    def initialize_population(self, base_model_path: Optional[str] = None, variation_factor: float = 0.1):
        """
        Initialize the population with new fighters.

        Args:
            base_model_path: Optional path to a pre-trained model to use as base
            variation_factor: How much to vary models if using base model (0-1)
        """
        if self.verbose:
            print(f"\nInitializing population of {self.population_size} fighters...")
            if base_model_path:
                print(f"  Using base model: {base_model_path}")
                print(f"  Variation factor: {variation_factor}")

        for i in range(self.population_size):
            name = self._create_fighter_name(i, self.generation)
            mass = np.random.uniform(*self.mass_range)

            # Create a simple environment for model initialization
            env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
                opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
                config=self.config,
                max_ticks=self.max_ticks,
                fighter_mass=mass,
                opponent_mass=70.0
            ))])

            # Create or load model
            if base_model_path and Path(base_model_path).exists():
                # Load from base model
                if self.algorithm == "ppo":
                    model = PPO.load(base_model_path, env=env)
                else:  # SAC
                    model = SAC.load(base_model_path, env=env)

                # Add variation for diversity (except first fighter)
                if i > 0 and variation_factor > 0:
                    self._add_variation_to_model(model, variation_factor)

            else:
                # Create new model from scratch
                if self.algorithm == "ppo":
                    model = PPO(
                        "MlpPolicy",
                        env,
                        verbose=0,
                        learning_rate=1e-4,
                        n_steps=2048 // self.n_envs_per_fighter,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.01  # Encourage exploration
                    )
                else:  # SAC
                    model = SAC(
                        "MlpPolicy",
                        env,
                        verbose=0,
                        learning_rate=3e-4,
                        buffer_size=50000,
                        learning_starts=100,
                        batch_size=256,
                        tau=0.005,
                        gamma=0.99
                    )

            fighter = PopulationFighter(
                name=name,
                model=model,
                generation=self.generation,
                mass=mass
            )

            self.population.append(fighter)
            self.elo_tracker.add_fighter(name)

            if self.verbose:
                print(f"  Created {name} (mass: {mass:.1f}kg)")

        self.logger.info(f"Initialized {len(self.population)} fighters")

    def _add_variation_to_model(self, model, variation_factor: float):
        """
        Add random variation to model parameters to increase diversity.

        Args:
            model: The model to vary
            variation_factor: How much variation to add (0-1)
        """
        import torch

        # Get policy parameters
        policy_params = model.policy.state_dict()

        # Add noise to weights and biases
        for key, value in policy_params.items():
            if 'weight' in key or 'bias' in key:
                # Add gaussian noise proportional to parameter magnitude
                noise = torch.randn_like(value) * variation_factor * 0.1
                value.data += value.data * noise

        # Update the model with varied parameters
        model.policy.load_state_dict(policy_params)

    def _get_fighter_decision_func(self, fighter: PopulationFighter) -> Callable:
        """Create a decision function for a trained fighter."""
        def decide(snapshot):
            # Convert snapshot to observation
            obs = np.array([
                snapshot["you"]["position"],
                snapshot["you"]["velocity"],
                snapshot["you"]["hp"] / snapshot["you"]["max_hp"],
                snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"],
                snapshot["opponent"]["distance"],
                snapshot["opponent"]["velocity"],
                snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"],
                snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"],
                snapshot["arena"]["width"]
            ], dtype=np.float32)

            # Get action from model
            action, _ = fighter.model.predict(obs, deterministic=False)

            # Convert continuous action to game action
            acceleration = float(action[0]) * 4.5  # Scale from [-1, 1] to [-4.5, 4.5]
            stance_idx = int(action[1])
            stances = ["neutral", "extended", "retracted", "defending"]
            stance = stances[min(stance_idx, 3)]

            return {"acceleration": acceleration, "stance": stance}

        return decide

    def create_matchmaking_pairs(self) -> List[Tuple[PopulationFighter, PopulationFighter]]:
        """
        Create pairs of fighters for training matches.

        Uses multiple strategies:
        1. Similar skill matches (based on ELO)
        2. Random matches for diversity
        3. Top vs bottom for teaching
        """
        pairs = []
        available_fighters = self.population.copy()
        random.shuffle(available_fighters)

        # Strategy 1: Similar skill pairs (50% of matches)
        balanced_matches = self.elo_tracker.suggest_balanced_matches(len(available_fighters) // 4)
        for name_a, name_b in balanced_matches:
            fighter_a = next((f for f in self.population if f.name == name_a), None)
            fighter_b = next((f for f in self.population if f.name == name_b), None)
            if fighter_a and fighter_b:
                pairs.append((fighter_a, fighter_b))

        # Strategy 2: Random pairs for the rest
        used_fighters = set()
        for pair in pairs:
            used_fighters.add(pair[0].name)
            used_fighters.add(pair[1].name)

        remaining = [f for f in available_fighters if f.name not in used_fighters]
        random.shuffle(remaining)

        for i in range(0, len(remaining) - 1, 2):
            pairs.append((remaining[i], remaining[i+1]))

        return pairs

    def train_fighter_batch(self,
                           fighter: PopulationFighter,
                           opponents: List[PopulationFighter],
                           episodes: int = 100) -> Dict:
        """
        Train a single fighter against a batch of opponents.

        Returns training statistics.
        """
        # Create environments with different opponents
        env_fns = []
        opponent_names = []

        for i in range(self.n_envs_per_fighter):
            opponent = opponents[i % len(opponents)]
            opponent_names.append(opponent.name)

            env_fn = lambda opp=opponent, fighter_mass=fighter.mass, opp_mass=opponent.mass: Monitor(
                AtomCombatEnv(
                    opponent_decision_func=self._get_fighter_decision_func(opp),
                    config=self.config,
                    max_ticks=self.max_ticks,
                    fighter_mass=fighter_mass,
                    opponent_mass=opp_mass
                )
            )
            env_fns.append(env_fn)

        # Create vectorized environment
        vec_env = SubprocVecEnv(env_fns) if self.n_envs_per_fighter > 1 else DummyVecEnv(env_fns)

        # Update the model's environment
        fighter.model.set_env(vec_env)

        # Create callback
        callback = PopulationCallback(fighter.name, self.elo_tracker)

        # Calculate timesteps
        avg_ticks_per_episode = 100
        total_timesteps = episodes * avg_ticks_per_episode

        # Train
        fighter.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            reset_num_timesteps=False  # Continue from previous training
        )

        fighter.training_episodes += episodes

        # Return statistics
        stats = {
            "fighter": fighter.name,
            "episodes": callback.episode_count,
            "mean_reward": np.mean(callback.recent_rewards) if callback.recent_rewards else 0,
            "opponents": opponent_names
        }

        vec_env.close()

        return stats

    def run_evaluation_matches(self, num_matches_per_pair: int = 3) -> None:
        """
        Run evaluation matches between all fighters to update ELO ratings.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("EVALUATION MATCHES")
            print("="*60)

        # Get all unique pairs
        pairs = []
        for i in range(len(self.population)):
            for j in range(i+1, len(self.population)):
                pairs.append((self.population[i], self.population[j]))

        random.shuffle(pairs)

        # Run matches
        for fighter_a, fighter_b in pairs:
            wins_a = 0
            wins_b = 0
            total_damage_a = 0
            total_damage_b = 0

            for _ in range(num_matches_per_pair):
                # Create environment for the match
                env = AtomCombatEnv(
                    opponent_decision_func=self._get_fighter_decision_func(fighter_b),
                    config=self.config,
                    max_ticks=self.max_ticks,
                    fighter_mass=fighter_a.mass,
                    opponent_mass=fighter_b.mass
                )

                # Run match
                obs, _ = env.reset()
                done = False

                while not done:
                    action, _ = fighter_a.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                # Record results
                if info.get("won"):
                    wins_a += 1
                elif info.get("opponent_hp", 0) <= 0:
                    wins_b += 1

                total_damage_a += info.get("episode_damage_dealt", 0)
                total_damage_b += info.get("episode_damage_taken", 0)

                env.close()

            # Determine overall result
            if wins_a > wins_b:
                result = "a_wins"
            elif wins_b > wins_a:
                result = "b_wins"
            else:
                result = "draw"

            # Update ELO ratings
            new_elo_a, new_elo_b = self.elo_tracker.update_ratings(
                fighter_a.name,
                fighter_b.name,
                result,
                total_damage_a / num_matches_per_pair,
                total_damage_b / num_matches_per_pair,
                {"generation": self.generation}
            )

            if self.verbose:
                result_str = f"{fighter_a.name} wins" if result == "a_wins" else \
                           (f"{fighter_b.name} wins" if result == "b_wins" else "Draw")
                print(f"  {fighter_a.name} ({new_elo_a:.0f}) vs {fighter_b.name} ({new_elo_b:.0f}): {result_str}")

            self.total_matches += num_matches_per_pair

    def evolve_population(self, keep_top: float = 0.5, mutation_rate: float = 0.1) -> None:
        """
        Evolve the population by replacing weak fighters with mutations of strong ones.

        Args:
            keep_top: Fraction of population to keep
            mutation_rate: How much to vary the cloned models
        """
        if self.verbose:
            print("\n" + "="*60)
            print("POPULATION EVOLUTION")
            print("="*60)

        rankings = self.elo_tracker.get_rankings()
        keep_count = max(2, int(len(self.population) * keep_top))

        # Identify survivors and those to replace
        survivors = []
        to_replace = []

        for fighter in self.population:
            fighter_rank = next((i for i, stats in enumerate(rankings) if stats.name == fighter.name), -1)
            if fighter_rank < keep_count and fighter_rank >= 0:
                survivors.append(fighter)
            else:
                to_replace.append(fighter)

        if self.verbose:
            print(f"  Keeping top {len(survivors)} fighters")
            print(f"  Replacing {len(to_replace)} fighters")

        # Replace weak fighters with mutations of strong ones
        self.generation += 1

        for i, old_fighter in enumerate(to_replace):
            # Select a parent from survivors (weighted by ELO)
            parent = random.choice(survivors)

            # Create new fighter name
            new_name = self._create_fighter_name(
                self.population.index(old_fighter),
                self.generation
            )

            # Vary the mass slightly
            mass_variation = np.random.uniform(-5, 5)
            new_mass = np.clip(parent.mass + mass_variation, *self.mass_range)

            # Clone the parent's model
            env = DummyVecEnv([lambda: Monitor(AtomCombatEnv(
                opponent_decision_func=lambda s: {"acceleration": 0, "stance": "neutral"},
                config=self.config,
                max_ticks=self.max_ticks,
                fighter_mass=new_mass,
                opponent_mass=70.0
            ))])

            if self.algorithm == "ppo":
                new_model = PPO.load(
                    parent.last_checkpoint if parent.last_checkpoint else parent.model,
                    env=env
                )
                # Apply mutation by slightly changing learning rate
                new_model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))
            else:  # SAC
                new_model = SAC.load(
                    parent.last_checkpoint if parent.last_checkpoint else parent.model,
                    env=env
                )
                new_model.learning_rate *= (1 + np.random.uniform(-mutation_rate, mutation_rate))

            # Create new fighter
            new_fighter = PopulationFighter(
                name=new_name,
                model=new_model,
                generation=self.generation,
                lineage=f"{parent.name}→{new_name}",
                mass=new_mass
            )

            # Replace in population
            idx = self.population.index(old_fighter)
            self.population[idx] = new_fighter

            # Add to ELO tracker (starts at default ELO)
            self.elo_tracker.add_fighter(new_name)

            if self.verbose:
                print(f"    Replaced {old_fighter.name} with {new_name} (child of {parent.name})")

        self.logger.info(f"Evolved to generation {self.generation}")

    def save_population(self, min_win_rate: float = None) -> None:
        """
        Save all fighters in the population.

        Also exports qualifying fighters to fighters/AIs/ directory in atom_fight.py format.

        Args:
            min_win_rate: Minimum win rate to export to AIs directory (default: use export_threshold)
        """
        if min_win_rate is None:
            min_win_rate = self.export_threshold
        generation_dir = self.models_dir / f"generation_{self.generation}"
        generation_dir.mkdir(exist_ok=True)

        for fighter in self.population:
            model_path = generation_dir / f"{fighter.name}.zip"
            fighter.model.save(model_path)
            fighter.last_checkpoint = str(model_path)

        # Save ELO rankings
        rankings_file = generation_dir / "rankings.txt"
        with open(rankings_file, 'w') as f:
            f.write(f"Generation {self.generation} Rankings\n")
            f.write("="*60 + "\n")
            for i, stats in enumerate(self.elo_tracker.get_rankings(), 1):
                f.write(f"{i}. {stats.name}: ELO={stats.elo:.0f}, "
                       f"Record={stats.wins}-{stats.losses}-{stats.draws}\n")

        # Export qualifying fighters to AIs directory
        self._export_qualifying_fighters(min_win_rate)

        if self.verbose:
            print(f"\nSaved generation {self.generation} to {generation_dir}")

    def _export_qualifying_fighters(self, min_win_rate: float = 0.5) -> None:
        """
        Export fighters that meet win rate threshold to fighters/AIs/ directory.

        Each fighter gets its own folder with:
        - {fighter_name}.onnx - ONNX model for inference
        - {fighter_name}.py - Python wrapper with decide() function
        - README.md - Fighter stats and metadata

        Args:
            min_win_rate: Minimum win rate to qualify for export
        """
        # Calculate win rates for all fighters
        rankings = self.elo_tracker.get_rankings()

        # Determine project root (where atom_fight.py is)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        ais_dir = project_root / "fighters" / "AIs"
        ais_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0

        for fighter in self.population:
            # Get fighter stats
            stats = next((s for s in rankings if s.name == fighter.name), None)
            if not stats:
                continue

            # Calculate win rate
            total_matches = stats.wins + stats.losses + stats.draws
            if total_matches == 0:
                win_rate = 0.0
            else:
                win_rate = (stats.wins + 0.5 * stats.draws) / total_matches

            # Check if fighter qualifies
            if win_rate >= min_win_rate:
                try:
                    self._export_fighter_to_ais(fighter, stats, win_rate, ais_dir)
                    exported_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to export {fighter.name}: {e}")
                    if self.verbose:
                        print(f"  ⚠️  Failed to export {fighter.name}: {e}")

        if self.verbose and exported_count > 0:
            print(f"\n✅ Exported {exported_count} qualifying fighters to {ais_dir}")

    def _export_fighter_to_ais(self,
                                fighter: PopulationFighter,
                                stats,
                                win_rate: float,
                                ais_dir: Path) -> None:
        """
        Export a single fighter to the AIs directory.

        Creates a folder with ONNX model, Python wrapper, and README.
        """
        # Create fighter directory
        fighter_dir = ais_dir / fighter.name
        fighter_dir.mkdir(parents=True, exist_ok=True)

        # Export model to ONNX
        onnx_path = fighter_dir / f"{fighter.name}.onnx"
        self._export_model_to_onnx(fighter.model, onnx_path)

        # Create Python wrapper
        py_path = fighter_dir / f"{fighter.name}.py"
        self._create_fighter_wrapper(fighter, py_path, onnx_path.name)

        # Create README
        readme_path = fighter_dir / "README.md"
        self._create_fighter_readme(fighter, stats, win_rate, readme_path)

        if self.verbose:
            print(f"  📦 Exported {fighter.name} (WR: {win_rate:.1%}, ELO: {stats.elo:.0f}) to {fighter_dir}")

    def _export_model_to_onnx(self, model, output_path: Path) -> None:
        """Export a Stable-Baselines3 model to ONNX format."""
        import torch

        # Get the policy network
        policy = model.policy

        # Create dummy input (observation space)
        dummy_input = torch.zeros(1, 9, dtype=torch.float32)

        # Export to ONNX
        torch.onnx.export(
            policy,
            dummy_input,
            str(output_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=12
        )

    def _create_fighter_wrapper(self, fighter: PopulationFighter, output_path: Path, onnx_filename: str) -> None:
        """Create a Python wrapper file with decide() function for atom_fight.py."""
        template = f'''"""
{fighter.name} - Trained AI Fighter

Generation: {fighter.generation}
Lineage: {fighter.lineage}
Mass: {fighter.mass:.1f}kg
Training Episodes: {fighter.training_episodes}

Auto-generated wrapper for trained ONNX model.
Compatible with atom_fight.py
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# ONNX model path (relative to this file)
ONNX_MODEL = "{onnx_filename}"

# Global session (loaded once)
_session = None
_stance_names = ["neutral", "extended", "retracted", "defending"]


def _load_session():
    """Load ONNX session (lazy loading)."""
    global _session
    if _session is None:
        model_path = Path(__file__).parent / ONNX_MODEL
        _session = ort.InferenceSession(str(model_path))
    return _session


def decide(snapshot):
    """
    Decision function for trained fighter.

    Args:
        snapshot: Combat snapshot from the arena
            - you: dict with position, velocity, hp, max_hp, stamina, max_stamina
            - opponent: dict with distance, velocity, hp, max_hp, stamina, max_stamina
            - arena: dict with width

    Returns:
        dict with:
            - acceleration: float (-4.5 to +4.5)
            - stance: str ("neutral", "extended", "retracted", "defending")
    """
    session = _load_session()

    # Convert snapshot to observation
    you = snapshot["you"]
    opponent = snapshot["opponent"]
    arena = snapshot["arena"]

    you_hp_norm = you["hp"] / you["max_hp"]
    you_stamina_norm = you["stamina"] / you["max_stamina"]
    opp_hp_norm = opponent["hp"] / opponent["max_hp"]
    opp_stamina_norm = opponent["stamina"] / opponent["max_stamina"]

    obs = np.array([
        you["position"],
        you["velocity"],
        you_hp_norm,
        you_stamina_norm,
        opponent["distance"],
        opponent["velocity"],
        opp_hp_norm,
        opp_stamina_norm,
        arena["width"]
    ], dtype=np.float32).reshape(1, -1)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {{input_name: obs}})

    # Parse action
    # Action space is Box: [acceleration_normalized, stance_selector]
    action = outputs[0][0]
    acceleration_normalized = np.clip(action[0], -1.0, 1.0)
    stance_idx = int(np.clip(action[1], 0, 3))

    # Scale acceleration (max_acceleration = 4.5)
    acceleration = float(acceleration_normalized * 4.5)
    stance = _stance_names[stance_idx]

    return {{"acceleration": acceleration, "stance": stance}}
'''

        with open(output_path, 'w') as f:
            f.write(template)

    def _create_fighter_readme(self,
                               fighter: PopulationFighter,
                               stats,
                               win_rate: float,
                               output_path: Path) -> None:
        """Create a README.md file with fighter stats and usage info."""
        total_matches = stats.wins + stats.losses + stats.draws

        readme = f'''# {fighter.name}

Trained AI Fighter from Population-Based Training

## Stats

- **Generation**: {fighter.generation}
- **Lineage**: {fighter.lineage}
- **Mass**: {fighter.mass:.1f}kg
- **Training Episodes**: {fighter.training_episodes}

### Performance Metrics

- **ELO Rating**: {stats.elo:.0f}
- **Win Rate**: {win_rate:.1%}
- **Record**: {stats.wins}W - {stats.losses}L - {stats.draws}D
- **Total Matches**: {total_matches}

## Usage

This fighter is compatible with `atom_fight.py`:

```bash
# Fight against another AI
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/rusher.py

# Watch the fight in terminal
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/tank.py --watch

# Generate HTML replay
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/balanced.py --html replay.html

# Custom mass (if different from trained mass)
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/rusher.py --mass-a {fighter.mass:.0f}
```

## Files

- `{fighter.name}.py` - Python wrapper with decide() function
- `{fighter.name}.onnx` - ONNX model (neural network weights)
- `README.md` - This file

## Requirements

```bash
pip install onnxruntime numpy
```

## Strategy

This fighter learned its strategy through population-based training, competing against
other evolving AI fighters. Its behavior emerged from reinforcement learning rather than
being hand-coded.

**Training Algorithm**: {self.algorithm.upper()}
**Population Size**: {self.population_size}
**Generation**: {self.generation}

## Notes

- The fighter was trained at {fighter.mass:.1f}kg mass. Performance may vary with different masses.
- Win rate of {win_rate:.1%} was achieved against the training population.
- The ONNX model requires `onnxruntime` to run inference.
'''

        with open(output_path, 'w') as f:
            f.write(readme)

    def train(self,
             generations: int = 10,
             episodes_per_generation: int = 500,
             evolution_frequency: int = 2,
             base_model_path: Optional[str] = None,
             keep_top: float = 0.5) -> None:
        """
        Run the full population training loop.

        Args:
            generations: Number of generations to evolve
            episodes_per_generation: Training episodes per generation
            evolution_frequency: Evolve every N generations
            base_model_path: Optional path to pre-trained model for initialization
            keep_top: Fraction of population to keep during evolution
        """
        if self.verbose:
            print("\n" + "="*80)
            print("STARTING POPULATION TRAINING")
            print("="*80)
            print(f"Population Size: {self.population_size}")
            print(f"Generations: {generations}")
            print(f"Episodes per Generation: {episodes_per_generation}")
            print(f"Evolution Frequency: Every {evolution_frequency} generations")
            if base_model_path:
                print(f"Base Model: {base_model_path}")
            print("="*80)

        # Initialize population if empty
        if not self.population:
            self.initialize_population(base_model_path=base_model_path)

        # Training loop
        for gen in range(generations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"GENERATION {self.generation + 1}/{generations}")
                print(f"{'='*80}")

            # Create matchmaking pairs
            pairs = self.create_matchmaking_pairs()

            # Train each fighter
            for fighter in self.population:
                # Select opponents for this fighter
                opponents = [
                    opp for pair in pairs
                    for opp in [pair[0], pair[1]]
                    if pair[0] == fighter or pair[1] == fighter
                    if opp != fighter
                ]

                # If no opponents from pairs, use random selection
                if not opponents:
                    opponents = [f for f in self.population if f != fighter]
                    opponents = random.sample(opponents, min(3, len(opponents)))

                if self.verbose:
                    print(f"\nTraining {fighter.name} against {len(opponents)} opponents...")

                # Train
                stats = self.train_fighter_batch(
                    fighter,
                    opponents,
                    episodes=episodes_per_generation // len(self.population)
                )

                if self.verbose:
                    print(f"  Completed {stats['episodes']} episodes, "
                         f"mean reward: {stats['mean_reward']:.1f}")

            # Evaluation matches
            self.run_evaluation_matches(num_matches_per_pair=3)

            # Show leaderboard
            if self.verbose:
                self.elo_tracker.print_leaderboard()

            # Evolution
            if (gen + 1) % evolution_frequency == 0 and gen < generations - 1:
                self.evolve_population(keep_top=keep_top)

            # Save checkpoint
            self.save_population()

            self.generation += 1

        # Final report
        if self.verbose:
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80)
            print(f"Total Generations: {self.generation}")
            print(f"Total Matches: {self.total_matches}")
            print("\nFinal Rankings:")
            self.elo_tracker.print_leaderboard()

            # Show diversity metrics
            metrics = self.elo_tracker.get_diversity_metrics()
            print(f"\nPopulation Diversity:")
            print(f"  ELO Spread: {metrics['elo_range']:.0f}")
            print(f"  ELO Std Dev: {metrics['elo_std']:.1f}")
            if 'win_rate_std' in metrics:
                print(f"  Win Rate Variance: {metrics['win_rate_std']:.3f}")

        self.logger.info("Training complete")
        self.logger.info(f"Final generation: {self.generation}")