"""
Population persistence and export helpers.

This module encapsulates generation checkpoint saving and AI export artifacts so
PopulationTrainer can remain a thin orchestration layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass(frozen=True)
class PopulationPersistenceContext:
    """Runtime context required by population persistence/export helpers."""

    models_dir: Path
    project_root: Path
    algorithm: str
    population_size: int
    generation: int
    verbose: bool
    logger: logging.Logger


class PopulationPersistenceService:
    """Saves generation artifacts and exports fighters in atom_fight format."""

    def __init__(self, context: PopulationPersistenceContext):
        self.context = context

    def generation_dir(self) -> Path:
        """Return (and ensure) current generation output directory."""
        generation_dir = self.context.models_dir / f"generation_{self.context.generation}"
        generation_dir.mkdir(exist_ok=True)
        return generation_dir

    def save_generation_models(self, population: List[Any], generation_dir: Path) -> None:
        """Persist each fighter model and update checkpoint metadata."""
        for fighter in population:
            model_path = generation_dir / f"{fighter.name}.zip"
            fighter.model.save(model_path)
            fighter.last_checkpoint = str(model_path)

    def write_rankings_file(self, rankings: List[Any], generation_dir: Path) -> Path:
        """Write generation ranking summary file."""
        rankings_file = generation_dir / "rankings.txt"
        with open(rankings_file, "w") as f:
            f.write(f"Generation {self.context.generation} Rankings\n")
            f.write("=" * 60 + "\n")
            for i, stats in enumerate(rankings, 1):
                f.write(
                    f"{i}. {stats.name}: ELO={stats.elo:.0f}, "
                    f"Record={stats.wins}-{stats.losses}-{stats.draws}\n"
                )
        return rankings_file

    def resolve_ais_dir(self) -> Path:
        """Resolve and ensure fighters/AIs export directory."""
        ais_dir = self.context.project_root / "fighters" / "AIs"
        ais_dir.mkdir(parents=True, exist_ok=True)
        return ais_dir

    @staticmethod
    def compute_win_rate(stats: Any) -> float:
        """Compute fighter win rate from EloTracker stats object."""
        total_matches = stats.wins + stats.losses + stats.draws
        if total_matches == 0:
            return 0.0
        return (stats.wins + 0.5 * stats.draws) / total_matches

    def export_fighter_bundle(
        self,
        fighter: Any,
        stats: Any,
        win_rate: float,
        ais_dir: Path,
    ) -> Path:
        """Export ONNX model + wrapper + README for a fighter."""
        fighter_dir = ais_dir / fighter.name
        fighter_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = fighter_dir / f"{fighter.name}.onnx"
        self.export_model_to_onnx(fighter.model, onnx_path)

        py_path = fighter_dir / f"{fighter.name}.py"
        self.create_fighter_wrapper(fighter, py_path, onnx_path.name)

        readme_path = fighter_dir / "README.md"
        self.create_fighter_readme(fighter, stats, win_rate, readme_path)

        return fighter_dir

    def export_model_to_onnx(self, model: Any, output_path: Path) -> None:
        """Export a Stable-Baselines3 model to ONNX format."""
        import torch

        policy = model.policy

        # Backward-compatible default for tests/mocks lacking observation_space.
        obs_shape = (9,)
        space = getattr(model, "observation_space", None)
        shape = getattr(space, "shape", None)
        if isinstance(shape, tuple):
            obs_shape = shape
        elif isinstance(shape, list):
            obs_shape = tuple(shape)
        if len(obs_shape) != 1:
            raise ValueError(f"Expected 1D observation space, got shape={obs_shape}")
        dummy_input = torch.zeros((1, obs_shape[0]), dtype=torch.float32)

        torch.onnx.export(
            policy,
            dummy_input,
            str(output_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=17,
        )

    def create_fighter_wrapper(self, fighter: Any, output_path: Path, onnx_filename: str) -> None:
        """Create a Python wrapper file with decide() for atom_fight.py."""
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
from src.atom.training.signal_engine import build_observation_from_snapshot

# ONNX model path (relative to this file)
ONNX_MODEL = "{onnx_filename}"

# Global session (loaded once)
_session = None
_stance_names = ["neutral", "extended", "defending"]


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
            - stance: str ("neutral", "extended", "defending")
    """
    session = _load_session()

    obs = build_observation_from_snapshot(snapshot).reshape(1, -1)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {{input_name: obs}})

    # Parse action
    # Action space is Box: [acceleration_normalized, stance_selector]
    action = outputs[0][0]
    acceleration_normalized = np.clip(action[0], -1.0, 1.0)
    stance_idx = int(np.clip(action[1], 0, 2))

    # Scale acceleration (max_acceleration = 4.5)
    acceleration = float(acceleration_normalized * 4.5)
    stance = _stance_names[stance_idx]

    return {{"acceleration": acceleration, "stance": stance}}
'''
        with open(output_path, "w") as f:
            f.write(template)

    def create_fighter_readme(
        self,
        fighter: Any,
        stats: Any,
        win_rate: float,
        output_path: Path,
    ) -> None:
        """Create README.md with fighter metadata and usage instructions."""
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
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/boxer.py

# Watch the fight in terminal
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/slugger.py --watch

# Generate HTML replay
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/counter_puncher.py --html replay.html

# Custom mass (if different from trained mass)
python atom_fight.py fighters/AIs/{fighter.name}/{fighter.name}.py fighters/examples/boxer.py --mass-a {fighter.mass:.0f}
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

**Training Algorithm**: {self.context.algorithm.upper()}
**Population Size**: {self.context.population_size}
**Generation**: {self.context.generation}

## Notes

- The fighter was trained at {fighter.mass:.1f}kg mass. Performance may vary with different masses.
- Win rate of {win_rate:.1%} was achieved against the training population.
- The ONNX model requires `onnxruntime` to run inference.
'''
        with open(output_path, "w") as f:
            f.write(readme)
