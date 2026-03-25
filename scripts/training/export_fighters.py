#!/usr/bin/env python3
"""
Export trained fighters from a population run as standalone Python files.

Creates fighters that load their SB3 model on first call and use the
canonical observation builder for snapshot-to-obs conversion. Compatible
with atom_fight.py and the web app.

Usage:
    python scripts/training/export_fighters.py training_outputs/run5/population/models/generation_19
    python scripts/training/export_fighters.py training_outputs/run5/population/models/generation_19 --top 3
    python scripts/training/export_fighters.py training_outputs/run5/population/models/generation_19 --output fighters/AIs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


FIGHTER_TEMPLATE = '''"""
{name} — Trained AI fighter exported from population training.

Record: {record}
ELO: {elo}
Generation: {generation}
Lineage: {lineage}

Trained using PPO with 4D logit action space.
"""

import numpy as np
from pathlib import Path

_model = None
_model_path = str(Path(__file__).parent / "{model_filename}")


def _load_model():
    global _model
    if _model is not None:
        return _model
    from stable_baselines3 import PPO
    _model = PPO.load(_model_path, device="cpu")
    return _model


def decide(snapshot):
    """
    Make a decision based on the current game state.

    Args:
        snapshot: dict with you, opponent, and arena state

    Returns:
        dict with acceleration (float) and stance (str)
    """
    from src.atom.training.signal_engine import build_observation_from_snapshot

    model = _load_model()
    obs = build_observation_from_snapshot(snapshot)
    action, _ = model.predict(obs, deterministic=False)

    acceleration = float(np.clip(action[0], -1.0, 1.0)) * 4.375
    stance_idx = int(np.argmax(action[1:4]))
    stances = ["neutral", "extended", "defending"]

    return {{
        "acceleration": acceleration,
        "stance": stances[stance_idx],
    }}
'''


def parse_rankings(rankings_path: Path) -> dict:
    """Parse rankings.txt into a dict of name -> {elo, record, rank}."""
    rankings = {}
    with open(rankings_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Generation") or line.startswith("="):
                continue
            # Format: "1. Fierce_Cobra_G13: ELO=1575, Record=30-12-0"
            parts = line.split(". ", 1)
            if len(parts) != 2:
                continue
            rank = int(parts[0])
            rest = parts[1]
            name = rest.split(":")[0].strip()
            elo = 0
            record = "0-0-0"
            if "ELO=" in rest:
                elo = int(rest.split("ELO=")[1].split(",")[0])
            if "Record=" in rest:
                record = rest.split("Record=")[1].strip()
            rankings[name] = {"rank": rank, "elo": elo, "record": record}
    return rankings


def export_fighter(model_path: Path, output_dir: Path, rankings: dict) -> Path:
    """Export a single fighter."""
    name = model_path.stem
    fighter_dir = output_dir / name
    fighter_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    dest_model = fighter_dir / model_path.name
    shutil.copy2(model_path, dest_model)

    # Parse generation from name (e.g., "Fierce_Cobra_G13" -> 13)
    generation = "0"
    if "_G" in name:
        generation = name.split("_G")[-1]

    info = rankings.get(name, {"elo": 0, "record": "0-0-0", "rank": 0})

    # Write fighter Python file
    fighter_py = fighter_dir / f"{name}.py"
    fighter_py.write_text(
        FIGHTER_TEMPLATE.format(
            name=name,
            record=info["record"],
            elo=info["elo"],
            generation=generation,
            lineage=name,
            model_filename=model_path.name,
        )
    )

    # Write README
    readme = fighter_dir / "README.md"
    readme.write_text(
        f"# {name}\n\n"
        f"Trained AI fighter from population training.\n\n"
        f"- **ELO:** {info['elo']}\n"
        f"- **Record:** {info['record']}\n"
        f"- **Generation:** {generation}\n"
        f"- **Rank:** {info['rank']}\n\n"
        f"## Usage\n\n"
        f"```bash\n"
        f"python atom_fight.py fighters/AIs/{name}/{name}.py fighters/examples/boxer.py --html replay.html\n"
        f"```\n"
    )

    return fighter_dir


def main():
    parser = argparse.ArgumentParser(description="Export trained fighters")
    parser.add_argument("generation_dir", help="Path to generation directory with .zip models")
    parser.add_argument("--top", type=int, default=0, help="Export only top N fighters (0=all)")
    parser.add_argument("--output", default="fighters/AIs", help="Output directory")
    args = parser.parse_args()

    gen_dir = Path(args.generation_dir)
    if not gen_dir.exists():
        print(f"ERROR: {gen_dir} not found", file=sys.stderr)
        sys.exit(1)

    rankings_path = gen_dir / "rankings.txt"
    rankings = parse_rankings(rankings_path) if rankings_path.exists() else {}

    models = sorted(gen_dir.glob("*.zip"))
    if not models:
        print(f"No .zip models found in {gen_dir}")
        sys.exit(1)

    # Sort by ranking
    if rankings:
        models.sort(key=lambda p: rankings.get(p.stem, {}).get("rank", 999))

    if args.top > 0:
        models = models[:args.top]

    output_dir = Path(args.output)
    print(f"Exporting {len(models)} fighters to {output_dir}/\n")

    for model_path in models:
        info = rankings.get(model_path.stem, {})
        rank = info.get("rank", "?")
        elo = info.get("elo", 0)
        fighter_dir = export_fighter(model_path, output_dir, rankings)
        print(f"  #{rank} {model_path.stem} (ELO {elo}) → {fighter_dir}")

    print(f"\nDone! Test with:")
    print(f"  python atom_fight.py {output_dir}/{models[0].stem}/{models[0].stem}.py fighters/examples/boxer.py --html replay.html")


if __name__ == "__main__":
    main()
