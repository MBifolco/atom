# Example Fighters Guide

This directory contains the hand-authored benchmark fighters used by curriculum
training (Level 5), quick smoke checks, and manual match debugging.

## Current Fighters

- `boxer.py`: pressure-first baseline with direct engagement.
- `counter_puncher.py`: reactive style that punishes overcommitment.
- `out_fighter.py`: range-control style focused on spacing.
- `slugger.py`: high-impact aggression with strong finishing pressure.
- `swarmer.py`: close-range tempo fighter that stays on the opponent.

## How These Fighters Are Used

- **Curriculum Level 5** (`Expert`) rotates through these 5 opponents.
- **Registry build** scans this directory and adds fighters to `fighters/registry.json`.
- **Manual evaluation** uses these as stable references for champion comparisons.

## Running Matches

```bash
# Example vs example
python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --html replay.html

# Trained AI vs benchmark
python atom_fight.py fighters/AIs/fighter_0/fighter_0.py fighters/examples/counter_puncher.py --watch

# Quick comparison run
python atom_fight.py fighters/examples/swarmer.py fighters/examples/out_fighter.py --episodes 20
```

## Adding a New Example Fighter

1. Add a new Python file in this directory with a `decide(snapshot)` function.
2. Keep the output shape compatible with `atom_fight.py`:
   - `{"acceleration": float, "stance": str}`
3. Use one of these stances: `neutral`, `extended`, `retracted`, `defending`.
4. Rebuild the registry:

```bash
python scripts/training/build_registry.py
```

## Notes

- File names should be snake_case and descriptive.
- Keep fighters deterministic unless randomness is intentional for a benchmark.
- If a fighter is removed, update docs and any helper scripts that reference it.
