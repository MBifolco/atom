# Legacy Training Code - Archived

This directory contains old training code that was superseded by `src/training/`.

## What Was Archived

- `training/src/` - Old duplicate of `src/training/`
- `training/train_fighter.py` - Superseded by `train_progressive.py` in root
- `training/train_population.py` - Superseded by population trainer in `src/training/trainers/`
- `training/train_progressive.py` - Old version

## Current Active Code

All training is now in:
- **`src/training/`** - Training infrastructure (gym_env, trainers, etc.)
- **`train_progressive.py`** (root) - Main training script
- **`resume_population_training.py`** (root) - Resume training

## Training Outputs

Training outputs were preserved in:
- **`training_outputs/`** in project root

## Date Archived

2025-11-18
