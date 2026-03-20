# Archived Diagnostics

Historical one-off investigation and patch scripts moved from repo root.

These are preserved for context and postmortem reference, but are not part of
the current runtime/training entrypoints.

## Layout

- `nan/`: NaN and reward-scaling investigations
- `observations/`: observation-shape and vmap observation patches/tests
- `population/`: population-specific diagnostic patches/tests
- `general/`: misc rollback/setup diagnostics

## Current Canonical Entry Points

Use these for normal workflows:

- `train_progressive.py`
- `run_local_baseline.py`
- `resume_population_training.py`
- `colab_bootstrap.sh`
- `tests/` (pytest suites)
