# Archived Assets

This directory contains historical scripts, tests, and artifacts that are no
longer part of the active training/runtime entrypoints.

## Current Layout

- `benchmarks/`: performance benchmark scripts used during optimization work
- `tests/`: one-off validation tests from earlier implementation phases
- `setup_scripts/`: historical environment/setup helpers
- `legacy_training/`: superseded training stack retained for reference
- `diagnostics/`: archived one-off debug/fix scripts moved from repo root
- `montage_legacy/`: superseded montage script versions
- `media/`: generated media artifacts moved out of root

## Active Entry Points (Repo Root)

For normal use, prefer:

- `atom_fight.py`
- `train_progressive.py`
- `run_local_baseline.py`
- `resume_population_training.py`
- `colab_bootstrap.sh`
- `build_registry.py`

## Related Documentation

- [`docs/README.md`](../docs/README.md)
- [`docs/TRAINING_REFACTOR_ROADMAP.md`](../docs/TRAINING_REFACTOR_ROADMAP.md)
- [`docs/LOCAL_TESTING_WORKFLOW.md`](../docs/LOCAL_TESTING_WORKFLOW.md)
- [`docs/REPO_HYGIENE_AUDIT_2026-03-20.md`](../docs/REPO_HYGIENE_AUDIT_2026-03-20.md)
