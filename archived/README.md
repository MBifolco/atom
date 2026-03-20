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

## Active Entry Points

For normal use, prefer:

- `atom_fight.py`
- `train_progressive.py`
- `colab_bootstrap.sh`
- `scripts/training/run_local_baseline.py`
- `scripts/training/resume_population_training.py`
- `scripts/training/build_registry.py`
- `scripts/montage/create_montage.py`
- `scripts/montage/render_replays.py`
- `scripts/ops/setup_gpu.sh`

## Related Documentation

- [`docs/README.md`](../docs/README.md)
- [`docs/TRAINING_REFACTOR_ROADMAP.md`](../docs/TRAINING_REFACTOR_ROADMAP.md)
- [`docs/LOCAL_TESTING_WORKFLOW.md`](../docs/LOCAL_TESTING_WORKFLOW.md)
- [`docs/analysis/REPO_HYGIENE_AUDIT_2026-03-20.md`](../docs/analysis/REPO_HYGIENE_AUDIT_2026-03-20.md)
