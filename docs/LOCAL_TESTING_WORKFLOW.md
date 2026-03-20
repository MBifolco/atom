# Local Testing Workflow

This project now supports a local-first workflow so we can iterate on training
logic without requiring Colab for every code change.

## Why

- Faster edit/test loop for trainer refactors.
- Deterministic seeded runs for reproducibility.
- Colab usage reduced to milestone validation gates.

## Daily Workflow

1. Run fast unit tests:

```bash
make test-unit
```

2. Run integration tests:

```bash
make test-integration
```

3. Run local deterministic baseline training when touching trainer logic:

```bash
make baseline-local
```

Or directly:

```bash
python run_local_baseline.py \
  --mode curriculum \
  --timesteps 10000 \
  --seed 1337 \
  --cores 1 \
  --device cpu
```

## Opt-In Heavier Gates

Training smoke tests:

```bash
make test-training
```

End-to-end checks:

```bash
make test-e2e
```

## Colab Usage

Use Colab only for milestone gates, not every commit:

1. End of semantic parity work.
2. End of checkpoint/recovery hardening.
3. Final operational validation before long runs.

Checklist:
- `docs/COLAB_VALIDATION_CHECKLIST.md`

## Seed Controls

- Test seed: `ATOM_TEST_SEED` (pytest fixtures)
- Training seed markers for subprocess runs:
  - `PYTHONHASHSEED`
  - `ATOM_GLOBAL_SEED`
  - `ATOM_TRAINING_SEED`

These are automatically set by `run_local_baseline.py`.
