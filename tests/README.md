# Atom Combat Test Suite

The test suite now supports a local-first workflow so we can move quickly
without depending on Colab for every change.

## Test Tiers

- `unit`: fastest deterministic logic tests.
- `integration`: local component tests across modules.
- `training`: opt-in training smoke/regression tests.
- `e2e`: opt-in full pipeline checks.
- `slow`: longer-running tests across any tier.
- `colab`: tests that require a Colab environment.

Legacy flat tests are auto-marked as `integration` in `tests/conftest.py`.

## Default Behavior

`pytest.ini` skips `training`, `e2e`, and `colab` tests by default:

```bash
python -m pytest
```

This keeps local loops fast while still running the broad integration surface.

## Common Commands

Run only unit tests:

```bash
python -m pytest -m unit
```

Run integration tests (excluding slow):

```bash
python -m pytest -m "integration and not slow"
```

Run all local tests including slow (still excluding training/e2e/colab):

```bash
python -m pytest -m "not training and not e2e and not colab"
```

Run training smoke tests explicitly:

```bash
python -m pytest tests/training -m training -s
```

Run end-to-end checks explicitly:

```bash
python -m pytest tests/e2e -m e2e -s
```

## Reproducibility

- `ATOM_TEST_SEED` controls deterministic Python/NumPy seeding in tests.
- `tests/conftest.py` applies seed fixtures and optional torch seeding fixture.
- Local baseline training runs can be launched with `run_local_baseline.py`.

Example:

```bash
ATOM_TEST_SEED=1337 python -m pytest -m unit
python run_local_baseline.py --mode curriculum --timesteps 10000 --seed 1337
```
