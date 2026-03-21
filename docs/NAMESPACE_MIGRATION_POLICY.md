# Namespace Migration Policy

This doc defines how we move from legacy `src.*` imports and wrapper modules toward the preferred `src.atom.*` namespaces without breaking the workflows we still rely on.

## Goals

- keep public entrypoints stable while the architecture becomes clearer
- retire low-value wrappers once current repo surfaces no longer depend on them
- avoid surprise breakage in Colab, CLI, and notebook workflows
- make future cleanup incremental and testable

## Priority Order

1. Keep root operational entrypoints stable longest.
   Examples: `atom_fight.py`, `train_progressive.py`, `colab_bootstrap.sh`.
2. Keep notebook and CLI module entrypoints stable until their callers are updated.
   Example: legacy `python -m src.training.utils.colab_preflight ...`.
3. Retire leaf helper wrappers first once current surfaces are clean.
   Examples: helper submodules under `src/training/utils/` or `src/training/trainers/population/`.
4. Retire core runtime wrappers last.
   Examples: `src/arena/*`, `src/protocol/*`, `src/orchestrator/*`.

## Retirement Checklist

Before removing a legacy wrapper module, do all of the following:

1. Verify current repo surfaces are already using the preferred namespace.
   Current surfaces include `apps/`, `scripts/`, `web/`, `README.md`, current docs, and notebooks.
2. Update tests that still import the legacy submodule path.
3. Preserve package-level compatibility if it still provides value.
   Example: `from src.training.utils import BaselineRunConfig` can keep working even after `src/training/utils/baseline_harness.py` is removed.
4. Add or update guardrail tests so the retirement is intentional and sticky.
5. Run no-cov validation on the affected slice plus one public-surface smoke command when relevant.

## Guardrail Expectations

Use these guardrails during migration:

- `tests/unit/test_preferred_import_surfaces.py` protects current repo surfaces from drifting back to legacy imports.
- `tests/unit/test_namespace_aliases.py` protects the wrappers we intentionally keep and asserts selected retired wrappers stay gone.
- Focused `pytest --no-cov` runs are preferred for migration slices so unrelated coverage gates do not blur the signal.

## What We Keep For Now

These legacy compatibility points still earn their keep:

- root entrypoints: `atom_fight.py`, `train_progressive.py`, `colab_bootstrap.sh`
- old package-level imports that remain convenient and low-risk
- old module entrypoints that active docs or notebook flows still call

## What We Prefer For New Code

For all new code and current docs:

- prefer `src.atom.runtime.*` for combat/runtime modules
- prefer `src.atom.training.*` for training modules
- prefer `src.atom.registry.*` and `src.atom.coaching.*` for platform services
- prefer `apps/...` paths when referencing user-facing app implementations

## Decision Rule

If a wrapper is only serving old tests or stale docs, update those references and retire it.
If a wrapper still supports an active user workflow, keep it until that workflow is migrated and covered by tests.
