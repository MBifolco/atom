# Repo Reorganization Plan

> Status: in progress (core migration complete; compatibility and cleanup still ongoing)
>
> Completed so far:
>
> - app entrypoints moved under `apps/` with root compatibility wrappers
> - Colab bootstrap implementation moved under `scripts/colab/` with a root shim
> - runtime modules established under `src/atom/runtime/` with legacy `src/...` wrappers
> - initial training and registry namespaces established under `src/atom/training/` and `src/atom/registry/`
> - curriculum trainer, curriculum components, and runtime-platform utilities now live behind `src/atom/training/...` with legacy compatibility wrappers
> - shared training modules and the population package now live behind `src/atom/training/...` with alias-based legacy module compatibility
> - coaching utilities now live behind `src/atom/coaching/...` with legacy aliases
> - guardrail tests now enforce alias behavior and preferred `src.atom.*` imports across current repo surfaces
> - utility modules now live under `src.atom.training.utils.*`; low-value legacy wrappers are being retired once current surfaces stop referencing them
> - broader no-cov smoke coverage now includes training-tier baseline tests, CLI help surfaces, web app import, and both new/legacy Colab preflight entrypoints
>
> Purpose: make the architecture visible in the repository layout without breaking the working training, Colab, and CLI flows we rely on today.

## Why Reorganize

The original vision treats Atom as a set of cooperating but bounded components:

- a sanctioned combat runtime
- independent fighter logic
- analysis and replay surfaces
- training and coaching systems outside the core match contract
- platform services such as registry, governance, and league operations

The current codebase already reflects much of that separation in code, but the repository layout still makes the project feel more like one large training-focused codebase than a platform made of clear subsystems.

That gap matters more now because:

- the training pipeline is working well enough to justify investing in the broader platform
- we want to add more complex combat and platform infrastructure next
- new contributors should be able to tell what is "core runtime" versus "training machinery" at a glance

## What We Have Today

The repo already contains the right ingredients:

- `src/arena`, `src/protocol`, `src/orchestrator`, `src/evaluator`, `src/renderer`, and `src/telemetry` are already mostly clean runtime components
- `src/training/...` contains a substantial and growing training subsystem
- `src/registry` and `src/coaching` represent platform-facing concerns
- `web/` is a real user-facing app
- root scripts such as `atom_fight.py`, `train_progressive.py`, and `colab_bootstrap.sh` are practical entrypoints, but they blur architectural boundaries

So this is not a "tear it down and start over" situation. It is a packaging and discoverability problem more than a fundamental design problem.

## Reorganization Goals

1. Keep the repository as a single monorepo.
2. Make the sanctioned runtime feel like a first-class subsystem.
3. Keep training important, but visibly separate from runtime.
4. Move app entrypoints out of the root over time.
5. Preserve compatibility for notebook, CLI, and existing scripts during migration.
6. Avoid large risky moves until compatibility shims are in place.

## Design Principles

### 1. Organize Around Bounded Contexts

The top-level conceptual split should be:

- runtime core
- training ecosystem
- platform services
- user-facing apps

### 2. Prefer Migration Over Rewrite

We should not stop active work to perform a massive rename. Every move should leave behind a stable compatibility layer until callers are updated.

### 3. Keep Operational Workflows Stable

Colab, training resume, replay generation, and CLI workflows are more important than a perfectly clean directory tree. Operational stability wins when there is a conflict.

### 4. Separate "Code Location" From "Public Interface"

Root scripts and old imports can remain as shims for a while. The important thing is that the real implementation gradually moves to clearer homes.

## Recommended Target Structure

```text
apps/
  cli/
    atom_fight.py
  training/
    train_progressive.py
  web/
    app.py

src/
  atom/
    runtime/
      arena/
      protocol/
      orchestrator/
      evaluator/
      renderer/
      telemetry/
    registry/
    training/
      pipelines/
      trainers/
      utils/
    coaching/

fighters/
scripts/
docs/
tests/
notebooks/
archived/
```

## Why This Shape

### `apps/`

`apps/` makes it obvious which code is an entrypoint or user-facing application rather than a reusable library module.

This is where we would want:

- the fight CLI
- the training CLI
- the web app

### `src/atom/runtime/`

This makes the core match system feel explicit and protected:

- arena
- protocol
- orchestrator
- evaluator
- renderer
- telemetry

These modules are the closest thing the project has to a sanctioned kernel.

### `src/atom/training/`

Training remains a major subsystem, but it is visibly downstream of the runtime rather than visually competing with it at the top level.

### `src/atom/registry/` and `src/atom/coaching/`

These sit outside the runtime and outside the training lab. That reflects the original vision more closely: they are platform capabilities, not match-time mechanics.

## Import Namespace Note

This plan deliberately separates repo layout from Python packaging. Because the project already imports through `src.*`, the first migration can use `src.atom.*` as a transitional namespace without bundling package-install work into the same change.

If we later adopt a formal installable package, public imports can be simplified to `atom.*`.

## Current-To-Target Mapping

| Current path | Target path | Notes |
| --- | --- | --- |
| `atom_fight.py` | `apps/cli/atom_fight.py` | Keep root `atom_fight.py` as a thin wrapper during migration |
| `train_progressive.py` | `apps/training/train_progressive.py` | Keep root wrapper for notebook and scripts |
| `colab_bootstrap.sh` | `scripts/colab/bootstrap.sh` | Keep root shim until notebook references are updated |
| `web/` | `apps/web/` | Move only after imports and deployment paths are settled |
| `src/arena/` | `src/atom/runtime/arena/` | Runtime core |
| `src/protocol/` | `src/atom/runtime/protocol/` | Runtime core |
| `src/orchestrator/` | `src/atom/runtime/orchestrator/` | Runtime core |
| `src/evaluator/` | `src/atom/runtime/evaluator/` | Runtime core |
| `src/renderer/` | `src/atom/runtime/renderer/` | Runtime core |
| `src/telemetry/` | `src/atom/runtime/telemetry/` | Runtime core |
| `src/training/` | `src/atom/training/` | Preserve internal substructure |
| `src/registry/` | `src/atom/registry/` | Platform service |
| `src/coaching/` | `src/atom/coaching/` | Platform service |

## Migration Phases

### Phase 1: Declare The Architecture In The Repo

Goal: make the intended boundaries explicit before moving code.

Tasks:

- add this plan
- update docs to describe the bounded contexts clearly
- identify which current modules belong to runtime, training, platform, and apps
- stop adding new root scripts unless they are true operational entrypoints

Risk: very low

Recommended now: yes

### Phase 2: Move Entrypoints Into `apps/`

Goal: reduce root-directory clutter without breaking users.

Tasks:

- create `apps/cli/atom_fight.py`
- create `apps/training/train_progressive.py`
- move the web app implementation under `apps/web/`
- keep root wrappers that import and delegate to the new app modules

Compatibility strategy:

- `python atom_fight.py ...` still works
- `python train_progressive.py ...` still works
- notebook and shell scripts continue calling the root wrappers until updated

Risk: low

Recommended next: yes

### Phase 3: Group The Runtime Core

Goal: make the combat kernel obvious and self-contained.

Tasks:

- move runtime modules under `src/atom/runtime/`
- leave compatibility re-export modules in old locations during transition
- update internal imports gradually rather than all at once
- ensure tests mirror the new runtime structure

Compatibility strategy:

- old imports continue working temporarily through forwarding modules
- new code should prefer the new runtime paths

Risk: medium

Recommended next: after app entrypoints are stable

### Phase 4: Move Training Under The New Package Layout

Goal: complete the architectural separation between runtime and training.

Tasks:

- move `src/training/...` to `src/atom/training/...`
- keep `pipelines`, `trainers`, and `utils` structure
- update direct imports from root scripts, tests, and notebooks
- preserve the current `ProgressiveTrainer` compatibility story

Risk: medium

Recommended next: after runtime moves are stable

### Phase 5: Move Platform Services

Goal: isolate registry and coaching as platform concerns rather than miscellaneous source folders.

Tasks:

- move `src/registry` to `src/atom/registry`
- move `src/coaching` to `src/atom/coaching`
- update docs and tests

Risk: low to medium

Recommended next: after runtime and training packages settle

### Phase 6: Enforce Boundaries

Goal: prevent drift back into a flat, ambiguous structure.

Tasks:

- update tests to mirror the new layout
- add documentation for allowed dependency directions
- prefer new imports in all touched files
- consider lightweight import-boundary checks if the codebase keeps growing

Risk: low

Recommended next: yes, once the structure exists

## Dependency Direction We Want

The desired dependency direction should look like this:

```text
apps -> platform/training/runtime
platform -> runtime
training -> runtime
runtime -> (no training/platform/app dependencies)
```

In practical terms:

- runtime code should not import training modules
- training can depend on runtime
- apps can depend on everything they need
- registry and coaching should not reach deep into app code

## What Should Stay Top-Level

These directories already make sense at the repository root and should remain there:

- `fighters/`
- `scripts/`
- `docs/`
- `tests/`
- `notebooks/`
- `archived/`

They are repository assets, not library packages.

## What We Should Not Do Yet

These are good future ideas, but they should not be bundled into the repo reorg:

- split the project into multiple repositories
- redesign the fighter artifact format
- redesign the protocol
- solve governance/sandboxing
- overhaul Colab/bootstrap flow again
- move generated output directories unless they are actively causing pain

Those are product or infrastructure changes, not repository-layout work.

## Recommended First Slice

If we execute this plan, the safest first implementation slice is:

1. create `apps/cli/atom_fight.py`
2. create `apps/training/train_progressive.py`
3. keep root wrappers for both
4. add a small `scripts/colab/` home for Colab-specific shell logic
5. leave `src/...` package moves for a later slice

That gives us visible improvement quickly without touching the most import-sensitive parts of the repo.

## Success Criteria

We should consider this reorganization successful when:

- a new contributor can identify runtime, training, platform, and app code quickly
- root-level clutter is reduced to a small number of intentional compatibility entrypoints
- the runtime core can be discussed as a coherent subsystem
- Colab and local training still work throughout the migration
- docs match the actual repository layout

## Related Docs

- [PLATFORM_ARCHITECTURE.md](PLATFORM_ARCHITECTURE.md)
- [TRAINING_REFACTOR_ROADMAP.md](TRAINING_REFACTOR_ROADMAP.md)
- [REFACTORING_OPPORTUNITIES.md](REFACTORING_OPPORTUNITIES.md)
- [original_vision/README.md](original_vision/README.md)
