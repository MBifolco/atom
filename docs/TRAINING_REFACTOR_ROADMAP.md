# Training Refactor Roadmap

## Purpose

This roadmap defines how we stabilize and refactor the training system without starting from scratch, with a focus on:

1. Eliminating recurrent NaN failures in long curriculum runs.
2. Making training behavior consistent across CPU env and vmap env paths.
3. Moving to a local-first test strategy so Colab is only needed at milestone gates.
4. Modernizing and expanding the test suite so regressions are caught early.

## Decision

Salvage and refactor the current codebase, not a full rewrite.

Why:
- Core simulation behavior is largely functional.
- Failures are concentrated in trainer architecture, reward-path divergence, normalization/recovery handling, and orchestration complexity.
- A staged refactor is lower risk and faster to production reliability than a greenfield reset.

## High-Level Outcomes

1. Training runs are reproducible and recoverable.
2. `gym_env` and `vmap_env_wrapper` share one canonical reward/observation path.
3. Trainer modules are decomposed into testable components.
4. Colab usage is reduced to explicit validation checkpoints.
5. The test suite is tiered, faster, and regression-focused.

## Scope Boundaries

In scope:
- Curriculum and population trainer refactor.
- Reward/observation unification.
- Checkpoint/resume/recovery hardening.
- Test suite revamp and CI-friendly local execution strategy.
- Colab notebook/bootstrap alignment with refactored training APIs.

Out of scope for this phase:
- New RL algorithms as first-class paths (beyond maintaining current behavior).
- Distributed multi-node training.
- Architecture-search / meta-learning features.

## Phase Plan

### Phase 0: Baseline and Guardrails

Goals:
- Establish deterministic baseline behavior and failure signatures.
- Define measurable gates before deep refactor.

Deliverables:
- Repro harness for curriculum and level-specific runs with fixed seeds.
- NaN diagnostics artifact bundle (policy stats, gradients, rewards, env snapshots).
- Baseline dashboard report for runtime, failure rate, and graduation progression.

Exit criteria:
- We can reproduce the current failure profile in a controlled run.
- We have pass/fail metrics for each subsequent phase.

### Phase 1: Canonical Reward/Observation Engine

Goals:
- Remove divergence between `gym_env` and `vmap_env_wrapper`.
- Ensure identical semantics for reward and observation construction.

Deliverables:
- Shared `reward_engine` module used by both paths.
- Shared `observation_builder` module used by both paths.
- Parity tests comparing single-env and batched-env outputs on matched transitions.

Exit criteria:
- Reward and observation parity tests pass.
- No duplicated reward formulas remain in env wrappers.

Phase 1 status (updated 2026-03-19):
- Completed: shared canonical signal engine now drives both `gym_env` and `vmap_env_wrapper`.
- Completed: parity regression coverage for scalar/batch observation building and reward behavior.
- Completed: protocol snapshot -> observation adapter (`build_observation_from_snapshot`) adopted across:
  - curriculum replay/eval decision path
  - replay recorder model evaluation path
  - population trainer model decision wrappers
  - legacy population training scripts
- Completed: stale local reward helper removed from `gym_env` to avoid divergent reward logic.
- Remaining before formal gate sign-off:
  - run a Colab milestone validation pass against latest branch state.

### Phase 2: Curriculum Trainer Decomposition

Goals:
- Break monolithic curriculum flow into isolated, testable components.

Target components:
- `LevelRunner`
- `GraduationPolicy`
- `RecoveryManager`
- `EnvFactory`
- `ModelFactory`
- `ProgressReporter`

Deliverables:
- State-machine-driven curriculum execution (explicit transitions).
- Structured error classes (NaN, OOM, env reset, checkpoint restore, etc.).
- Cleaner level transitions with explicit normalization/restore rules.

Exit criteria:
- Curriculum loop logic is no longer concentrated in one large method.
- Unit/integration tests exist for each component.

Phase 2 status (updated 2026-03-19):
- Completed (slice 1): extracted reusable components into `curriculum_components.py`:
  - `GraduationPolicy`
  - `ProgressReporter`
  - `RecoveryManager`
  - `LevelRunner`
  - `EnvFactory`
  - `ModelFactory`
- Completed (slice 1): `CurriculumTrainer` now delegates progression, reporting, env/model creation, and retry/recovery orchestration to these components.
- Completed (slice 1): added dedicated unit tests for component behavior (`tests/unit/test_curriculum_components.py`) and validated existing curriculum/replay suites.
- Completed (slice 2): added explicit level-transition state machine (`LevelTransitionStateMachine`) and integrated it into `CurriculumTrainer.advance_level()`.
- Completed (slice 2): moved callback episode-step orchestration into componentized `CallbackStepProcessor` + `ReplayEvaluationService`, with compatibility wrapper retained for existing replay tests/tools.
- Completed (slice 2): expanded component unit coverage for transition state machine and callback step processing behavior.
- Completed (slice 3): introduced structured curriculum training-domain errors (`CurriculumTrainingError` family) with unified context payloads for NaN exhaustion, checkpoint recovery failure, and unexpected loop execution failures.
- Completed (slice 3): `CurriculumTrainer.train()` now uses unified error logging/cleanup for structured training loop failures.
- Completed: Phase 2 curriculum trainer decomposition goals.

### Phase 3: Checkpoint and Recovery Hardening

Goals:
- Ensure recovery resumes from recent state and does not destabilize training.

Deliverables:
- Periodic checkpoint callback (model + optimizer-relevant training state).
- Recovery policy with bounded retries and backoff strategy.
- Resume tests: mid-level, post-level-transition, and post-NaN restart.

Exit criteria:
- Forced-failure resume tests pass.
- Recovery no longer depends on stale initial checkpoints.

Phase 3 status (updated 2026-03-19):
- Completed (slice 1): added periodic checkpoint callback (`PeriodicCheckpointCallback`) integrated into `LevelRunner` to persist checkpoint bundles throughout training (not only loop start).
- Completed (slice 1): checkpoint bundles now persist:
  - model snapshot (`.zip`, includes optimizer-relevant state managed by SB3)
  - training resume state payload (`.state.json`)
  - VecNormalize stats (`.vecnormalize.pkl`) when available
- Completed (slice 1): recovery retries now apply bounded exponential backoff policy via `RecoveryManager.backoff_seconds`.
- Completed (slice 1): `CurriculumTrainer` now captures/restores progress + callback state for resume and re-syncs environment level when restored state crosses level boundaries.
- Completed (slice 1): added focused resume/recovery tests for:
  - mid-level checkpoint state roundtrip
  - post-level-transition checkpoint state roundtrip
  - NaN restart state restoration path in `LevelRunner`
- Completed (slice 2): recovery now resolves environments dynamically (`env_getter`) so checkpoint load/retry paths use post-restore env state instead of stale pre-transition env references.
- Completed (slice 2): forced-failure resume coverage now includes latest-checkpoint recovery boundary assertions (verifies restore/load uses most recent periodic checkpoint state).
- Completed (slice 2): trainer-level state restore tests now cover:
  - progress/callback capture+restore roundtrip
  - non-vmap level-change resync (`set_opponent` path)
  - vmap level-change resync (recreate env + `model.set_env`)
- Completed (slice 3): added explicit resume-from-latest support through `CurriculumTrainer.train(resume_from_latest=True)` and `train_progressive.py --resume-curriculum`, with configurable checkpoint cadence (`--checkpoint-interval`).
- Completed (slice 3): executed a timeboxed local real-loop smoke+resume validation:
  - initial curriculum run generated periodic checkpoint bundles (`checkpoint_0/1024/2048`)
  - resume run loaded latest bundle and continued from a non-zero step (`checkpoint_2048`)
- Completed: Phase 3 checkpoint and recovery hardening goals.

### Phase 4: Population Trainer Refactor

Goals:
- Split process orchestration from fighter training logic.
- Remove duplication and clarify CPU/GPU backend behavior.

Deliverables:
- `TrainingWorker` abstraction with pluggable backend strategy.
- Clean task orchestration layer separate from model/env lifecycle.
- Consolidated model save/load/mutation interfaces.

Exit criteria:
- Population training critical paths are covered by focused tests.
- Large monolithic methods are replaced by composable units.

### Phase 5: Colab and Operationalization

Goals:
- Keep Colab as a validation target, not a development bottleneck.

Deliverables:
- Updated notebook with stable bootstrap, smoke, full-run, and resume workflows.
- Runtime preflight checks with actionable error output.
- Colab validation checklist tied to milestone gates.

Exit criteria:
- Minimal manual intervention to run milestone validation in Colab.

## Local-First Testing Strategy (Primary Request)

Yes, we can test most work locally without requiring you to run Colab every cycle.

### Tier 0: Fast deterministic unit tests (local, every change)
- Pure reward/observation tests.
- Graduation and progression policy tests.
- Mutation/selection and helper logic tests.
- Runtime platform and config parsing tests.

Target runtime: under 2 minutes.

### Tier 1: Component integration tests (local)
- `gym_env` and `vmap_env_wrapper` parity checks.
- Curriculum level transition tests with short episodes.
- Checkpoint/save/load/resume flow tests.

Target runtime: 2-8 minutes.

### Tier 2: Training smoke tests (local)
- Short PPO curriculum runs with strict seeds and reduced timesteps.
- NaN regression smoke with assertions on finite policy parameters.

Target runtime: 5-20 minutes.

### Tier 3: Colab milestone gates (not every PR)
- Run only at milestone boundaries:
  - End of Phase 1 (semantic parity complete)
  - End of Phase 3 (recovery hardened)
  - End of Phase 5 (operational readiness)

This keeps Colab runs infrequent and high-value.

## Test Suite Revamp Plan

### Proposed structure

`tests/unit/`
- deterministic pure logic tests

`tests/integration/`
- trainer + env integration (short)

`tests/training/`
- smoke runs, NaN regression, checkpoint recovery

`tests/e2e/`
- minimal end-to-end pipeline checks

### New test assets

1. Seeded fixtures for reproducibility.
2. Synthetic opponent fixtures for controlled scenarios.
3. Golden transition snapshots for reward/observation parity.
4. Failure injection helpers (NaN, timeout, interrupted worker, corrupted checkpoint).

### Priority regression tests

1. `test_reward_parity_single_vs_vmap`
2. `test_curriculum_level_transition_preserves_expected_behavior`
3. `test_recovery_from_nan_resumes_from_recent_checkpoint`
4. `test_policy_parameters_remain_finite_after_smoke_training`
5. `test_population_parallel_training_task_recovery`

## Delivery Cadence

For each phase:
1. Refactor in small vertical slices.
2. Add tests first for the target behavior.
3. Land behind compatibility flags where needed.
4. Remove legacy paths once parity is proven.

## Risks and Mitigations

Risk: Refactor breaks existing long-run behavior.  
Mitigation: Keep baselines + phase gates + parity tests before removing legacy code.

Risk: Colab-only issues appear late.  
Mitigation: Milestone-based Colab gates and preflight checks.

Risk: Performance regressions.  
Mitigation: Track fixed smoke benchmarks each phase.

## Immediate Next Steps

1. Implement Phase 0 reproducibility harness and baseline report.
2. Start Phase 1 by extracting a shared reward engine with parity tests.
3. Add Tier 0/Tier 1 test scaffolding and markers to support local-first workflow.
