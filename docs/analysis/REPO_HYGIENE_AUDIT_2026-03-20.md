# Repo Hygiene Audit (2026-03-20)

## Scope

This audit reviewed top-level scripts/docs in `atom/` and identified candidates to:

- delete
- archive
- move
- update

The goal is to keep root entry points clean while preserving useful historical/debug artifacts.

## What Was Verified

- Top-level tracked files from `git ls-files`
- Internal references using `rg` (excluding `.git`, `outputs/`, `training_outputs/`)
- Broken markdown links in `README.md` and `docs/`
- Large tracked files

## Keep in Root (Current Entry Points)

These appear to be core or actively used operational scripts:

- `atom_fight.py`
- `train_progressive.py`
- `colab_bootstrap.sh`

## Moved Out of Root (Follow-up Cleanup)

Operational tooling was moved under `scripts/` to reduce root clutter:

- `scripts/training/build_registry.py`
- `scripts/training/run_local_baseline.py`
- `scripts/training/resume_population_training.py`
- `scripts/montage/render_replays.py`
- `scripts/montage/create_html_montage.py`
- `scripts/montage/create_html_montage_from_existing.py`
- `scripts/montage/create_montage.py`
- `scripts/ops/setup_gpu.sh`
- `scripts/ops/tail_latest_log.sh`

## High-Confidence Archive Candidates

These are one-off/debug-era scripts with no meaningful runtime references in the current codebase:

- `debug_nan_issue.py`
- `diagnose_level5_nan.py`
- `diagnose_reward_patterns.py`
- `diagnose_training_nan.py`
- `explain_vecnormalize.py`
- `find_dimension_mismatch.py`
- `find_real_nan_cause.py`
- `fix_gpu_setup.sh`
- `fix_nan_properly.py`
- `fix_nan_with_reward_clipping.py`
- `fix_observations_properly.py`
- `fix_population_nan.py`
- `fix_population_obs.py`
- `quick_reward_pattern_test.py`
- `restore_13_dim_properly.py`
- `revert_to_stable.py`
- `test_13dim_observations.py`
- `test_gpu_memory.py`
- `test_nan_fixes.py`
- `test_nan_protection.py`
- `test_population_fixes.py`
- `test_population_models.py`
- `test_proper_nan_fix.py`
- `test_stance_fix.py`
- `test_vmap_obs.py`
- `vmap_obs_patch.py`

Destination (completed):

- `archived/diagnostics/` with `nan/`, `observations/`, `population/`, and `general/`

Note: `test_vmap_hang.py` was subsequently moved to `archived/diagnostics/general/`
as part of the follow-up cleanup slice.

## Legacy/Versioned Script Candidates

- `create_html_montage_old.py`
- `create_html_montage_v2.py`

Destination (completed):

- `archived/montage_legacy/`

## Tracked Artifact Candidate

- `training_montage_20251121_163936.mp4` (large tracked media file)

Current status:

- Moved to `archived/media/`.
- If repo size reduction is required, history rewrite is still needed (moving files in
  current tip does not shrink historical Git object data).

## Docs and Link Hygiene (Completed in this pass)

- Updated stale links and missing references in:
  - `README.md`
  - `docs/README.md`
  - `docs/PLATFORM_ARCHITECTURE.md`
  - `docs/REPLAY_MONTAGE.md`
  - `fighters/README.md`
  - `archived/legacy_training/training/README.md`
- Added:
  - `fighters/test_dummies/README.md`
- Added local scratch ignore:
  - `.gitignore` now ignores `tmp.md`

## Recommended Next Cleanup Slice

1. Review whether additional operational root scripts should move into `scripts/`.
2. Consider adding a lightweight CI check for broken markdown links.
3. Consider a history rewrite if repository size optimization is a goal.
