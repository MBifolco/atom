# Post-Run3 Fix Plan

Compiled from Claude + Codex reviews of the March 23, 2026 overnight run (run3).
Run3 validated Phase 1 fixes (stance action space, graduation quality gate, ELO
reset, rank-weighted parent selection). This document tracks the remaining work.

## Run3 Outcomes

**What's working:**
- Curriculum graduate uses all 3 stances (extended ~41%, defending ~26%)
- Combat quality gate passing with real damage (mean 44-60, nonzero rate 94-98%)
- Holdout: 5/10 deterministic wins (both stationary, both movement, 1/2 advanced)
- Zero founders in final population — all 8 fighters are descendants
- 14 champion turnovers across 20 generations, 9 unique champions
- Real lineage dynasties (G3 → G15 → G17)

**What's still broken:**
- ~~1:1 opponent pairing produces mirrored training dynamics~~ (Fixed)
- Holdout checkpoint snapshots are still not distinct per level
- ONNX export fails 91 times per run (cpu/cuda tensor mismatch)
- Run manifest records wrong phase
- "Final Rankings (All Time)" label is misleading post-ELO-reset

---

## Priority 1: Population Training Pair Diversity — Fixed

~~Each fighter trained against exactly one opponent per generation, producing
mirrored training dynamics.~~

**Fixed (March 2026):** Style fingerprint and diversity matchmaking system.
Each fighter now gets 3 opponents per generation, scored by a composite of
style diversity (70%) and ELO proximity (30%). Children are guaranteed at
least 1 incumbent opponent. Gen 0 uses random multi-opponent assignment.

**Done looks like:** `generation_summary.jsonl` shows `opponent_names` length
> 1 for every fighter, and child-vs-child-only clusters disappear.

---

## Priority 2: Holdout Checkpoint Snapshots

**Problem:** The deferred holdout flush correctly avoids stale mid-rollout
weights, but levels 1-4 all graduate within the same PPO rollout cycle. The
flush evaluates all 4 checkpoints against the same model. Results are identical.

This satisfies "don't evaluate stale weights" but does NOT provide distinct
per-level checkpoint comparison, which limits curriculum analysis quality.

**Approach:** Save a full checkpoint bundle at each graduation in
`advance_level`, then have the flush evaluate each saved snapshot. Bundle
must include:
- model weights (`.zip`)
- VecNormalize state (`.pkl`) if present
- checkpoint metadata: level label, global timestep, level index

This way level 1 holdout uses the level-1-graduated weights, level 2 uses
level-2, etc.

**Key files:**
- `src/atom/training/trainers/curriculum_trainer.py` — `advance_level()`,
  `_flush_pending_holdouts()`, `_record_holdout_evaluation()`

**Estimated scope:** Small-medium.

**Done looks like:** Each row in `holdout_eval.jsonl` has distinct
`global_timestep` values and different `mean_damage_dealt` per checkpoint.

---

## Priority 3: ONNX Export Fix

**Problem:** `torch.onnx.export` fails with `RuntimeError: Unhandled
FakeTensor Device Propagation for aten.mm.default, found two different
devices cpu, cuda:0`. 91 failures in run3's `export_failures.jsonl`.

**Approach:** Export from a CPU-loaded copy of the saved model, not by
mutating the live training model in place:
```python
# Load a separate copy on CPU for export
export_model = PPO.load(saved_model_path, device="cpu")
torch.onnx.export(export_model.policy, ...)
```
This avoids needing to move the training model back to GPU after export.

**Key files:**
- `src/atom/training/trainers/population/population_persistence.py:115`

**Estimated scope:** Small.

**Done looks like:** `export_failures.jsonl` is empty on a smoke run.

---

## Priority 4: Run Manifest Phase Fix

**Problem:** `run_manifest.json` records `"phase": "curriculum"` even for
full pipeline runs. The manifest is written at curriculum init and never
updated when population training starts.

**Approach:** Either:
- Write manifest once with `"phase": "complete"` when mode is complete, or
- Update manifest at each phase transition

**Key files:**
- `src/atom/training/utils/observability.py` or wherever manifest is written
- `src/atom/training/pipelines/progressive_trainer.py`

**Estimated scope:** Trivial.

**Done looks like:** `run_manifest.json` reports `"phase": "complete"` for
complete runs.

---

## Priority 5: Post-Reset Reporting Label

**Problem:** The final report string in `population_training_loop.py:158`
still says `"Final Rankings (All Time)"`, which is misleading now that ELO
ratings are reset per generation. Rankings are generation-scoped, not all-time.

**Approach:** Change label to `"Final Rankings (Current Generation)"` or
similar. Also audit `print_leaderboard()` in `elo_tracker.py` for the same
wording.

**Estimated scope:** Trivial.

**Done looks like:** Leaderboard output says "Current Generation" not
"All Time".

---

## Open Questions (Not Actionable Yet)

### Training Budget Adequacy

Reward trajectory data (Q1-Q4) from run3 suggests fighters plateau by Q3,
but this signal is confounded by the 1:1 pairing design (now fixed). Revisit
after the next training run with diversity matchmaking to see if the pattern
holds or was an artifact.

### Parent Selection Skew

Run3 parent counts from `lineage_events.jsonl`:
- Rogue_Viper: 12 children
- Rogue_Eagle: 8
- Stone_Jaguar_G1: 4
- Stone_Cobra_G3: 4
- Brisk_Panther_G3: 4
- Prime_Jaguar_G15: 4

Better than run2 (where Rogue_Eagle produced 75% of all children), but still
skewed. The rank-weighted selection is working — top fighters produce more
offspring — but the distribution is steeper than expected. May want to flatten
the weights (e.g., `[N, N-1, ..., 1]` → `[N, N-0.5, ..., 1]`) if diversity
remains an issue after fixing pair diversity.

### Evaluation Seed Determinism

Still using fixed seeds for evaluation matches. The ELO reset broke the
frozen-ranking symptom, but individual matchup outcomes are still deterministic.
Lower priority — revisit if rankings appear artificially stable after pair
diversity fix.
