# Post-Run3 Fix Plan

Compiled from Claude + Codex reviews of the March 23, 2026 overnight run (run3).
Run3 validated Phase 1 fixes (stance action space, graduation quality gate, ELO
reset, rank-weighted parent selection). This document tracks the remaining work.

## Run3 Outcomes

**What's working:**
- Curriculum graduate uses all 3 stances (extended ~41%, defending ~26%)
- Combat quality gate passing with real damage (mean 44-60, nonzero rate 94-98%)
- Holdout: 5/10 deterministic wins (stationary, movement, retreater)
- Zero founders in final population — all 8 fighters are descendants
- 14 champion turnovers across 20 generations, 9 unique champions
- Real lineage dynasties (G3 → G15 → G17)

**What's still broken:**
- 1:1 opponent pairing produces mirrored training dynamics
- Holdout checkpoint snapshots are still not distinct per level
- ONNX export fails 91 times per run (cpu/cuda tensor mismatch)
- Run manifest records wrong phase

---

## Priority 1: Population Training Pair Diversity

**Problem:** Each fighter trains against exactly one opponent per generation.
Paired fighters develop mirrored training dynamics (strong evidence from
identical mean_reward and reward quartiles in `generation_summary.jsonl`).
Children paired with other children get negative rewards and don't improve.
Children paired with incumbents do much better.

**Impact:** This is now the clearest training-quality bottleneck. The stance
fix, graduation gate, ELO reset, and parent selection are all working — the
remaining issue is that fighters don't get diverse training experience.

**Approach:** Ensure each fighter trains against multiple opponents per
generation, ideally a mix of incumbents and children. Options:
- Round-robin training (each fighter gets N episodes vs each other fighter)
- Random opponent rotation per episode
- Prioritize pairing children with incumbents (not other children)

**Key files:**
- `src/atom/training/trainers/population/population_training_loop.py:65-86`
- `src/atom/training/trainers/population/population_trainer.py:1068-1105`
  (`create_matchmaking_pairs()`)

**Estimated scope:** Medium — requires rethinking the training loop structure.

---

## Priority 2: Holdout Checkpoint Snapshots

**Problem:** The deferred holdout flush (Phase 2) correctly avoids stale
mid-rollout weights, but levels 1-4 all graduate within the same PPO rollout
cycle. The flush fires on the next `_on_rollout_start`, evaluating all 4
checkpoints against the same (now-updated) model. Results are identical.

This satisfies "don't evaluate stale weights" but does NOT provide distinct
per-level checkpoint comparison, which limits curriculum analysis quality.

**Approach:** Save the model to disk at each graduation (in `advance_level`),
then have the flush evaluate each saved snapshot independently. This way
level 1 holdout uses the level-1-graduated weights, level 2 uses level-2, etc.

**Key files:**
- `src/atom/training/trainers/curriculum_trainer.py` — `advance_level()`,
  `_flush_pending_holdouts()`, `_record_holdout_evaluation()`

**Estimated scope:** Small-medium — save model to temp path at graduation,
load it in flush for holdout eval.

---

## Priority 3: ONNX Export Fix

**Problem:** `torch.onnx.export` fails with `RuntimeError: Unhandled
FakeTensor Device Propagation for aten.mm.default, found two different
devices cpu, cuda:0`. This happens because the model has mixed cpu/cuda
tensors. 91 failures in run3's `export_failures.jsonl`.

**Approach:** Move model to CPU before ONNX export:
```python
model.policy.to("cpu")
torch.onnx.export(...)
```

**Key files:**
- `src/atom/training/trainers/population/population_persistence.py:115`

**Estimated scope:** Small — 1-2 lines.

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

**Estimated scope:** Trivial — one-line fix.

---

## Open Questions (Not Actionable Yet)

### Training Budget Adequacy

Reward trajectory data (Q1-Q4) from run3 suggests fighters plateau by Q3,
but this signal is confounded by the 1:1 pairing design — paired fighters
converge to mirror each other, which looks like a plateau but might actually
be an artifact. Revisit after fixing pair diversity.

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
