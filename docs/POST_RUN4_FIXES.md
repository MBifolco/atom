# Post-Run4 Fixes

Three targeted fixes identified from the March 23 overnight run (run4).
Ordered by expected impact on training diversity.

## Fix 1: Stop Truncating Opponents (Highest Impact)

**Problem:** The matchmaker assigns 3 opponents per fighter, but
`parallel_orchestrator.py:219` clips to `opponents[:n_envs_per_fighter]`
where `n_envs_per_fighter` defaults to 2. Only 2 of 3 assigned opponents
reach the training worker.

**Fix:** Remove the truncation. The vmap env already distributes opponent
models across environments via round-robin. All assigned opponents should
be passed through.

**Files:** `parallel_orchestrator.py:219`

**Done looks like:** `generation_summary.jsonl` shows `opponent_names`
length matching the matchmaker's `opponents_per_fighter` setting (3).

## Fix 2: Per-Fighter Training Seeds (High Impact)

**Problem:** `_create_vmap_training_environment()` hardcodes `seed=42` for
every fighter. Combined with identical starting weights (curriculum
graduate), this produces byte-identical reward trajectories for fighters
with the same opponents.

**Fix:** Pass a per-fighter seed through the training task tuple:
`seed = base_seed + fighter_index * 1000 + generation`.
Add seed to `TrainingTask` tuple and `_create_vmap_training_environment`.

**Files:**
- `parallel_orchestrator.py` — add seed to TrainingTask, compute per-task
- `population_trainer.py:443` — use seed param instead of `42`
- `population_trainer.py:504` — add seed to worker function signature

**Done looks like:** Fighters with the same opponents no longer produce
byte-identical reward trajectories in `generation_summary.jsonl`.

## Fix 3: ONNX Export — Switch to Legacy Path

**Problem:** The old mixed-device error (cpu/cuda:0) is fixed. The current
74 failures are `torch.export.export` tracing failures — an upstream
incompatibility between `torch.onnx.export(dynamo=True)` (default in torch
2.10) and SB3's `ActorCriticPolicy`. This is not our bug.

**Fix:** Force legacy ONNX export by passing `dynamo_export=False` or using
the older `torch.onnx.export` API directly without the new dynamo path.

**Files:** `population_persistence.py:115`

**Done looks like:** `export_failures.jsonl` is empty.

## Testing Strategy (Without Full Training Runs)

### Unit tests for opponent count fidelity
- Build training tasks from a matchmaker output with 3 opponents
- Verify the task's `opponent_data` has length 3 (not truncated to 2)

### Unit tests for seed uniqueness
- Build training tasks for 8 fighters
- Verify each task has a distinct seed value

### Short integration test for training diversity
- Run a 2-generation population training with `--mode quick` equivalent
- Parse generation_summary.jsonl
- Assert that fighters with different seeds have different reward trajectories
- This can run in ~60 seconds on CPU, not 3 hours on GPU
