# Population Training Speedup Plan

This document captures the current bottlenecks in population training, the
highest-leverage speedup opportunities, and an aggressive test plan for Colab
GPU runs.

## Goal

Curriculum training is already fast enough for now. The next optimization target
is population training, especially the self-play/evolution loop.

## Current Bottlenecks

### 1. GPU population training still defaults to sequential

When `use_vmap` is enabled and `n_parallel_fighters` is not provided, the
population trainer forces `n_parallel_fighters = 1` to avoid ROCm OOM issues.

Code references:
- [src/atom/training/trainers/population/population_trainer.py](../src/atom/training/trainers/population/population_trainer.py:816)

### 2. Population vmap env count is still based on an old ROCm-era assumption

Population training currently uses `n_vmap_envs=45`, with comments indicating it
was reduced to fit `8` fighters in `8 GB` of VRAM.

Code references:
- [src/atom/training/pipelines/progressive_trainer.py](../src/atom/training/pipelines/progressive_trainer.py:292)
- [src/atom/training/pipelines/progressive_trainer.py](../src/atom/training/pipelines/progressive_trainer.py:356)

### 3. Sequential GPU still goes through the process-pool path

Even when only one fighter is trained at a time, the code still uses the
parallel orchestration path:
- save model to temp zip
- send work to a subprocess
- reload the model from temp zip

That adds avoidable process and serialization overhead.

Code references:
- [src/atom/training/trainers/population/population_trainer.py](../src/atom/training/trainers/population/population_trainer.py:1097)
- [src/atom/training/trainers/population/parallel_orchestrator.py](../src/atom/training/trainers/population/parallel_orchestrator.py:71)
- [src/atom/training/trainers/population/parallel_orchestrator.py](../src/atom/training/trainers/population/parallel_orchestrator.py:172)

### 4. Every generation includes non-trivial evaluation and persistence work

After each generation, the trainer currently:
- runs evaluation matches
- saves the full generation
- exports qualifying fighters

Code references:
- [src/atom/training/trainers/population/population_trainer.py](../src/atom/training/trainers/population/population_trainer.py:1218)
- [src/atom/training/trainers/population/population_trainer.py](../src/atom/training/trainers/population/population_trainer.py:1289)
- [src/atom/training/trainers/population/population_trainer.py](../src/atom/training/trainers/population/population_trainer.py:1466)
- [src/atom/training/trainers/population/population_persistence.py](../src/atom/training/trainers/population/population_persistence.py:41)
- [src/atom/training/trainers/population/population_persistence.py](../src/atom/training/trainers/population/population_persistence.py:75)

## GPU Memory Reference

These are the useful working numbers for planning population runs:

| GPU | Typical VRAM |
|-----|--------------|
| NVIDIA L4 | 24 GB |
| NVIDIA A100 | 40 GB or 80 GB depending on SKU |

For Colab, treat A100 as "usually 40 GB unless the environment explicitly shows
otherwise."

## Important Observation

The current code comment says:

- `8 fighters x 45 envs = 360 total envs (~7.2 GB VRAM)`

Code reference:
- [src/atom/training/pipelines/progressive_trainer.py](../src/atom/training/pipelines/progressive_trainer.py:293)

If that estimate is even roughly correct, then:

- an **L4 (24 GB)** should be able to attempt **8 parallel fighters at the
  current 45-env setting**
- an **A100 (40 GB)** should have plenty of headroom for more aggressive env
  counts

If those runs still fail, the likely constraint is not raw env memory alone.
It is more likely:
- per-process JAX/PyTorch runtime duplication
- process startup overhead
- model serialization/reload overhead
- fragmentation or allocator behavior

## Aggressive First Test Strategy

The first sweep should isolate **fighter parallelism** before changing env
count. That gives a clean answer to "can Colab handle more than one fighter at a
time with the current code?"

Hold constant:
- `n_vmap_envs = 45`
- same population size
- same generations / episodes-per-generation for each comparison

Sweep:
1. `--n-parallel-fighters 2`
2. `--n-parallel-fighters 4`
3. `--n-parallel-fighters 8`

This sweep can be run immediately with the current CLI. By contrast, `n_vmap_envs` is still hardcoded to `45` in the progressive pipeline, so the env-count sweep is a second step unless we expose a flag or temporarily patch the code.

Recommended mindset:
- start at `8` if we want the fastest answer
- if it fails, walk back to `4`
- if `4` fails, walk back to `2`

### L4 Starting Recommendation

Start with:

```bash
python train_progressive.py \
  --mode complete \
  --use-vmap \
  --n-parallel-fighters 8
```

Reason:
- current env count is already conservative
- the code's own VRAM comment suggests this should be plausible
- if it fails, we learn quickly whether the real issue is subprocess/runtime
  duplication instead of nominal env memory

### A100 Starting Recommendation

Start with:

```bash
python train_progressive.py \
  --mode complete \
  --use-vmap \
  --n-parallel-fighters 8
```

If stable, the next lever is increasing env count, not fighter count, since the
population size is already `8`.

## Second Sweep: Increase vmap envs Once Parallelism Works

If `8` parallel fighters is stable at `45` envs, then increase the env count.

### L4 Suggested Env Sweep

1. `8 fighters x 45 envs`
2. `8 fighters x 60 envs`
3. `8 fighters x 90 envs`

Only continue upward if:
- no OOM
- no severe slowdown from allocator thrash
- rollout throughput still improves

### A100 40 GB Suggested Env Sweep

1. `8 fighters x 90 envs`
2. `8 fighters x 120 envs`
3. `8 fighters x 180 envs`

### A100 80 GB Suggested Env Sweep

1. `8 fighters x 120 envs`
2. `8 fighters x 180 envs`
3. `8 fighters x 250 envs`

`250` matters because it gets population training closer to curriculum's
high-throughput shape.

## What To Measure

For each run, record:

1. GPU type and VRAM
2. `n_parallel_fighters`
3. `n_vmap_envs`
4. population size
5. episodes per generation
6. wall-clock time per generation
7. wall-clock time spent in:
   - fighter training
   - evaluation
   - saving/export
8. failures:
   - OOM
   - NaNs
   - subprocess crashes
   - major slowdown / thrashing

The main output we want is a table of:

| GPU | parallel fighters | vmap envs | gen time | stable? | notes |

## Highest-Leverage Code Improvements

These are the next changes to implement after the first empirical sweep.

### 1. Bypass process-pool orchestration when `n_parallel_fighters == 1`

This is the highest-confidence quick win.

Instead of:
- save zip
- subprocess
- reload zip

Use the existing in-process sequential training path directly.

Expected benefit:
- faster sequential population training
- simpler failure mode
- less temp artifact churn

### 2. Make defaults runtime-aware

Current defaults were chosen for ROCm safety. They should become backend-aware:

- `rocm`: conservative defaults
- `cuda`: aggressive defaults

Suggested policy:
- ROCm default: `n_parallel_fighters = 1`
- CUDA default: `n_parallel_fighters = 2` or `4`

### 3. Reduce evaluation cost during active evolution

Current loop runs evaluation matches every generation.

Suggested options:
- evaluation every generation with `1` match per pair
- fuller evaluation every `N` generations

### 4. Reduce persistence/export frequency

Current loop saves and exports every generation.

Suggested options:
- save every `N` generations
- export ONNX only on final generation or every `N` generations
- export only top `K` or threshold-crossing fighters

### 5. Add CUDA-specific autotuning for `n_vmap_envs`

Instead of a single hardcoded `45`, probe upward until throughput stops
improving or memory becomes unstable.

### 6. Long-term: persistent worker processes

The larger architectural speedup is to keep worker processes and models warm
instead of saving/reloading model artifacts each generation.

## Suggested Execution Order

1. Run aggressive parallel-fighter sweep at current `45` envs
2. If stable, sweep env count upward
3. Implement in-process sequential fallback for `n_parallel_fighters == 1`
4. Make runtime-aware defaults
5. Trim evaluation and export cadence
6. Consider persistent workers if population training is still the bottleneck

## Recommendation

Do not tiptoe.

For Colab:
- test `--n-parallel-fighters 8` first
- keep `n_vmap_envs=45` for the first pass
- walk back only if the run actually fails

That gives the fastest answer to whether the current bottleneck is really VRAM,
or whether it is the old orchestration model we are still carrying forward.
