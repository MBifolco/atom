# Training Observability Plan

This plan defines the minimum instrumentation we need before making larger
changes to curriculum training, population evolution, or the overall training
methodology.

## Why This Exists

Recent full runs show that the current training stack can complete end to end,
but we still lack enough visibility to answer the most important questions with
confidence.

Examples from the March 21, 2026 seeded Colab run:
- curriculum eventually completed, but Level 1 needed more than 6000 episodes to
  graduate
- curriculum hit a recoverable NaN during later stages
- population evolution definitely ran, but original elite fighters still held
  the top slots at the end of training
- later-generation fighters survived, but they did not overtake the incumbent
  elites
- ONNX export failures occurred during generation saves, adding noise to the end
  of each generation

The current logs tell us that something happened. They do not tell us whether
training is effective, sample-efficient, stable, or improving in the right way.

## Core Questions We Need To Answer

### Curriculum

1. Is the curriculum graduate genuinely strong, or just good at the training
   opponents?
2. Does later curriculum training improve generalization, or does it overfit and
   narrow behavior?
3. Are we spending too many episodes in early levels relative to real gains?
4. Are NaNs random runtime instability, or do they correlate with specific
   levels, reward patterns, or policy statistics?

### Population

1. Are child fighters improving on their parents, or just surviving as weaker
   variants?
2. Is selection pressure too conservative, allowing incumbents to dominate
   indefinitely?
3. Are mutations too weak, too destructive, or simply not targeted enough?
4. Are we measuring the right thing, or is all-time ELO hiding the actual
   current-generation picture?
5. Is the curriculum graduate so strong or so specialized that evolution is
   exploring in a poor region of policy space?

## Working Hypotheses

These are the main hypotheses the observability work should help us confirm or
reject.

1. A bad curriculum run can poison population training.
2. Curriculum may be overtraining on stage-specific opponents instead of
   producing a broadly useful base fighter.
3. Population evolution is active, but not producing children that consistently
   outperform incumbent survivors.
4. The current leaderboard is too coarse because it emphasizes all-time ELO more
   than generation-local progress.
5. Export/logging noise is making it harder to see the actual training signal.

## Principles

1. Instrument first, change behavior second.
2. Prefer structured logs over ad hoc print statements.
3. Measure both absolute performance and change over time.
4. Separate training metrics from evaluation metrics.
5. Keep observability lightweight enough to run in Colab by default.
6. Make outputs easy to compare across fixed-seed runs.

## Current Gaps

The current system already provides some useful information:
- curriculum win rate and reward component summaries in
  [curriculum_trainer.py](../src/atom/training/trainers/curriculum_trainer.py)
- population generation summaries and leaderboard prints in
  [population_trainer.py](../src/atom/training/trainers/population/population_trainer.py)
- replay recording hooks in
  [curriculum_components.py](../src/atom/training/trainers/curriculum_components.py)
  and
  [population_training_loop.py](../src/atom/training/trainers/population/population_training_loop.py)
- training reports saved by curriculum in
  [curriculum_trainer.py](../src/atom/training/trainers/curriculum_trainer.py:756)

What is missing:
- no fixed holdout evaluation suite for curriculum checkpoints
- no structured lineage report for population evolution
- no parent-vs-child comparison artifacts
- no generation-local leaderboard snapshots separate from all-time ratings
- no explicit run manifest that captures seed, dependency set, GPU type, and
  major training settings in one place
- no direct metric for champion turnover or survivor retention
- no behavior-diversity metrics beyond ELO spread and win-rate variance
- no clean way to compare one seeded run against another

## Deliverables

The first observability pass should produce these artifacts for every full run.

### Run Manifest

File:
- `output_dir/analysis/run_manifest.json`

Contents:
- git commit / branch
- seed
- Python version
- package versions relevant to training
- GPU type and VRAM
- high-level training config
- whether the run used vmap
- number of parallel fighters
- environment counts

Purpose:
- establish exactly what run we are looking at
- reduce confusion caused by dependency drift or notebook edits

### Curriculum Metrics

Files:
- `output_dir/curriculum/analysis/level_summaries.jsonl`
- `output_dir/curriculum/analysis/holdout_eval.jsonl`
- `output_dir/curriculum/analysis/policy_stats.jsonl`
- `output_dir/curriculum/analysis/failure_events.jsonl`

Each level summary should include:
- level name
- episodes attempted
- wins
- overall win rate
- recent win rate at graduation or failure
- wall-clock time for the level
- timesteps consumed
- reward component aggregates
- whether the level ended by graduation, retry recovery, sanity gate, or max
  budget

Each holdout evaluation record should include:
- checkpoint identifier
- fixed opponent suite results
- win rate against each holdout opponent
- mean damage dealt
- mean damage taken
- mean fight length

Each policy stats record should include:
- checkpoint identifier
- action distribution summary
- stance distribution summary
- action entropy summary
- NaN or invalid-action counters if present

Each failure event should include:
- timestamp
- current level
- global timestep
- error type
- recovery action taken
- whether recovery succeeded

### Population Metrics

Files:
- `output_dir/population/analysis/generation_summary.jsonl`
- `output_dir/population/analysis/lineage_events.jsonl`
- `output_dir/population/analysis/current_leaderboard.jsonl`
- `output_dir/population/analysis/parent_child_matchups.jsonl`
- `output_dir/population/analysis/diversity_metrics.jsonl`
- `output_dir/population/analysis/export_failures.jsonl`

Each generation summary should include:
- generation number
- wall-clock time spent in fighter training
- wall-clock time spent in evaluation
- wall-clock time spent saving/exporting
- episodes per fighter
- total episodes completed
- active population names
- champion name before and after training
- champion turnover indicator
- survivor count carried forward
- child count introduced

Each lineage event should include:
- generation number
- child fighter name
- parent fighter name
- replaced fighter name
- parent ELO at time of mutation
- child post-training ELO
- child post-evaluation rank

Each current leaderboard snapshot should include:
- generation number
- fighter name
- active-generation rank
- active-generation ELO
- all-time ELO
- wins/losses/draws
- lineage label
- whether fighter is incumbent or child

Each parent-child matchup record should include:
- generation number
- parent name
- child name
- result summary over a fixed number of matches
- damage ratio
- whether child outperformed parent

Each diversity record should include:
- ELO range
- ELO std dev
- win-rate variance
- action entropy variance if available
- stance usage divergence if available
- matchup spread / transitivity signal if available

Each export failure record should include:
- generation number
- fighter name
- export target
- exception summary
- whether training artifacts were still saved successfully

## Minimal Evaluation Suites

### Curriculum Holdout Suite

We should define a small fixed suite of opponents that is not identical to the
training schedule.

Suggested structure:
- 2 stationary opponents
- 2 movement-based opponents
- 2 spacing/resource opponents
- 2 advanced behavior opponents
- 2 hardcoded expert opponents

Purpose:
- measure generalization during curriculum
- detect overtraining or over-specialization earlier

### Population Comparison Suite

For each evolution round, run a small fixed evaluation battery:
- incumbent champion vs new best child
- each new child vs its direct parent
- current top 4 round-robin snapshot

Purpose:
- understand whether evolution is creating improvement or just churn

## Metrics We Should Watch Closely

### Curriculum Effectiveness

1. Time to graduate each level
2. Holdout win rate at each checkpoint
3. Reward component drift across levels
4. Action entropy collapse or explosion
5. Damage dealt vs damage taken on holdout fights
6. Number and timing of NaN recoveries

### Population Effectiveness

1. Champion turnover rate
2. Fraction of children surviving more than one evolution cycle
3. Fraction of children beating their direct parent
4. Fraction of children entering top 4
5. Best-child vs incumbent champion head-to-head
6. Diversity trend over generations

## How To Interpret The March 21 Run

The March 21 seeded run suggests these specific follow-up questions:
- why did Level 1 require more than 6000 episodes before graduating?
- why did later levels complete quickly after that slow start?
- why did a curriculum NaN appear late but recover successfully?
- why did evolved children survive, yet fail to break into the top 4?
- why did the final top 4 remain incumbent names while G7 children occupied the
  bottom 4?

That pattern suggests evolution is active but not yet competitive at the top.
The observability work should confirm whether that comes from:
- weak mutation
- too little selective replacement
- unfair incumbent advantage
- poor initialization from curriculum
- or evaluation blind spots

## Implementation Phases

### Phase 1: Structured Run Metadata

Goal:
- make every run reproducible and comparable

Add:
- run manifest
- generation summary JSONL
- curriculum level summary JSONL
- export failure log

Likely hook points:
- [progressive_trainer.py](../src/atom/training/pipelines/progressive_trainer.py)
- [curriculum_trainer.py](../src/atom/training/trainers/curriculum_trainer.py)
- [population_trainer.py](../src/atom/training/trainers/population/population_trainer.py)
- [population_persistence.py](../src/atom/training/trainers/population/population_persistence.py)

### Phase 2: Holdout Evaluation

Goal:
- measure actual fighter quality instead of only training-stage success

Add:
- curriculum checkpoint holdout suite
- population champion vs child suite
- parent vs child evaluation records

Likely hook points:
- [curriculum_components.py](../src/atom/training/trainers/curriculum_components.py)
- [population_evaluation.py](../src/atom/training/trainers/population/population_evaluation.py)

### Phase 3: Lineage and Diversity Tracking

Goal:
- understand whether evolution is producing meaningful exploration

Add:
- lineage event log
- active-generation leaderboard snapshots
- diversity metric snapshots

Likely hook points:
- [population_evolution.py](../src/atom/training/trainers/population/population_evolution.py)
- [population_training_loop.py](../src/atom/training/trainers/population/population_training_loop.py)
- [elo_tracker.py](../src/atom/training/trainers/population/elo_tracker.py)

### Phase 4: Decision-Oriented Dashboards

Goal:
- make it easy to compare runs without reading raw logs

Add:
- compact Markdown or HTML run summary
- curriculum comparison table across seeds
- population lineage table across generations

This can wait until the structured metrics exist.

## Proposed Experiment Matrix

Once the instrumentation exists, run a small fixed-seed matrix.

### Curriculum Matrix

Keep the same seed set for all comparisons.

Compare:
1. current curriculum settings
2. shorter curriculum budget per level
3. lower PPO update cost
4. altered graduation thresholds

Evaluate with:
- holdout suite
- time to graduation
- NaN frequency
- final population performance downstream

### Population Matrix

Keep the same curriculum graduate and same seed set.

Compare:
1. current mutation rate
2. lower mutation rate
3. higher mutation rate
4. evolution every generation vs every 2 generations
5. current evaluation method vs stronger parent-child comparison gate

Evaluate with:
- child survival rate
- top-4 breakthrough rate
- champion turnover
- best-child vs incumbent win rate

## Success Criteria

The observability work is successful if we can answer these questions from
artifacts, not guesswork:
- Did curriculum improve generalization or just training-opponent performance?
- Did evolution create better fighters than incumbents?
- Which parent lineages are actually productive?
- Are we overtraining early curriculum levels?
- Are NaN recoveries random noise or a predictable failure mode?

## Near-Term Recommendation

Before revisiting the full training methodology, implement Phase 1 and the
minimum pieces of Phase 2.

That gives us:
- reproducible run metadata
- structured curriculum outcomes
- structured population generation outcomes
- enough holdout data to decide whether to change curriculum, population,
  or both

Without that layer, methodology changes will be hard to evaluate and easy to
misread.
