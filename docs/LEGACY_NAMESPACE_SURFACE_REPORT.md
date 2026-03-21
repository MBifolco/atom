# Legacy Namespace Surface Report

Snapshot of the intentionally retained legacy `src.*` surface after the current cleanup passes.

## Kept Intentionally

### Root Entry Points

These remain because they are active user-facing workflows:

- `atom_fight.py`
- `train_progressive.py`
- `colab_bootstrap.sh`

### Module Entry Points Still Used Operationally

These remain because notebook or operational flows still call them directly:

- `src/training/utils/colab_preflight.py`
- `src/training/utils/runtime_platform.py`

### Package-Level Compatibility Imports

These remain because they are low-friction compatibility surfaces and still useful while migration continues:

- `src/arena/__init__.py`
- `src/protocol/__init__.py`
- `src/orchestrator/__init__.py`
- `src/evaluator/__init__.py`
- `src/renderer/__init__.py`
- `src/telemetry/__init__.py`
- `src/training/__init__.py`
- `src/training/pipelines/__init__.py`
- `src/training/trainers/__init__.py`
- `src/training/trainers/population/__init__.py`
- `src/training/utils/__init__.py`
- `src/registry/__init__.py`
- `src/coaching/__init__.py`

### Legacy Runtime Leaf Modules Still Kept

These are still referenced by tests and/or current docs, so they are not yet retired:

- `src/arena/arena_1d_jax_jit.py`
- `src/arena/fighter.py`
- `src/arena/world_config.py`
- `src/protocol/combat_protocol.py`
- `src/orchestrator/match_orchestrator.py`

### Legacy Training Leaf Modules Still Kept

These still have meaningful test or compatibility value:

- `src/training/gym_env.py`
- `src/training/opponents_jax.py`
- `src/training/progressive_replay_recorder.py`
- `src/training/replay_recorder.py`
- `src/training/signal_engine.py`
- `src/training/vmap_env_wrapper.py`
- `src/training/trainers/curriculum_components.py`
- `src/training/trainers/curriculum_trainer.py`
- `src/training/trainers/population/population_trainer.py`

## Retired In Recent Passes

These legacy leaf wrappers have now been removed because current surfaces no longer depend on them:

- training utility leaves like `baseline_harness.py`, `determinism.py`, `nan_detector.py`, `stable_ppo.py`, `stable_ppo_config.py`
- runtime helper leaves like `src/evaluator/spectacle_evaluator.py`, `src/renderer/ascii_renderer.py`, `src/renderer/html_renderer.py`, `src/telemetry/replay_store.py`
- population helper leaves like `elo_tracker.py`, `fighter_loader.py`, `parallel_orchestrator.py`, `population_evaluation.py`, `population_evolution.py`, `population_persistence.py`, `population_protocols.py`, `population_training_loop.py`
- unreferenced compatibility leaves for training/population internals such as `train_multicore.py`, `train_population.py`, `train_population_multi.py`, `test_single_match.py`, `test_fighter_loading.py`, `debug_gym_env.py`

## Decision Rule Going Forward

Retire a legacy wrapper when both of these are true:

1. Current repo surfaces no longer reference it.
2. Its remaining value is only stale docs or old tests that can be migrated cleanly.

Keep a legacy wrapper when it still supports an active notebook, CLI, or package-level compatibility path.
