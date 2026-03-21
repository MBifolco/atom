# Namespace Map

Quick before/after reference for the namespace migration.

## Preferred Rule

For new code and current docs, prefer `src.atom.*`.
Use legacy `src.*` only where an intentional compatibility surface still exists.

## Common Import Migrations

| Before | After |
| --- | --- |
| `from src.arena import WorldConfig` | `from src.atom.runtime.arena import WorldConfig` |
| `from src.arena.arena_1d_jax_jit import Arena1DJAXJit` | `from src.atom.runtime.arena.arena_1d_jax_jit import Arena1DJAXJit` |
| `from src.protocol.combat_protocol import generate_snapshot` | `from src.atom.runtime.protocol.combat_protocol import generate_snapshot` |
| `from src.orchestrator.match_orchestrator import MatchOrchestrator` | `from src.atom.runtime.orchestrator.match_orchestrator import MatchOrchestrator` |
| `from src.evaluator import SpectacleEvaluator` | `from src.atom.runtime.evaluator import SpectacleEvaluator` |
| `from src.renderer import HtmlRenderer` | `from src.atom.runtime.renderer import HtmlRenderer` |
| `from src.telemetry import load_replay` | `from src.atom.runtime.telemetry import load_replay` |
| `from src.training.gym_env import AtomCombatEnv` | `from src.atom.training.gym_env import AtomCombatEnv` |
| `from src.training.pipelines import ProgressiveTrainer` | `from src.atom.training.pipelines import ProgressiveTrainer` |
| `from src.training.trainers.curriculum_trainer import CurriculumTrainer` | `from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer` |
| `from src.training.trainers.population.population_trainer import PopulationTrainer` | `from src.atom.training.trainers.population.population_trainer import PopulationTrainer` |
| `from src.training.utils.runtime_platform import detect_runtime_platform` | `from src.atom.training.utils.runtime_platform import detect_runtime_platform` |
| `from src.registry import FighterRegistry` | `from src.atom.registry import FighterRegistry` |
| `from src.coaching import CoachingWrapper` | `from src.atom.coaching import CoachingWrapper` |

## Entry Point Migrations

| Before | After |
| --- | --- |
| `python atom_fight.py ...` | `python -m apps.cli.atom_fight ...` |
| `python train_progressive.py ...` | `python -m apps.training.train_progressive ...` |
| `uvicorn web.app:app --reload` | `uvicorn apps.web.app:app --reload` |
| `bash colab_bootstrap.sh` | `bash scripts/colab/bootstrap.sh` |

## Notes

- Root entrypoints still exist as compatibility wrappers.
- Some package-level `src.*` imports still work intentionally during the migration.
- See [Namespace Migration Policy](NAMESPACE_MIGRATION_POLICY.md) for retirement rules.
- See [Legacy Namespace Surface Report](LEGACY_NAMESPACE_SURFACE_REPORT.md) for what old surfaces remain on purpose.
