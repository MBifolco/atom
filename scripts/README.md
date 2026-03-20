# Utility Scripts

Lightweight utilities that are useful operationally but are not primary project
entrypoints.

## Layout

- `analysis/`
  - `analyze_population_progress.py`: summarize Elo/diversity trends from a population run
- `ops/`
  - `clear_gpu_memory.py`: clear framework caches and print GPU status (primarily ROCm-focused)

## Example Usage

```bash
python scripts/analysis/analyze_population_progress.py outputs/progressive_YYYYMMDD_HHMMSS
python scripts/ops/clear_gpu_memory.py
```
