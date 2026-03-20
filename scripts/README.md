# Utility Scripts

Lightweight utilities that are useful operationally but are not primary project
entrypoints.

## Layout

- `analysis/`
  - `analyze_population_progress.py`: summarize Elo/diversity trends from a population run
- `training/`
  - `build_registry.py`: rebuild fighter registry metadata from `fighters/`
  - `run_local_baseline.py`: deterministic local baseline training harness
  - `resume_population_training.py`: resume/extend population training from checkpoints
- `montage/`
  - `render_replays.py`: render replay HTML files for manual capture
  - `create_montage.py`: create montage videos from replay telemetry
  - `create_html_montage.py`: generate progressive training HTML montage artifacts
  - `create_html_montage_from_existing.py`: build montage HTML from historical replay index data
- `ops/`
  - `clear_gpu_memory.py`: clear framework caches and print GPU status (primarily ROCm-focused)
  - `check_markdown_links.py`: validate relative markdown links across the repo
  - `setup_gpu.sh`: sourceable local ROCm/JAX environment setup helper
  - `tail_latest_log.sh`: tail newest population training log in `outputs/`

## Example Usage

```bash
python scripts/analysis/analyze_population_progress.py outputs/progressive_YYYYMMDD_HHMMSS
python scripts/training/run_local_baseline.py --mode curriculum --timesteps 10000
python scripts/montage/render_replays.py --run-dir outputs/progressive_YYYYMMDD_HHMMSS
source scripts/ops/setup_gpu.sh
python scripts/ops/clear_gpu_memory.py
python scripts/ops/check_markdown_links.py
```
