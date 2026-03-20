# Colab Setup Guide

This guide shows how to run Atom training in Google Colab without recloning every session.

## Notebook Option

Use the ready-to-run notebook in this repo:
- `notebooks/Atom_Training_Colab.ipynb`
- Open it in Colab from GitHub and run cells top-to-bottom.

## Workflow Summary

1. Keep a persistent repo cache on Google Drive.
2. Sync that cache into `/content/atom` each Colab session.
3. Run preflight checks before bootstrap/smoke/full/resume stages.
4. Install dependencies and run training from `/content/atom`.
5. Save outputs/checkpoints in Drive so sessions can resume.

## First-Time Setup (in Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Recommended Session Bootstrap

Use the notebook bootstrap cell or run manually from the local repo copy:

```bash
export ATOM_REPO_URL="https://github.com/<org>/<repo>.git"
export ATOM_BRANCH="colab"
bash colab_bootstrap.sh
```

Notes:
- `ATOM_REPO_URL` is required only when Drive cache does not exist yet.
- The script updates your Drive cache (`git pull`) then syncs to `/content/atom`.
- The script runs a bootstrap preflight (`src.training.utils.colab_preflight`) unless `ATOM_SKIP_PREFLIGHT=1`.
- CUDA JAX install is enabled by default (`ATOM_INSTALL_JAX_CUDA=1`).
- JAX CUDA is pinned to a known working version by default (`ATOM_JAX_VERSION=0.7.2`).

## Preflight Checks (Recommended)

Run preflight before long jobs:

```bash
python -m src.training.utils.colab_preflight --stage smoke --output-dir /content/drive/MyDrive/atom_runs/quick_test --require-gpu --strict
```

Other stages:
- `--stage bootstrap`
- `--stage full --output-dir <run_dir>`
- `--stage resume --checkpoint-dir <run_dir>`

## Train

Quick smoke test:

```bash
!python train_progressive.py --mode quick --device auto --use-vmap --output-dir /content/drive/MyDrive/atom_runs/quick_test || true
```

Note: quick mode is mainly an infrastructure check. It may fail curriculum graduation with low timesteps.

Full run:

```bash
!python train_progressive.py --mode complete --device auto --use-vmap --timesteps 2000000 --output-dir /content/drive/MyDrive/atom_runs/run1
```

## Resume After Disconnect

```bash
!python resume_population_training.py \
  --checkpoint-dir /content/drive/MyDrive/atom_runs/run1 \
  --start-gen 8 \
  --total-gens 20 \
  --use-vmap
```

## Runtime Notes

- Colab runtimes are ephemeral; always write checkpoints/logs to Drive.
- Current code auto-detects CUDA vs ROCm and no longer forces ROCm for vmap workers.
- If you want CPU-only runs, skip `--use-vmap`.
- Use the milestone gate checklist in `docs/COLAB_VALIDATION_CHECKLIST.md` for Phase 1/3/5 validation.
