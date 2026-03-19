# Colab Setup Guide

This guide shows how to run Atom training in Google Colab without recloning every session.

## Notebook Option

Use the ready-to-run notebook in this repo:
- `notebooks/Atom_Training_Colab.ipynb`
- Open it in Colab from GitHub and run cells top-to-bottom.

## Workflow Summary

1. Keep a persistent repo cache on Google Drive.
2. Sync that cache into `/content/atom` each Colab session.
3. Install dependencies and run training from `/content/atom`.
4. Save outputs/checkpoints in Drive so sessions can resume.

## First-Time Setup (in Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/<org>/<repo>.git /content/atom
```

## Recommended Session Bootstrap

Use the project script (run after Drive is mounted):

```bash
%cd /content/atom
!export ATOM_REPO_URL="https://github.com/<org>/<repo>.git" && bash colab_bootstrap.sh
```

Notes:
- `ATOM_REPO_URL` is only required the first time (when Drive cache does not exist yet).
- The script updates your Drive cache (`git pull`) then syncs to `/content/atom`.
- CUDA JAX install is enabled by default (`ATOM_INSTALL_JAX_CUDA=1`).
- JAX CUDA is pinned to a known working version by default (`ATOM_JAX_VERSION=0.7.2`).

## Train

Quick smoke test:

```bash
!python train_progressive.py --mode quick --device cuda --use-vmap --output-dir /content/drive/MyDrive/atom_runs/quick_test
```

Full run:

```bash
!python train_progressive.py --mode complete --device cuda --use-vmap --output-dir /content/drive/MyDrive/atom_runs/run1
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
