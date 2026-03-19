#!/usr/bin/env bash
set -euo pipefail

# Colab bootstrap for Atom training.
# Usage (inside Colab after mounting Drive):
#   export ATOM_REPO_URL="https://github.com/<org>/<repo>.git"  # first run only
#   bash colab_bootstrap.sh
#
# Optional env vars:
#   ATOM_DRIVE_REPO=/content/drive/MyDrive/dev/atom
#   ATOM_WORK_REPO=/content/atom
#   ATOM_BRANCH=main
#   ATOM_INSTALL_JAX_CUDA=1   # 1=install JAX CUDA wheel, 0=skip

DRIVE_REPO="${ATOM_DRIVE_REPO:-/content/drive/MyDrive/dev/atom}"
WORK_REPO="${ATOM_WORK_REPO:-/content/atom}"
BRANCH="${ATOM_BRANCH:-main}"
REPO_URL="${ATOM_REPO_URL:-}"
INSTALL_JAX_CUDA="${ATOM_INSTALL_JAX_CUDA:-1}"

if [[ ! -d "/content/drive" ]]; then
  echo "ERROR: /content/drive not found. Mount Drive first:"
  echo "  from google.colab import drive"
  echo "  drive.mount('/content/drive')"
  exit 1
fi

if [[ ! -d "$DRIVE_REPO/.git" ]]; then
  if [[ -z "$REPO_URL" ]]; then
    echo "ERROR: First run requires ATOM_REPO_URL (repo clone URL)."
    echo "Example: export ATOM_REPO_URL='https://github.com/<org>/<repo>.git'"
    exit 1
  fi
  echo "Cloning repo into Drive cache: $DRIVE_REPO"
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$DRIVE_REPO"
fi

echo "Updating Drive repo cache ($BRANCH)..."
cd "$DRIVE_REPO"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "Syncing to local runtime workspace: $WORK_REPO"
mkdir -p "$WORK_REPO"
rsync -a --delete \
  --exclude ".pytest_cache/" \
  --exclude "__pycache__/" \
  --exclude ".mypy_cache/" \
  --exclude "outputs/" \
  --exclude "htmlcov/" \
  "$DRIVE_REPO"/ "$WORK_REPO"/

cd "$WORK_REPO"
echo "Installing Python dependencies..."
python -m pip install -U pip
python -m pip install -r requirements.txt

if [[ "$INSTALL_JAX_CUDA" == "1" ]]; then
  echo "Installing JAX CUDA wheel..."
  python -m pip install -U "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

echo
echo "Bootstrap complete."
echo "Working directory: $WORK_REPO"
echo "Try a quick smoke test:"
echo "  python train_progressive.py --mode quick --device cuda --use-vmap"
