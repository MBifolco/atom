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
#   ATOM_JAX_VERSION=0.7.2
#   ATOM_DRIVE_REPO_SYNC_MODE=stash  # stash|reset|skip_pull
#   ATOM_SKIP_PREFLIGHT=0  # 1=skip bootstrap preflight checks

DRIVE_REPO="${ATOM_DRIVE_REPO:-/content/drive/MyDrive/dev/atom}"
WORK_REPO="${ATOM_WORK_REPO:-/content/atom}"
BRANCH="${ATOM_BRANCH:-main}"
REPO_URL="${ATOM_REPO_URL:-}"
INSTALL_JAX_CUDA="${ATOM_INSTALL_JAX_CUDA:-1}"
JAX_VERSION="${ATOM_JAX_VERSION:-0.7.2}"
SYNC_MODE="${ATOM_DRIVE_REPO_SYNC_MODE:-stash}"
SKIP_PREFLIGHT="${ATOM_SKIP_PREFLIGHT:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  echo "Running bootstrap preflight checks..."
  PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m src.training.utils.colab_preflight --stage bootstrap --strict
fi

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

# Handle dirty working tree in Drive cache before checkout/pull.
is_dirty=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  is_dirty=1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  is_dirty=1
fi

skip_pull=0
skip_branch_sync=0
if [[ "$is_dirty" -eq 1 ]]; then
  echo "Drive repo cache has local changes."
  case "$SYNC_MODE" in
    stash)
      stash_name="colab-bootstrap-$(date +%Y%m%d_%H%M%S)"
      git stash push -u -m "$stash_name" >/dev/null || true
      echo "Stashed local changes as: $stash_name"
      ;;
    reset)
      echo "Discarding local changes in Drive cache (SYNC_MODE=reset)..."
      git reset --hard HEAD
      git clean -fd
      ;;
    skip_pull)
      echo "Skipping branch sync and pull because repo is dirty (SYNC_MODE=skip_pull)."
      echo "Working tree remains on branch: $(git rev-parse --abbrev-ref HEAD)"
      skip_pull=1
      skip_branch_sync=1
      ;;
    *)
      echo "ERROR: Unknown ATOM_DRIVE_REPO_SYNC_MODE='$SYNC_MODE' (use stash|reset|skip_pull)."
      exit 1
      ;;
  esac
fi

if [[ "$skip_branch_sync" -eq 0 ]]; then
  # Drive cache may have been initialized with --single-branch, which limits
  # remote.origin.fetch to one branch. Expand it so switching branches works.
  git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
  git fetch --prune origin

  if ! git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    echo "ERROR: Remote branch '$BRANCH' not found on origin."
    echo "Available remote branches:"
    git for-each-ref --format='%(refname:short)' refs/remotes/origin/ \
      | sed 's#^origin/##' \
      | grep -v '^HEAD$' \
      | sed 's#^#  #'
    exit 1
  fi

  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH" "origin/$BRANCH"
  fi
fi

if [[ "$skip_pull" -eq 0 ]]; then
  git pull --ff-only origin "$BRANCH"
fi

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
  echo "Installing JAX CUDA wheel (jax==$JAX_VERSION)..."
  python -m pip install -U "jax[cuda12]==$JAX_VERSION" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

echo
echo "Bootstrap complete."
echo "Working directory: $WORK_REPO"
echo "Try a quick smoke test:"
echo "  python train_progressive.py --mode quick --device auto --use-vmap --output-dir /content/drive/MyDrive/atom_runs/quick_test"
