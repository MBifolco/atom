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
#   ATOM_DRIVE_REPO_SCAN_UNTRACKED=0 # 1=include slow untracked-file scan in Drive cache
#   ATOM_SKIP_DRIVE_SYNC=0 # 1=skip Drive cache sync entirely and use existing WORK_REPO checkout
#   ATOM_SKIP_PREFLIGHT=0  # 1=skip bootstrap preflight checks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRIVE_REPO="${ATOM_DRIVE_REPO:-/content/drive/MyDrive/dev/atom}"
WORK_REPO="${ATOM_WORK_REPO:-/content/atom}"
BRANCH="${ATOM_BRANCH:-main}"
REPO_URL="${ATOM_REPO_URL:-}"
INSTALL_JAX_CUDA="${ATOM_INSTALL_JAX_CUDA:-1}"
JAX_VERSION="${ATOM_JAX_VERSION:-0.7.2}"
SYNC_MODE="${ATOM_DRIVE_REPO_SYNC_MODE:-stash}"
SKIP_PREFLIGHT="${ATOM_SKIP_PREFLIGHT:-0}"
SCAN_UNTRACKED="${ATOM_DRIVE_REPO_SCAN_UNTRACKED:-0}"
SKIP_DRIVE_SYNC="${ATOM_SKIP_DRIVE_SYNC:-0}"
REQUIREMENTS_FILE="${ATOM_REQUIREMENTS_FILE:-$PROJECT_ROOT/requirements-colab.txt}"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
fi

SECONDS=0

timestamp() {
  date '+%H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

run_step() {
  local label="$1"
  shift
  local started_at=$SECONDS
  log "START: $label"
  if "$@"; then
    log "DONE : $label (${SECONDS-started_at}s)"
  else
    local rc=$?
    log "FAIL : $label (${SECONDS-started_at}s, exit $rc)"
    return $rc
  fi
}

refresh_git_index() {
  git update-index -q --refresh >/dev/null 2>&1 || true
}

has_tracked_changes() {
  refresh_git_index
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    ! git diff-index --quiet HEAD --
  else
    ! git diff --quiet || ! git diff --cached --quiet
  fi
}

find_first_untracked_file() {
  git ls-files --others --exclude-standard | sed -n '1p'
}

cd "$PROJECT_ROOT"

log "Bootstrap configuration:"
log "  DRIVE_REPO=$DRIVE_REPO"
log "  WORK_REPO=$WORK_REPO"
log "  BRANCH=$BRANCH"
log "  SYNC_MODE=$SYNC_MODE"
log "  INSTALL_JAX_CUDA=$INSTALL_JAX_CUDA"
log "  SCAN_UNTRACKED=$SCAN_UNTRACKED"
log "  SKIP_DRIVE_SYNC=$SKIP_DRIVE_SYNC"
log "  REQUIREMENTS_FILE=$REQUIREMENTS_FILE"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  log "Running bootstrap preflight checks..."
  started_at=$SECONDS
  if PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -u -m src.atom.training.utils.colab_preflight --stage bootstrap --strict; then
    log "DONE : bootstrap preflight (${SECONDS-started_at}s)"
  else
    rc=$?
    log "FAIL : bootstrap preflight (${SECONDS-started_at}s, exit $rc)"
    exit "$rc"
  fi
fi

if [[ ! -d "/content/drive" ]]; then
  echo "ERROR: /content/drive not found. Mount Drive first:"
  echo "  from google.colab import drive"
  echo "  drive.mount('/content/drive')"
  exit 1
fi

if [[ "$SKIP_DRIVE_SYNC" == "1" ]]; then
  log "Skipping Drive repo cache sync (ATOM_SKIP_DRIVE_SYNC=1)."
  log "Using existing runtime workspace at $WORK_REPO"
  if [[ ! -d "$WORK_REPO/.git" ]]; then
    echo "ERROR: ATOM_SKIP_DRIVE_SYNC=1 requires an existing git checkout at $WORK_REPO"
    exit 1
  fi
  cd "$WORK_REPO"
else
  if [[ ! -d "$DRIVE_REPO/.git" ]]; then
    if [[ -z "$REPO_URL" ]]; then
      echo "ERROR: First run requires ATOM_REPO_URL (repo clone URL)."
      echo "Example: export ATOM_REPO_URL='https://github.com/<org>/<repo>.git'"
      exit 1
    fi
    run_step "Clone repo into Drive cache" git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$DRIVE_REPO"
  fi

  log "Updating Drive repo cache ($BRANCH)..."
  cd "$DRIVE_REPO"

  git config core.filemode false

  log "Checking Drive repo cache for tracked changes..."
  is_dirty=0
  if has_tracked_changes; then
    is_dirty=1
    log "Tracked changes detected in Drive repo cache."
  else
    log "No tracked changes detected in Drive repo cache."
  fi

  if [[ "$SCAN_UNTRACKED" == "1" ]]; then
    log "Scanning Drive repo cache for untracked files..."
    if [[ -n "$(find_first_untracked_file)" ]]; then
      is_dirty=1
      log "Untracked files detected in Drive repo cache."
    else
      log "No untracked files detected in Drive repo cache."
    fi
  else
    log "Skipping untracked file scan in Drive repo cache (ATOM_DRIVE_REPO_SCAN_UNTRACKED=0)."
  fi

  skip_pull=0
  skip_branch_sync=0
  if [[ "$is_dirty" -eq 1 ]]; then
    echo "Drive repo cache has local changes."
    case "$SYNC_MODE" in
      stash)
        stash_name="colab-bootstrap-$(date +%Y%m%d_%H%M%S)"
        if ! git stash push -u -m "$stash_name" >/dev/null; then
          echo "ERROR: Failed to stash Drive cache changes."
          echo "Try rerunning with: ATOM_DRIVE_REPO_SYNC_MODE=reset"
          exit 1
        fi
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
    log "Re-checking Drive repo cache after sync preparation..."
    still_dirty=0
    if has_tracked_changes; then
      still_dirty=1
    fi
    if [[ "$SCAN_UNTRACKED" == "1" ]] && [[ -n "$(find_first_untracked_file)" ]]; then
      still_dirty=1
    fi
    if [[ "$still_dirty" -eq 1 ]]; then
      echo "ERROR: Drive repo cache is still dirty after sync preparation."
      echo "Current status:"
      git status --short
      echo "Try rerunning with: ATOM_DRIVE_REPO_SYNC_MODE=reset"
      exit 1
    fi
  fi

  if [[ "$skip_branch_sync" -eq 0 ]]; then
    git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    run_step "Fetch origin refs" git fetch --prune origin

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
      run_step "Checkout local branch $BRANCH" git checkout "$BRANCH"
    else
      run_step "Create local branch $BRANCH from origin/$BRANCH" git checkout -b "$BRANCH" "origin/$BRANCH"
    fi
  fi

  if [[ "$skip_pull" -eq 0 ]]; then
    run_step "Fast-forward pull origin/$BRANCH" git pull --ff-only origin "$BRANCH"
  fi

  log "Syncing to local runtime workspace: $WORK_REPO"
  mkdir -p "$WORK_REPO"
  run_step "Sync Drive cache to runtime workspace" rsync -a --delete \
    --exclude ".pytest_cache/" \
    --exclude "__pycache__/" \
    --exclude ".mypy_cache/" \
    --exclude "outputs/" \
    --exclude "htmlcov/" \
    "$DRIVE_REPO"/ "$WORK_REPO"/

  cd "$WORK_REPO"
fi

log "Installing Python dependencies..."
run_step "Upgrade pip" python -m pip install -U pip
run_step "Install $(basename "$REQUIREMENTS_FILE")" python -m pip install -r "$REQUIREMENTS_FILE"

if [[ "$INSTALL_JAX_CUDA" == "1" ]]; then
  log "Installing JAX CUDA wheel (jax==$JAX_VERSION)..."
  run_step "Install JAX CUDA wheel" python -m pip install -U "jax[cuda12]==$JAX_VERSION" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

log "Bootstrap complete in ${SECONDS}s."
log "Working directory: $WORK_REPO"
log "Try a quick smoke test:"
log "  python train_progressive.py --mode quick --device auto --use-vmap --output-dir /content/drive/MyDrive/atom_runs/quick_test"
