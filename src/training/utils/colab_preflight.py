"""
Colab runtime preflight checks for training workflows.

Designed to fail fast with actionable fixes before long-running jobs start.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .runtime_platform import detect_runtime_platform

VALID_STAGES = {"bootstrap", "smoke", "full", "resume"}
VALID_SYNC_MODES = {"stash", "reset", "skip_pull"}
PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    action: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    stage: str
    checks: Sequence[CheckResult]

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == WARN]

    @property
    def passed(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == PASS]

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "ok": self.ok,
            "summary": {
                "passed": len(self.passed),
                "warnings": len(self.warnings),
                "failures": len(self.failures),
            },
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "action": check.action,
                }
                for check in self.checks
            ],
        }


def run_preflight(
    *,
    stage: str,
    env: Mapping[str, str] | None = None,
    output_dir: str | None = None,
    checkpoint_dir: str | None = None,
    require_gpu: bool = False,
) -> PreflightReport:
    """Run preflight checks for a given workflow stage."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Expected one of: {sorted(VALID_STAGES)}")

    runtime_env = dict(os.environ if env is None else env)
    checks: list[CheckResult] = []

    _append_python_check(checks)

    repo_url = runtime_env.get("ATOM_REPO_URL", "").strip()
    branch = runtime_env.get("ATOM_BRANCH", "main").strip()
    drive_repo = runtime_env.get("ATOM_DRIVE_REPO", "/content/drive/MyDrive/dev/atom")
    work_repo = runtime_env.get("ATOM_WORK_REPO", "/content/atom")
    sync_mode = runtime_env.get("ATOM_DRIVE_REPO_SYNC_MODE", "stash").strip()

    _append_sync_mode_check(checks, sync_mode=sync_mode)
    _append_drive_mount_check(checks, drive_repo=drive_repo)

    if stage == "bootstrap":
        _append_command_check(checks, "git")
        _append_command_check(checks, "rsync")
        _append_branch_check(checks, branch=branch)
        _append_repo_url_check(checks, repo_url=repo_url, drive_repo=drive_repo)
        _append_work_repo_parent_check(checks, work_repo=work_repo)

    if stage in {"smoke", "full", "resume"}:
        _append_work_repo_check(checks, work_repo=work_repo)
        _append_train_entrypoint_check(checks, work_repo=work_repo)

    if stage in {"smoke", "full"}:
        _append_output_dir_check(checks, output_dir=output_dir)

    if stage == "resume":
        _append_resume_checkpoint_check(checks, checkpoint_dir=checkpoint_dir)

    if require_gpu:
        _append_gpu_check(checks)

    return PreflightReport(stage=stage, checks=checks)


def render_report(report: PreflightReport) -> str:
    """Render report to human-readable text."""
    lines: list[str] = [f"Colab preflight ({report.stage})"]
    for check in report.checks:
        label = check.status.upper()
        lines.append(f"[{label}] {check.name}: {check.message}")
        if check.action:
            lines.append(f"        fix: {check.action}")

    lines.append(
        "Summary: "
        f"{len(report.passed)} passed, {len(report.warnings)} warnings, {len(report.failures)} failures"
    )
    return "\n".join(lines)


def _append_python_check(checks: list[CheckResult]) -> None:
    if sys.version_info >= (3, 10):
        checks.append(CheckResult("python-version", PASS, f"{sys.version.split()[0]}"))
        return
    checks.append(
        CheckResult(
            "python-version",
            FAIL,
            f"Python {sys.version.split()[0]} detected (requires >= 3.10).",
            action="Use a Colab runtime with Python 3.10+.",
        )
    )


def _append_sync_mode_check(checks: list[CheckResult], *, sync_mode: str) -> None:
    if sync_mode in VALID_SYNC_MODES:
        checks.append(CheckResult("drive-sync-mode", PASS, f"{sync_mode}"))
        return
    checks.append(
        CheckResult(
            "drive-sync-mode",
            FAIL,
            f"Unsupported ATOM_DRIVE_REPO_SYNC_MODE='{sync_mode}'.",
            action="Set ATOM_DRIVE_REPO_SYNC_MODE to one of: stash, reset, skip_pull.",
        )
    )


def _append_drive_mount_check(checks: list[CheckResult], *, drive_repo: str) -> None:
    drive_root = Path("/content/drive")
    drive_repo_path = Path(drive_repo)

    if str(drive_repo_path).startswith("/content/drive") and not drive_root.exists():
        checks.append(
            CheckResult(
                "drive-mounted",
                FAIL,
                f"{drive_root} not found.",
                action="Run: from google.colab import drive; drive.mount('/content/drive')",
            )
        )
        return

    parent = drive_repo_path.parent
    if parent.exists():
        checks.append(CheckResult("drive-mounted", PASS, f"Drive parent exists: {parent}"))
        return

    checks.append(
        CheckResult(
            "drive-mounted",
            WARN,
            f"Drive repo parent does not exist yet: {parent}",
            action="Confirm your ATOM_DRIVE_REPO path is correct for this runtime.",
        )
    )


def _append_command_check(checks: list[CheckResult], command: str) -> None:
    path = shutil.which(command)
    if path:
        checks.append(CheckResult(f"command-{command}", PASS, f"found at {path}"))
        return
    checks.append(
        CheckResult(
            f"command-{command}",
            FAIL,
            f"Required command '{command}' not found in PATH.",
            action=f"Install '{command}' or switch to a runtime image that includes it.",
        )
    )


def _append_branch_check(checks: list[CheckResult], *, branch: str) -> None:
    if branch and "<" not in branch and ">" not in branch:
        checks.append(CheckResult("repo-branch", PASS, branch))
        return
    checks.append(
        CheckResult(
            "repo-branch",
            FAIL,
            f"Invalid ATOM_BRANCH='{branch}'.",
            action="Set ATOM_BRANCH to a real branch name, e.g. 'colab'.",
        )
    )


def _append_repo_url_check(checks: list[CheckResult], *, repo_url: str, drive_repo: str) -> None:
    drive_git = Path(drive_repo) / ".git"
    if drive_git.exists():
        checks.append(
            CheckResult(
                "repo-url",
                PASS,
                "Drive cache already initialized; ATOM_REPO_URL is optional for this run.",
            )
        )
        return

    if not repo_url:
        checks.append(
            CheckResult(
                "repo-url",
                FAIL,
                "ATOM_REPO_URL is missing and Drive cache is not initialized.",
                action="Set ATOM_REPO_URL, e.g. 'https://github.com/<org>/<repo>.git'.",
            )
        )
        return

    if "<org>" in repo_url or "<repo>" in repo_url:
        checks.append(
            CheckResult(
                "repo-url",
                FAIL,
                f"ATOM_REPO_URL still contains placeholders: {repo_url}",
                action="Replace <org>/<repo> with your actual GitHub path.",
            )
        )
        return

    if not (
        repo_url.startswith("https://")
        or repo_url.startswith("git@")
        or repo_url.startswith("ssh://")
    ):
        checks.append(
            CheckResult(
                "repo-url",
                WARN,
                f"Unusual repo URL format: {repo_url}",
                action="Use a standard HTTPS or SSH git URL if clone fails.",
            )
        )
        return

    checks.append(CheckResult("repo-url", PASS, repo_url))


def _append_work_repo_parent_check(checks: list[CheckResult], *, work_repo: str) -> None:
    parent = Path(work_repo).parent
    if parent.exists():
        checks.append(CheckResult("work-repo-parent", PASS, f"{parent}"))
        return
    checks.append(
        CheckResult(
            "work-repo-parent",
            FAIL,
            f"Parent directory for ATOM_WORK_REPO is missing: {parent}",
            action="Set ATOM_WORK_REPO to a path under an existing directory (for Colab: /content/atom).",
        )
    )


def _append_work_repo_check(checks: list[CheckResult], *, work_repo: str) -> None:
    repo = Path(work_repo)
    if repo.exists() and repo.is_dir():
        checks.append(CheckResult("work-repo", PASS, str(repo)))
        return
    checks.append(
        CheckResult(
            "work-repo",
            FAIL,
            f"Working repo not found: {repo}",
            action="Run the bootstrap cell first to clone/sync the repository.",
        )
    )


def _append_train_entrypoint_check(checks: list[CheckResult], *, work_repo: str) -> None:
    train_file = Path(work_repo) / "train_progressive.py"
    if train_file.exists():
        checks.append(CheckResult("train-entrypoint", PASS, str(train_file)))
        return
    checks.append(
        CheckResult(
            "train-entrypoint",
            FAIL,
            f"Missing training entrypoint: {train_file}",
            action="Re-run bootstrap to sync the latest repository into ATOM_WORK_REPO.",
        )
    )


def _append_output_dir_check(checks: list[CheckResult], *, output_dir: str | None) -> None:
    if not output_dir:
        checks.append(
            CheckResult(
                "output-dir",
                FAIL,
                "Output directory was not provided.",
                action="Provide --output-dir under /content/drive/MyDrive/... for persistence.",
            )
        )
        return

    output_path = Path(output_dir)
    parent = output_path.parent
    if not parent.exists():
        checks.append(
            CheckResult(
                "output-dir",
                WARN,
                f"Parent directory does not exist yet: {parent}",
                action="Create the parent folder in Drive or choose an existing output path.",
            )
        )
    else:
        checks.append(CheckResult("output-dir", PASS, str(output_path)))

    if not str(output_path).startswith("/content/drive"):
        checks.append(
            CheckResult(
                "output-persistence",
                WARN,
                f"Output path is not on Drive: {output_path}",
                action="Use /content/drive/MyDrive/... to preserve checkpoints across runtime restarts.",
            )
        )


def _append_resume_checkpoint_check(checks: list[CheckResult], *, checkpoint_dir: str | None) -> None:
    if not checkpoint_dir:
        checks.append(
            CheckResult(
                "resume-checkpoint-dir",
                FAIL,
                "Checkpoint directory was not provided.",
                action="Set --checkpoint-dir to your Drive run folder (e.g. /content/drive/MyDrive/atom_runs/run1).",
            )
        )
        return

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        checks.append(
            CheckResult(
                "resume-checkpoint-dir",
                FAIL,
                f"Checkpoint directory not found: {checkpoint_path}",
                action="Confirm the run path and that training artifacts were written to Drive.",
            )
        )
        return

    markers = (
        list(checkpoint_path.glob("**/checkpoint_*.zip"))
        + list(checkpoint_path.glob("**/generation_*.json"))
        + list(checkpoint_path.glob("**/final_population.json"))
    )
    if markers:
        checks.append(
            CheckResult(
                "resume-checkpoint-dir",
                PASS,
                f"Found {len(markers)} checkpoint marker files under {checkpoint_path}",
            )
        )
        return

    checks.append(
        CheckResult(
            "resume-checkpoint-dir",
            WARN,
            f"No checkpoint marker files found under {checkpoint_path}",
            action="If this is a fresh run folder, complete a training run before using resume.",
        )
    )


def _append_gpu_check(checks: list[CheckResult]) -> None:
    platform = detect_runtime_platform()
    if platform in {"cuda", "rocm"}:
        checks.append(CheckResult("gpu-runtime", PASS, f"Detected accelerator platform: {platform}"))
        return
    checks.append(
        CheckResult(
            "gpu-runtime",
            FAIL,
            "No CUDA/ROCm accelerator detected for GPU-required workflow.",
            action=(
                "In Colab, set Runtime -> Change runtime type -> Hardware accelerator: GPU, "
                "then restart runtime and re-run bootstrap."
            ),
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Colab preflight checks before bootstrap/smoke/full/resume workflows."
    )
    parser.add_argument(
        "--stage",
        choices=sorted(VALID_STAGES),
        required=True,
        help="Workflow stage to validate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Expected output directory for smoke/full stages.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory for resume stage.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if no CUDA/ROCm accelerator is detected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report instead of plain text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when failures are present.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_preflight(
        stage=args.stage,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        require_gpu=args.require_gpu,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report))

    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
