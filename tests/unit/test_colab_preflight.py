"""Unit tests for Colab preflight validation helpers."""

from pathlib import Path

from src.training.utils import colab_preflight


def _base_env(tmp_path: Path) -> dict[str, str]:
    drive_root = tmp_path / "drive"
    drive_repo = drive_root / "dev" / "atom"
    work_repo = tmp_path / "work" / "atom"
    drive_repo.parent.mkdir(parents=True, exist_ok=True)
    work_repo.parent.mkdir(parents=True, exist_ok=True)
    return {
        "ATOM_REPO_URL": "https://github.com/example/atom.git",
        "ATOM_BRANCH": "colab",
        "ATOM_DRIVE_REPO": str(drive_repo),
        "ATOM_WORK_REPO": str(work_repo),
        "ATOM_DRIVE_REPO_SYNC_MODE": "stash",
    }


def test_bootstrap_requires_repo_url_when_drive_cache_missing(tmp_path: Path):
    env = _base_env(tmp_path)
    env["ATOM_REPO_URL"] = ""

    report = colab_preflight.run_preflight(stage="bootstrap", env=env)

    assert any(check.name == "repo-url" and check.status == colab_preflight.FAIL for check in report.checks)


def test_bootstrap_repo_url_optional_when_drive_cache_exists(tmp_path: Path):
    env = _base_env(tmp_path)
    env["ATOM_REPO_URL"] = ""
    drive_git_dir = Path(env["ATOM_DRIVE_REPO"]) / ".git"
    drive_git_dir.mkdir(parents=True, exist_ok=True)

    report = colab_preflight.run_preflight(stage="bootstrap", env=env)

    assert not any(check.name == "repo-url" and check.status == colab_preflight.FAIL for check in report.checks)


def test_invalid_sync_mode_fails(tmp_path: Path):
    env = _base_env(tmp_path)
    env["ATOM_DRIVE_REPO_SYNC_MODE"] = "invalid-mode"

    report = colab_preflight.run_preflight(stage="bootstrap", env=env)

    assert any(check.name == "drive-sync-mode" and check.status == colab_preflight.FAIL for check in report.checks)


def test_smoke_stage_requires_training_entrypoint(tmp_path: Path):
    env = _base_env(tmp_path)
    Path(env["ATOM_WORK_REPO"]).mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "outputs" / "quick_test"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    report = colab_preflight.run_preflight(
        stage="smoke",
        env=env,
        output_dir=str(output_dir),
    )

    assert any(
        check.name == "train-entrypoint" and check.status == colab_preflight.FAIL
        for check in report.checks
    )


def test_smoke_stage_passes_with_entrypoint_and_output_dir(tmp_path: Path):
    env = _base_env(tmp_path)
    work_repo = Path(env["ATOM_WORK_REPO"])
    work_repo.mkdir(parents=True, exist_ok=True)
    (work_repo / "train_progressive.py").write_text("print('ok')\n")
    output_dir = tmp_path / "outputs" / "quick_test"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    report = colab_preflight.run_preflight(
        stage="smoke",
        env=env,
        output_dir=str(output_dir),
    )

    assert not any(check.status == colab_preflight.FAIL for check in report.checks)


def test_resume_stage_requires_checkpoint_markers(tmp_path: Path):
    env = _base_env(tmp_path)
    work_repo = Path(env["ATOM_WORK_REPO"])
    work_repo.mkdir(parents=True, exist_ok=True)
    (work_repo / "train_progressive.py").write_text("print('ok')\n")
    checkpoint_dir = tmp_path / "atom_runs" / "run1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    report = colab_preflight.run_preflight(
        stage="resume",
        env=env,
        checkpoint_dir=str(checkpoint_dir),
    )

    assert any(
        check.name == "resume-checkpoint-dir" and check.status == colab_preflight.WARN
        for check in report.checks
    )


def test_gpu_requirement_fails_when_platform_is_cpu(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    monkeypatch.setattr(colab_preflight, "detect_runtime_platform", lambda: "cpu")

    report = colab_preflight.run_preflight(
        stage="bootstrap",
        env=env,
        require_gpu=True,
    )

    assert any(check.name == "gpu-runtime" and check.status == colab_preflight.FAIL for check in report.checks)
