"""Structured observability helpers for training runs."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

from .runtime_platform import detect_runtime_platform

_TRAINING_PACKAGES = (
    "stable-baselines3",
    "sbx",
    "torch",
    "jax",
    "jaxlib",
    "numpy",
    "gymnasium",
    "gym",
    "chex",
)


def ensure_analysis_dir(base_dir: str | Path) -> Path:
    """Ensure and return the structured-analysis directory."""
    analysis_dir = Path(base_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def append_jsonl(path: str | Path, payload: Any) -> Path:
    """Append a JSON payload as a single line to a JSONL file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_json(payload)
    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, sort_keys=True))
        handle.write("\n")
    return file_path


def write_json(path: str | Path, payload: Any) -> Path:
    """Write normalized JSON payload to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_json(payload)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return file_path


def build_run_manifest(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    training_config: Mapping[str, Any],
    seed_report: Any | None = None,
) -> dict[str, Any]:
    """Build a structured run manifest for curriculum/population runs."""
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    return {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "git": _collect_git_metadata(repo_root),
        "runtime": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "accelerator_platform": detect_runtime_platform(),
            "gpu": _collect_gpu_metadata(),
            "package_versions": _collect_package_versions(),
        },
        "training": _normalize_for_json(training_config),
        "seed_report": _normalize_for_json(seed_report),
    }


def _collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "short_commit": _run_command(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root),
        "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "dirty": _git_dirty(repo_root),
    }


def _git_dirty(repo_root: Path) -> bool | None:
    status = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    if status is None:
        return None
    return bool(status.strip())


def _collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _TRAINING_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _collect_gpu_metadata() -> dict[str, Any]:
    gpu_info = {
        "platform": detect_runtime_platform(),
        "name": None,
        "total_vram_mb": None,
        "count": 0,
        "source": None,
    }

    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            gpu_info.update(
                {
                    "name": torch.cuda.get_device_name(device_index),
                    "total_vram_mb": int(properties.total_memory / (1024 * 1024)),
                    "count": torch.cuda.device_count(),
                    "source": "torch",
                }
            )
            return gpu_info
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        output = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            rows = [line.strip() for line in output.splitlines() if line.strip()]
            if rows:
                first = rows[0]
                name, _, memory = first.partition(",")
                try:
                    total_vram_mb = int(memory.strip()) if memory.strip() else None
                except ValueError:
                    total_vram_mb = None
                gpu_info.update(
                    {
                        "name": name.strip(),
                        "total_vram_mb": total_vram_mb,
                        "count": len(rows),
                        "source": "nvidia-smi",
                    }
                )
    return gpu_info


def _run_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    return value or None


def _normalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_json(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_for_json(item) for item in value)
    return value
