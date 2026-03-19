"""
Runtime GPU platform detection/configuration utilities.

This module is intentionally stdlib-only so it can run before importing JAX/Torch.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _normalize_platform(platform: str | None) -> str | None:
    """Normalize JAX platform names to cuda/rocm/cpu."""
    if not platform:
        return None

    # JAX may receive comma-separated fallbacks, e.g. "cuda,cpu".
    primary = platform.split(",")[0].strip().lower()
    if primary in {"gpu", "cuda"}:
        return "cuda"
    if primary in {"rocm", "hip"}:
        return "rocm"
    if primary == "cpu":
        return "cpu"
    return primary


def detect_runtime_platform() -> str:
    """
    Detect the preferred accelerator platform without importing GPU frameworks.

    Priority:
    1. Explicit user override via JAX_PLATFORMS
    2. NVIDIA/CUDA
    3. AMD/ROCm
    4. CPU fallback
    """
    explicit = _normalize_platform(os.environ.get("JAX_PLATFORMS"))
    if explicit:
        return explicit

    if _command_reports_gpu(["nvidia-smi", "-L"]):
        return "cuda"

    if _command_reports_gpu(["rocm-smi", "--showproductname"]):
        return "rocm"

    return "cpu"


def _command_reports_gpu(command: list[str]) -> bool:
    """Return True when a vendor tool is present and reports at least one device."""
    if not shutil.which(command[0]):
        return False

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False

    output = f"{result.stdout}\n{result.stderr}".strip().lower()
    if not output:
        return False

    # Common "installed but no usable device" messages.
    unusable_markers = (
        "no devices found",
        "no devices were found",
        "no gpu",
        "not found",
    )
    return not any(marker in output for marker in unusable_markers)


def configure_runtime_gpu_env(
    enable_gpu: bool = True,
    memory_fraction: float = 0.75,
    visible_device: str = "0",
) -> str:
    """
    Configure environment variables for CUDA/ROCm/CPU runtimes.

    Returns:
        Detected/selected platform ("cuda", "rocm", or "cpu")
    """
    platform = detect_runtime_platform()

    # Keep these generic defaults unless user already set them.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(memory_fraction))

    if not enable_gpu or platform == "cpu":
        # Let JAX choose CPU naturally unless caller explicitly requested something else.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        return "cpu"

    if platform == "cuda":
        # CUDA is explicit to avoid accidental CPU fallback on Colab.
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible_device)
        return "cuda"

    if platform == "rocm":
        # Do not force JAX_PLATFORMS=rocm here. On some hosts ROCm tools exist
        # but JAX ROCm plugin may not initialize; letting JAX auto-select allows
        # graceful CPU fallback.
        # Helpful defaults for many RDNA2 desktop setups; no-op if user overrides.
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", visible_device)
        os.environ.setdefault("ROCR_VISIBLE_DEVICES", visible_device)
        os.environ.setdefault("GPU_DEVICE_ORDINAL", visible_device)

        # ROCm/JAX stability tweak used by this project.
        xla_flags = os.environ.get("XLA_FLAGS", "").strip()
        triton_flag = "--xla_gpu_enable_triton_gemm=false"
        if triton_flag not in xla_flags.split():
            os.environ["XLA_FLAGS"] = f"{xla_flags} {triton_flag}".strip()

        return "rocm"

    # Unknown accelerator label: keep it, but avoid forcing vendor-specific vars.
    return platform
