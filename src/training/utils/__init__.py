"""Training utilities for Atom Combat."""

__all__ = [
    "NaNDetector",
    "configure_runtime_gpu_env",
    "detect_runtime_platform",
    "set_global_seeds",
    "build_seeded_env",
    "BaselineRunConfig",
    "BaselineRunResult",
    "run_baseline",
    "run_preflight",
    "render_preflight_report",
]


def __getattr__(name):
    if name == "NaNDetector":
        from src.atom.training.utils.nan_detector import NaNDetector
        return NaNDetector
    if name in {"configure_runtime_gpu_env", "detect_runtime_platform"}:
        from src.atom.training.utils.runtime_platform import configure_runtime_gpu_env, detect_runtime_platform
        return {
            "configure_runtime_gpu_env": configure_runtime_gpu_env,
            "detect_runtime_platform": detect_runtime_platform,
        }[name]
    if name in {"set_global_seeds", "build_seeded_env"}:
        from src.atom.training.utils.determinism import set_global_seeds, build_seeded_env
        return {
            "set_global_seeds": set_global_seeds,
            "build_seeded_env": build_seeded_env,
        }[name]
    if name in {"BaselineRunConfig", "BaselineRunResult", "run_baseline"}:
        from src.atom.training.utils.baseline_harness import BaselineRunConfig, BaselineRunResult, run_baseline
        return {
            "BaselineRunConfig": BaselineRunConfig,
            "BaselineRunResult": BaselineRunResult,
            "run_baseline": run_baseline,
        }[name]
    if name in {"run_preflight", "render_preflight_report"}:
        from src.atom.training.utils.colab_preflight import render_report, run_preflight
        return {
            "run_preflight": run_preflight,
            "render_preflight_report": render_report,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
