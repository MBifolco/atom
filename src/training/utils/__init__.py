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
]


def __getattr__(name):
    if name == "NaNDetector":
        from .nan_detector import NaNDetector
        return NaNDetector
    if name in {"configure_runtime_gpu_env", "detect_runtime_platform"}:
        from .runtime_platform import configure_runtime_gpu_env, detect_runtime_platform
        return {
            "configure_runtime_gpu_env": configure_runtime_gpu_env,
            "detect_runtime_platform": detect_runtime_platform,
        }[name]
    if name in {"set_global_seeds", "build_seeded_env"}:
        from .determinism import set_global_seeds, build_seeded_env
        return {
            "set_global_seeds": set_global_seeds,
            "build_seeded_env": build_seeded_env,
        }[name]
    if name in {"BaselineRunConfig", "BaselineRunResult", "run_baseline"}:
        from .baseline_harness import BaselineRunConfig, BaselineRunResult, run_baseline
        return {
            "BaselineRunConfig": BaselineRunConfig,
            "BaselineRunResult": BaselineRunResult,
            "run_baseline": run_baseline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
