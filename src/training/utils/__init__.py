"""Training utilities for Atom Combat."""

__all__ = ["NaNDetector", "configure_runtime_gpu_env", "detect_runtime_platform"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
