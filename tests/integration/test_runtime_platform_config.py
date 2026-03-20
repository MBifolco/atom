"""Integration tests for runtime platform environment configuration."""

import os

from src.training.utils.runtime_platform import configure_runtime_gpu_env


def test_configure_runtime_gpu_env_cpu_mode_sets_jax_platform(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    platform = configure_runtime_gpu_env(enable_gpu=False)
    assert platform == "cpu"
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
