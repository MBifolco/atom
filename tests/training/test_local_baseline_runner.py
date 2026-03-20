"""Training-tier smoke tests for the local baseline runner."""

import sys
from pathlib import Path

from src.training.utils.baseline_harness import BaselineRunConfig, run_baseline


def test_run_baseline_writes_log_and_metadata(monkeypatch, tmp_path: Path):
    config = BaselineRunConfig(
        output_dir=tmp_path,
        mode="curriculum",
        timesteps=1,
        seed=5,
        device="cpu",
        max_ticks=10,
    )

    # Keep this test fast by overriding the command to a tiny Python process.
    monkeypatch.setattr(
        config,
        "build_command",
        lambda python_executable=sys.executable: [
            python_executable,
            "-c",
            "print('baseline smoke ok')",
        ],
    )

    result = run_baseline(config=config, stream_output=False, check=True)
    assert result.succeeded
    assert result.log_path.exists()
    assert result.metadata_path.exists()
    assert "baseline smoke ok" in result.log_path.read_text(encoding="utf-8")
