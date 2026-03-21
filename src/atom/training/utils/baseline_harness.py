"""
Local reproducibility harness for progressive training baselines.

This lets us run deterministic local checks without requiring Colab for every
iteration. It is intentionally lightweight and script-friendly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .determinism import build_seeded_env, set_global_seeds


@dataclass
class BaselineRunConfig:
    """Configuration for a local baseline run."""

    output_dir: str | Path
    mode: str = "curriculum"
    timesteps: int = 10_000
    seed: int = 1337
    device: str = "cpu"
    use_vmap: bool = False
    cores: int | None = 1
    max_ticks: int = 250
    override_episodes_per_level: int | None = None
    resume_curriculum: bool = False
    checkpoint_interval: int = 100000

    def validate(self) -> None:
        if self.mode not in {"quick", "curriculum", "population", "complete"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.timesteps <= 0:
            raise ValueError(f"timesteps must be > 0, got {self.timesteps}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.max_ticks <= 0:
            raise ValueError(f"max_ticks must be > 0, got {self.max_ticks}")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"unsupported device: {self.device}")
        if self.cores is not None and self.cores <= 0:
            raise ValueError(f"cores must be > 0 when provided, got {self.cores}")
        if self.override_episodes_per_level is not None and self.override_episodes_per_level <= 0:
            raise ValueError(
                "override_episodes_per_level must be > 0 when provided"
            )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be > 0, got {self.checkpoint_interval}"
            )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def build_command(self, python_executable: str = sys.executable) -> list[str]:
        """
        Build a `train_progressive.py` command for this baseline configuration.
        """
        self.validate()

        cmd = [
            python_executable,
            "train_progressive.py",
            "--mode",
            self.mode,
            "--timesteps",
            str(self.timesteps),
            "--device",
            self.device,
            "--max-ticks",
            str(self.max_ticks),
            "--output-dir",
            str(self.output_path),
        ]
        if self.cores is not None:
            cmd.extend(["--cores", str(self.cores)])
        if self.use_vmap:
            cmd.append("--use-vmap")
        if self.override_episodes_per_level is not None:
            cmd.extend(
                [
                    "--override-episodes-per-level",
                    str(self.override_episodes_per_level),
                ]
            )
        cmd.extend(["--checkpoint-interval", str(self.checkpoint_interval)])
        if self.resume_curriculum:
            cmd.append("--resume-curriculum")
        return cmd

    def build_environment(self) -> dict[str, str]:
        env = build_seeded_env(self.seed)
        if self.device == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
        elif self.device == "cuda":
            env["JAX_PLATFORMS"] = "cuda"
        return env


@dataclass(frozen=True)
class BaselineRunResult:
    """Execution result for a local baseline run."""

    command: list[str]
    returncode: int
    duration_seconds: float
    log_path: Path
    metadata_path: Path
    output_dir: Path

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def run_baseline(
    config: BaselineRunConfig,
    stream_output: bool = True,
    check: bool = False,
) -> BaselineRunResult:
    """
    Execute a local seeded baseline run and persist metadata/log artifacts.

    Args:
        config: Baseline run config.
        stream_output: Stream subprocess logs to stdout while also saving to file.
        check: Raise CalledProcessError on non-zero return code.

    Returns:
        BaselineRunResult with command and artifact paths.
    """
    config.validate()
    set_global_seeds(config.seed)

    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    command = config.build_command()
    env = config.build_environment()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"baseline_{timestamp}.log"
    metadata_path = output_dir / f"baseline_{timestamp}.json"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=Path(__file__).resolve().parents[3],
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if stream_output:
                print(line, end="")

        process.wait()
        returncode = process.returncode

    duration_seconds = time.time() - start

    config_payload = asdict(config)
    config_payload["output_dir"] = str(config.output_path)

    metadata = {
        "config": config_payload,
        "command": command,
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)

    return BaselineRunResult(
        command=command,
        returncode=returncode,
        duration_seconds=duration_seconds,
        log_path=log_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
    )
