"""Tests for local baseline harness command/environment construction."""

from pathlib import Path

from src.training.utils.baseline_harness import BaselineRunConfig


def test_baseline_command_includes_requested_overrides(tmp_path: Path):
    config = BaselineRunConfig(
        output_dir=tmp_path,
        mode="curriculum",
        timesteps=12345,
        seed=999,
        device="cuda",
        cores=2,
        use_vmap=True,
        max_ticks=321,
        override_episodes_per_level=50,
        resume_curriculum=True,
        checkpoint_interval=777,
    )

    cmd = config.build_command(python_executable="python3")

    assert cmd[:2] == ["python3", "train_progressive.py"]
    assert "--timesteps" in cmd and "12345" in cmd
    assert "--device" in cmd and "cuda" in cmd
    assert "--cores" in cmd and "2" in cmd
    assert "--max-ticks" in cmd and "321" in cmd
    assert "--use-vmap" in cmd
    assert "--override-episodes-per-level" in cmd and "50" in cmd
    assert "--checkpoint-interval" in cmd and "777" in cmd
    assert "--resume-curriculum" in cmd
    assert "--output-dir" in cmd and str(tmp_path) in cmd


def test_baseline_environment_is_seeded(tmp_path: Path):
    config = BaselineRunConfig(output_dir=tmp_path, seed=4242, device="cpu")
    env = config.build_environment()
    assert env["PYTHONHASHSEED"] == "4242"
    assert env["ATOM_GLOBAL_SEED"] == "4242"
    assert env["ATOM_TRAINING_SEED"] == "4242"
    assert env["JAX_PLATFORMS"] == "cpu"


def test_baseline_config_validate_rejects_bad_values(tmp_path: Path):
    config = BaselineRunConfig(output_dir=tmp_path, timesteps=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "timesteps" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid timesteps")


def test_baseline_config_validate_rejects_bad_cores(tmp_path: Path):
    config = BaselineRunConfig(output_dir=tmp_path, cores=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "cores" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid cores")


def test_baseline_config_validate_rejects_bad_checkpoint_interval(tmp_path: Path):
    config = BaselineRunConfig(output_dir=tmp_path, checkpoint_interval=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "checkpoint_interval" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid checkpoint_interval")
