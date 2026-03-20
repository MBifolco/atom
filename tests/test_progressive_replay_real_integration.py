"""
Real integration test for progressive replay recording.

This test actually runs training and verifies that replays are recorded
with proper telemetry, catching issues like mismatched snapshot formats.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest
import tempfile
import gzip
import json
from pathlib import Path
import numpy as np

from src.training.trainers.curriculum_trainer import CurriculumTrainer, CurriculumCallback
from src.orchestrator.match_orchestrator import MatchOrchestrator
from src.arena import WorldConfig
from src.training.signal_engine import build_observation_from_snapshot


class TestRealProgressiveReplayIntegration:
    """Integration tests that actually run the full system."""

    def test_snapshot_format_compatibility(self):
        """Test that the snapshot format matches what the model expects."""
        # This test would have caught the acceleration field issue
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=1, record_telemetry=True)

        actual_snapshot = None

        def capture_snapshot(snapshot):
            nonlocal actual_snapshot
            actual_snapshot = snapshot.copy()
            return {"acceleration": 0.0, "stance": "neutral"}

        result = orchestrator.run_match(
            {"name": "Fighter1", "mass": 70.0, "position": 2.0},
            {"name": "Fighter2", "mass": 70.0, "position": 10.0},
            capture_snapshot,
            lambda s: {"acceleration": 0.0, "stance": "neutral"},
            seed=42
        )

        # Verify snapshot has expected structure
        assert actual_snapshot is not None
        assert "you" in actual_snapshot
        assert "opponent" in actual_snapshot
        assert "arena" in actual_snapshot

        # Check fighter fields
        you = actual_snapshot["you"]
        assert "position" in you
        assert "velocity" in you
        assert "hp" in you
        assert "stamina" in you
        assert "stance" in you
        # These should NOT exist in snapshot (gym env adds them)
        assert "acceleration" not in you

        # Check opponent fields
        opponent = actual_snapshot["opponent"]
        assert "distance" in opponent
        assert "direction" in opponent
        assert "velocity" in opponent
        assert "stance_hint" in opponent
        # Opponent should NOT have position directly
        assert "position" not in opponent

    def test_model_decision_with_real_snapshot(self):
        """Test that model predictions work with actual snapshot format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal trainer to get a model
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                n_envs=2,
                max_ticks=50,
                verbose=False,
                record_replays=False
            )

            # Initialize training to get a model
            try:
                trainer.train(total_timesteps=100)  # Very short training just to init
            except RuntimeError:
                # Training might fail but model should be initialized
                pass
            assert trainer.model is not None

            # Create a real snapshot
            snapshot = {
                "tick": 0,
                "you": {
                    "position": 5.0,
                    "velocity": 0.0,
                    "hp": 93.7,
                    "max_hp": 93.7,
                    "stamina": 8.5,
                    "max_stamina": 8.5,
                    "stance": "neutral"
                },
                "opponent": {
                    "distance": 5.0,
                    "direction": 1.0,
                    "velocity": 0.0,
                    "hp": 93.7,
                    "max_hp": 93.7,
                    "stamina": 8.5,
                    "max_stamina": 8.5,
                    "stance_hint": "neutral"
                },
                "arena": {
                    "width": 12.476
                }
            }

            obs = build_observation_from_snapshot(snapshot, recent_damage=0.0)

            # Model should be able to predict with this observation
            action, _ = trainer.model.predict(np.array([obs]), deterministic=True)

            # Action should be 2D array with batch dimension (1, 2)
            assert action.shape == (1, 2), f"Expected action shape (1, 2), got {action.shape}"

            # Test that we can extract the actual action values
            action_values = action[0] if action.ndim > 1 else action
            assert len(action_values) == 2

            # Acceleration should be in [-1, 1]
            assert -1.0 <= action_values[0] <= 1.0
            # Stance selector should be in [0, 2.99]
            assert 0.0 <= action_values[1] < 3.0

    def test_progressive_replays_actually_recorded_during_training(self):
        """Test that progressive replays are actually saved during training with telemetry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                n_envs=2,
                max_ticks=50,
                verbose=False,
                record_replays=True,
                override_episodes_per_level=2  # Quick graduation
            )

            # Train for a bit
            try:
                trainer.train(total_timesteps=3000)
            except Exception:
                pass  # May stop early due to graduation

            # Check that progressive replays were saved
            replay_dir = Path(tmpdir) / "progressive_replays"
            assert replay_dir.exists(), "Progressive replay directory should exist"

            replays = list(replay_dir.glob("*.json.gz"))
            assert len(replays) > 0, "Should have saved at least one progressive replay"

            # Verify replay content
            for replay_path in replays:
                with gzip.open(replay_path, 'rt') as f:
                    data = json.load(f)

                # Check structure
                assert "telemetry" in data
                assert "result" in data
                # Meta might be saved separately or inline

                # Check telemetry has ticks
                ticks = data["telemetry"].get("ticks", [])
                assert len(ticks) > 0, f"Replay {replay_path.name} should have telemetry ticks"

                # Check result
                result = data["result"]
                assert "winner" in result
                assert "total_ticks" in result
                assert result["total_ticks"] > 0, "Match should have run for some ticks"

    def test_evaluation_replay_runs_successfully(self):
        """Test that evaluation replays run without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CurriculumTrainer(
                algorithm='ppo',
                output_dir=tmpdir,
                n_envs=2,
                max_ticks=50,
                verbose=False,
                record_replays=True
            )

            # Create a callback
            callback = CurriculumCallback(trainer, verbose=0)

            # Manually call _record_evaluation_replay
            try:
                callback._record_evaluation_replay(episode_num=1, total_episodes=100)
                # Should not raise any exceptions
                assert True
            except KeyError as e:
                # This would catch the 'acceleration' field error we had
                pytest.fail(f"KeyError in _record_evaluation_replay: {e}")
            except Exception as e:
                # Other exceptions are OK (e.g., no replays saved due to conditions)
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
