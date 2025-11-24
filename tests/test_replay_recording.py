#!/usr/bin/env python3
"""
Test script to verify replay recording functionality works.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.replay_recorder import ReplayRecorder
from src.arena import WorldConfig


def test_replay_recorder():
    """Test basic replay recorder functionality."""
    print("Testing Replay Recorder...")
    print("-" * 60)

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Output directory: {tmpdir}")

        # Create replay recorder
        config = WorldConfig()
        recorder = ReplayRecorder(
            output_dir=tmpdir,
            max_ticks=250,
            samples_per_stage=3,
            min_matches_for_sampling=3,
            verbose=True
        )

        print(f"Replay recorder created successfully")
        print(f"Replays directory: {recorder.replays_dir}")
        print(f"Config arena width: {recorder.config.arena_width}")

        # Test with test dummy opponents
        test_opponents = [
            "fighters/test_dummies/atomic/stationary_neutral.py",
            "fighters/test_dummies/atomic/stationary_extended.py",
            "fighters/test_dummies/atomic/stationary_defending.py",
        ]

        # Verify opponent files exist
        print("\nChecking opponent files:")
        for opp_path in test_opponents:
            exists = Path(opp_path).exists()
            print(f"  {opp_path}: {'EXISTS' if exists else 'NOT FOUND'}")

        # Try to record a curriculum stage (without a model)
        print("\nAttempting to record curriculum stage...")
        try:
            recorder.record_curriculum_stage(
                stage_name="Test Stage",
                level_num=1,
                model=None,  # No model for testing
                opponent_paths=test_opponents,
                num_matches_per_opponent=1
            )
            print("Record curriculum stage completed")
        except Exception as e:
            print(f"Error recording curriculum stage: {e}")
            import traceback
            traceback.print_exc()

        # Check if any replays were saved
        print("\nChecking for saved replays:")
        replays_dir = Path(tmpdir) / "replays"
        if replays_dir.exists():
            replay_files = list(replays_dir.glob("*.json.gz"))
            print(f"  Found {len(replay_files)} replay files")
            for replay_file in replay_files:
                print(f"    - {replay_file.name}")
        else:
            print("  Replays directory doesn't exist")

        # Check if replay index was created
        index_path = Path(tmpdir) / "replay_index.json"
        if index_path.exists():
            print(f"\nReplay index exists: {index_path}")
            import json
            with open(index_path) as f:
                index_data = json.load(f)
                print(f"  Total replays in index: {index_data.get('total_replays', 0)}")
        else:
            print("\nNo replay index found")

        # Save the index
        print("\nSaving replay index...")
        recorder.save_replay_index()
        if index_path.exists():
            print("  Index saved successfully")
        else:
            print("  Failed to save index")

    print("\n" + "=" * 60)
    print("Test completed")


if __name__ == "__main__":
    test_replay_recorder()