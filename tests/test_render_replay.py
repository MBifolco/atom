#!/usr/bin/env python3
"""
Test rendering a replay file to HTML to verify the full pipeline works.
"""

import sys
import tempfile
from pathlib import Path
import gzip
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.atom.training.replay_recorder import ReplayRecorder
from src.arena import WorldConfig
from scripts.montage.create_montage import render_replay_to_html


def test_render_replay():
    """Test rendering a replay to HTML."""
    print("Testing Replay Rendering to HTML...")
    print("=" * 60)

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Output directory: {tmpdir}")

        # Step 1: Record a replay
        config = WorldConfig()
        recorder = ReplayRecorder(
            output_dir=tmpdir,
            max_ticks=100,  # Shorter for testing
            samples_per_stage=3,
            min_matches_for_sampling=3,
            verbose=False
        )

        # Record one curriculum stage
        print("\n1. Recording test matches...")
        recorder.record_curriculum_stage(
            stage_name="Test",
            level_num=1,
            model=None,  # Use None for testing
            opponent_paths=[
                "fighters/test_dummies/atomic/forward_charger.py",
                "fighters/test_dummies/atomic/retreater.py",
                "fighters/test_dummies/atomic/oscillator.py"
            ],
            num_matches_per_opponent=1
        )

        # Save the replay index
        recorder.save_replay_index()

        # Step 2: Find a replay file
        replays_dir = Path(tmpdir) / "replays"
        replay_files = list(replays_dir.glob("*.json.gz"))

        if not replay_files:
            print("ERROR: No replay files created")
            return

        print(f"\n2. Found {len(replay_files)} replay files")
        test_replay = replay_files[0]
        print(f"   Testing with: {test_replay.name}")

        # Step 3: Try to render to HTML
        print("\n3. Rendering replay to HTML...")
        html_path = Path(tmpdir) / "test_replay.html"

        try:
            # Load and inspect the replay first
            with gzip.open(test_replay, 'rt') as f:
                replay_data = json.load(f)

            print(f"   - Total ticks: {replay_data['result']['total_ticks']}")
            print(f"   - Winner: {replay_data['result']['winner']}")
            print(f"   - Event types: {set(e['type'] for e in replay_data.get('events', []))}")

            # Render to HTML
            output_html = render_replay_to_html(
                replay_path=test_replay,
                output_html=html_path,
                playback_speed=2.0
            )

            if output_html and Path(output_html).exists():
                file_size = Path(output_html).stat().st_size
                print(f"\n✅ HTML rendering successful!")
                print(f"   - Output: {output_html}")
                print(f"   - Size: {file_size:,} bytes")

                # Read first few lines to verify it's valid HTML
                with open(output_html, 'r') as f:
                    lines = f.readlines()[:5]
                    if any('<!DOCTYPE html>' in line for line in lines):
                        print("   - Valid HTML structure detected")

            else:
                print("\n❌ HTML rendering failed")

        except Exception as e:
            print(f"\n❌ Error rendering replay: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Render test completed")


if __name__ == "__main__":
    test_render_replay()