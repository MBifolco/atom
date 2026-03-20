#!/usr/bin/env python3
"""
Render Replays to HTML (Manual Montage Creation)

Simple script that renders replay files to HTML for manual video recording.
Use this if you don't want to install Playwright.

Usage:
    python scripts/montage/render_replays.py --run-dir outputs/progressive_20251114_120000

Then:
    1. Open HTML files in browser
    2. Record with screen capture software (OBS, QuickTime, etc.)
    3. Edit together in video editor
"""

import json
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.atom.runtime.telemetry.replay_store import load_replay
from src.atom.runtime.renderer.html_renderer import HtmlRenderer
from src.atom.runtime.evaluator.spectacle_evaluator import SpectacleScore


def render_all_replays(
    run_dir: Path,
    output_dir: Path = None,
    playback_speed: float = 3.0
):
    """
    Render all replays to HTML files.

    Args:
        run_dir: Training run directory
        output_dir: Where to save HTML files (default: run_dir/rendered_html)
        playback_speed: Playback speed multiplier
    """
    print("\n" + "="*80)
    print("RENDERING REPLAYS TO HTML")
    print("="*80)

    # Paths
    if output_dir is None:
        output_dir = run_dir / "rendered_html"

    output_dir.mkdir(exist_ok=True)

    # Load replay indices from curriculum and population subdirectories
    curriculum_index_path = run_dir / "curriculum" / "replay_index.json"
    population_index_path = run_dir / "population" / "replay_index.json"

    all_replays = []

    # Load curriculum replays
    if curriculum_index_path.exists():
        with open(curriculum_index_path) as f:
            curriculum_data = json.load(f)
            # Add source directory to each replay metadata
            for replay in curriculum_data['replays']:
                replay['_source_dir'] = 'curriculum'
            all_replays.extend(curriculum_data['replays'])
            print(f"✅ Found {len(curriculum_data['replays'])} curriculum replays")
    else:
        print(f"ℹ️  No curriculum replays found at {curriculum_index_path}")

    # Load population replays
    if population_index_path.exists():
        with open(population_index_path) as f:
            population_data = json.load(f)
            # Add source directory to each replay metadata
            for replay in population_data['replays']:
                replay['_source_dir'] = 'population'
            all_replays.extend(population_data['replays'])
            print(f"✅ Found {len(population_data['replays'])} population replays")
    else:
        print(f"ℹ️  No population replays found at {population_index_path}")

    if not all_replays:
        print(f"\n❌ ERROR: No replay data found!")
        print(f"\nMake sure you ran training with --record-replays flag")
        print(f"Checked locations:")
        print(f"  - {curriculum_index_path}")
        print(f"  - {population_index_path}")
        sys.exit(1)

    # Create combined index
    index_data = {'total_replays': len(all_replays), 'replays': all_replays}

    replays = index_data['replays']
    print(f"\nFound {len(replays)} replays in index")

    # Separate by type
    curriculum_replays = [r for r in replays if r['stage_type'] == 'curriculum']
    population_replays = [r for r in replays if r['stage_type'] == 'population']

    print(f"  - Curriculum: {len(curriculum_replays)}")
    print(f"  - Population: {len(population_replays)}")
    print(f"\nPlayback speed: {playback_speed}x")
    print(f"Output directory: {output_dir}\n")

    # Sort
    curriculum_replays.sort(key=lambda r: (r['stage'], r['spectacle_rank']))
    population_replays.sort(key=lambda r: (r['stage'], r['spectacle_rank']))

    html_files = []

    # Create a playlist HTML
    playlist_items = []

    # Render curriculum replays
    if curriculum_replays:
        print("="*80)
        print("PART 1: CURRICULUM TRAINING")
        print("="*80)

        playlist_items.append({
            'section': 'Part 1: Curriculum Training (From Zero to Graduate)',
            'replays': []
        })

        for i, replay_meta in enumerate(curriculum_replays, 1):
            stage = replay_meta['stage']
            rank = replay_meta['spectacle_rank']

            print(f"\n[{i}/{len(curriculum_replays)}] {stage} ({rank} spectacle: {replay_meta['spectacle_score']:.3f})")
            print(f"  {replay_meta['fighter_a']} vs {replay_meta['fighter_b']} → {replay_meta['winner']}")

            # Find replay file in correct subdirectory
            source_dir = replay_meta.get('_source_dir', 'curriculum')
            replays_dir = run_dir / source_dir / "replays"
            replay_files = list(replays_dir.glob(f"{stage}_{rank}_*.json.gz"))
            if not replay_files:
                print(f"  ⚠️  Warning: Replay file not found in {replays_dir}, skipping")
                continue

            replay_path = replay_files[0]

            # Load replay data
            replay_data = load_replay(str(replay_path))

            # Extract components
            telemetry = replay_data['telemetry']
            result_data = replay_data['result']
            events = replay_data['events']
            metadata = replay_data.get('metadata', {})

            # Create mock MatchResult
            class MatchResult:
                def __init__(self, data, events):
                    self.winner = data['winner']
                    self.total_ticks = data['total_ticks']
                    self.final_hp_a = data['final_hp_a']
                    self.final_hp_b = data['final_hp_b']
                    self.events = events

            match_result = MatchResult(result_data, events)

            # Create SpectacleScore if available
            spectacle_score = None
            if 'spectacle_breakdown' in metadata:
                breakdown = metadata['spectacle_breakdown']
                spectacle_score = SpectacleScore(
                    duration=breakdown['duration'],
                    close_finish=breakdown['close_finish'],
                    stamina_drama=breakdown['stamina_drama'],
                    comeback_potential=breakdown['comeback_potential'],
                    positional_exchange=breakdown['positional_exchange'],
                    pacing_variety=breakdown['pacing_variety'],
                    collision_drama=breakdown['collision_drama'],
                    overall=breakdown['overall']
                )

            # Render to HTML
            html_filename = f"curriculum_{i:02d}_{stage}_{rank}.html"
            html_path = output_dir / html_filename

            renderer = HtmlRenderer()
            renderer.generate_replay_html(
                telemetry=telemetry,
                match_result=match_result,
                output_path=str(html_path),
                spectacle_score=spectacle_score,
                playback_speed=playback_speed
            )

            duration_sec = result_data['total_ticks'] / 12.0 / playback_speed
            print(f"  ✅ Rendered to {html_filename} (~{duration_sec:.1f}s)")

            html_files.append(html_path)
            playlist_items[-1]['replays'].append({
                'filename': html_filename,
                'title': f"{stage.replace('curriculum_level_', 'Level ').replace('_', ' ').title()} - {rank.capitalize()} Spectacle",
                'duration': duration_sec,
                'spectacle': replay_meta['spectacle_score']
            })

    # Render population replays
    if population_replays:
        print("\n" + "="*80)
        print("PART 2: POPULATION EVOLUTION")
        print("="*80)

        playlist_items.append({
            'section': 'Part 2: Population Evolution (Natural Selection)',
            'replays': []
        })

        for i, replay_meta in enumerate(population_replays, 1):
            stage = replay_meta['stage']
            rank = replay_meta['spectacle_rank']

            print(f"\n[{i}/{len(population_replays)}] {stage} ({rank} spectacle: {replay_meta['spectacle_score']:.3f})")
            print(f"  {replay_meta['fighter_a']} vs {replay_meta['fighter_b']} → {replay_meta['winner']}")

            # Find replay file in correct subdirectory
            source_dir = replay_meta.get('_source_dir', 'population')
            replays_dir = run_dir / source_dir / "replays"
            replay_files = list(replays_dir.glob(f"{stage}_{rank}_*.json.gz"))
            if not replay_files:
                print(f"  ⚠️  Warning: Replay file not found in {replays_dir}, skipping")
                continue

            replay_path = replay_files[0]

            # Load and render (same as curriculum)
            replay_data = load_replay(str(replay_path))
            telemetry = replay_data['telemetry']
            result_data = replay_data['result']
            events = replay_data['events']
            metadata = replay_data.get('metadata', {})

            class MatchResult:
                def __init__(self, data, events):
                    self.winner = data['winner']
                    self.total_ticks = data['total_ticks']
                    self.final_hp_a = data['final_hp_a']
                    self.final_hp_b = data['final_hp_b']
                    self.events = events

            match_result = MatchResult(result_data, events)

            spectacle_score = None
            if 'spectacle_breakdown' in metadata:
                breakdown = metadata['spectacle_breakdown']
                spectacle_score = SpectacleScore(
                    duration=breakdown['duration'],
                    close_finish=breakdown['close_finish'],
                    stamina_drama=breakdown['stamina_drama'],
                    comeback_potential=breakdown['comeback_potential'],
                    positional_exchange=breakdown['positional_exchange'],
                    pacing_variety=breakdown['pacing_variety'],
                    collision_drama=breakdown['collision_drama'],
                    overall=breakdown['overall']
                )

            html_filename = f"population_{i:02d}_{stage}_{rank}.html"
            html_path = output_dir / html_filename

            renderer = HtmlRenderer()
            renderer.generate_replay_html(
                telemetry=telemetry,
                match_result=match_result,
                output_path=str(html_path),
                spectacle_score=spectacle_score,
                playback_speed=playback_speed
            )

            duration_sec = result_data['total_ticks'] / 12.0 / playback_speed
            print(f"  ✅ Rendered to {html_filename} (~{duration_sec:.1f}s)")

            html_files.append(html_path)
            playlist_items[-1]['replays'].append({
                'filename': html_filename,
                'title': f"{stage.replace('population_gen_', 'Generation ')} - {rank.capitalize()} Spectacle",
                'duration': duration_sec,
                'spectacle': replay_meta['spectacle_score']
            })

    # Create playlist/index HTML
    create_playlist_html(output_dir, playlist_items, playback_speed)

    # Summary
    total_duration = sum(
        item['duration']
        for section in playlist_items
        for item in section['replays']
    )

    print("\n" + "="*80)
    print("✅ RENDERING COMPLETE")
    print("="*80)
    print(f"Rendered: {len(html_files)} HTML files")
    print(f"Total duration: ~{total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Output: {output_dir}/")
    print("\nNext steps:")
    print(f"  1. Open {output_dir}/playlist.html in browser")
    print("  2. Click through replays and enjoy!")
    print("\nFor video montage:")
    print("  - Use screen recording software (OBS, QuickTime, etc.)")
    print("  - Record each HTML replay")
    print("  - Edit together in video editor")
    print("\nOr use automated method:")
    print(f"  python scripts/montage/create_montage.py --run-dir {run_dir}")


def create_playlist_html(output_dir: Path, playlist_items: list, playback_speed: float):
    """Create a playlist/index HTML for easy navigation."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Training Montage - Replay Playlist</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #eee;
            padding: 40px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #ff6b6b;
            border-bottom: 3px solid #ff6b6b;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4ecdc4;
            margin-top: 40px;
        }}
        .replay-list {{
            list-style: none;
            padding: 0;
        }}
        .replay-item {{
            background: #16213e;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4ecdc4;
            border-radius: 4px;
        }}
        .replay-item:hover {{
            background: #1f2b4d;
            border-left-color: #ff6b6b;
        }}
        .replay-item a {{
            color: #fff;
            text-decoration: none;
            font-size: 18px;
            display: block;
        }}
        .replay-meta {{
            color: #95a5a6;
            font-size: 14px;
            margin-top: 5px;
        }}
        .spectacle-score {{
            display: inline-block;
            background: #4ecdc4;
            color: #000;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .info-box {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🥊 Atom Combat - Training Montage Playlist</h1>

    <div class="info-box">
        <p><strong>Playback Speed:</strong> {playback_speed}x</p>
        <p><strong>Instructions:</strong> Click any replay below to view it in your browser.</p>
        <p>For recording: Use screen capture software to record each replay, then edit together.</p>
    </div>
"""

    for section in playlist_items:
        html_content += f"\n    <h2>{section['section']}</h2>\n"
        html_content += '    <ul class="replay-list">\n'

        for replay in section['replays']:
            html_content += f"""        <li class="replay-item">
            <a href="{replay['filename']}" target="_blank">
                {replay['title']}
                <span class="spectacle-score">{replay['spectacle']:.3f}</span>
            </a>
            <div class="replay-meta">
                Duration: ~{replay['duration']:.1f}s | File: {replay['filename']}
            </div>
        </li>
"""

        html_content += '    </ul>\n'

    html_content += """
</body>
</html>
"""

    playlist_path = output_dir / "playlist.html"
    with open(playlist_path, 'w') as f:
        f.write(html_content)

    print(f"\n📋 Created playlist: {playlist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render replays to HTML files for manual montage creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render all replays
  python scripts/montage/render_replays.py --run-dir outputs/progressive_20251114_120000

  # Custom playback speed
  python scripts/montage/render_replays.py --run-dir outputs/progressive_20251114_120000 --speed 5.0

  # Custom output directory
  python scripts/montage/render_replays.py --run-dir outputs/progressive_20251114_120000 --output-dir ~/Desktop/replays
        """
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing replays/"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HTML files (default: RUN_DIR/rendered_html)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="Playback speed multiplier (default: 3.0)"
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    render_all_replays(
        run_dir=run_dir,
        output_dir=output_dir,
        playback_speed=args.speed
    )


if __name__ == "__main__":
    main()
