#!/usr/bin/env python3
"""
Create Training Montage from Recorded Replays

Renders saved replay files to video and creates a montage showing:
- Part 1: Curriculum training progression (learning fundamentals)
- Part 2: Population evolution (natural selection)

Requirements:
    pip install playwright
    playwright install chromium
    # OR use system Chrome/Chromium

Usage:
    # Create montage from training run
    python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000

    # Custom playback speed (3x faster)
    python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000 --speed 3.0

    # Use system Chrome instead of Playwright
    python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000 --use-chrome
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.replay_store import load_replay
from src.renderer.html_renderer import HtmlRenderer
from src.evaluator.spectacle_evaluator import SpectacleScore


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    if not shutil.which('ffmpeg'):
        print("❌ ERROR: ffmpeg not found!")
        print("\nPlease install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        sys.exit(1)
    else:
        print("✅ ffmpeg found")


def check_playwright():
    """Check if Playwright is available."""
    try:
        from playwright.sync_api import sync_playwright
        print("✅ Playwright available")
        return True
    except ImportError:
        print("⚠️  Playwright not found")
        print("\nTo use Playwright for video recording:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return False


def check_chrome():
    """Check if Chrome/Chromium is available."""
    chrome_paths = [
        '/usr/bin/google-chrome',
        '/usr/bin/chromium-browser',
        '/usr/bin/chromium',
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
    ]

    for path in chrome_paths:
        if Path(path).exists():
            print(f"✅ Chrome/Chromium found: {path}")
            return path

    # Try which command
    chrome_bin = shutil.which('google-chrome') or shutil.which('chromium-browser') or shutil.which('chromium')
    if chrome_bin:
        print(f"✅ Chrome/Chromium found: {chrome_bin}")
        return chrome_bin

    print("⚠️  Chrome/Chromium not found")
    return None


def render_replay_to_html(
    replay_path: Path,
    output_html: Path,
    playback_speed: float = 1.0
) -> Path:
    """
    Render a replay to HTML.

    Args:
        replay_path: Path to replay .json.gz file
        output_html: Path to save HTML file
        playback_speed: Playback speed multiplier

    Returns:
        Path to generated HTML file
    """
    # Load replay data
    replay_data = load_replay(str(replay_path))

    # Extract components
    telemetry = replay_data['telemetry']
    result_data = replay_data['result']
    events = replay_data['events']
    metadata = replay_data.get('metadata', {})

    # Create a mock MatchResult object
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
    renderer = HtmlRenderer()
    html_path = renderer.generate_replay_html(
        telemetry=telemetry,
        match_result=match_result,
        output_path=str(output_html),
        spectacle_score=spectacle_score,
        playback_speed=playback_speed
    )

    return html_path


def record_html_to_video_playwright(
    html_path: Path,
    output_video: Path,
    duration_seconds: float,
    fps: int = 30,
    resolution: tuple = (1920, 1080)
):
    """
    Record HTML animation to video using Playwright.

    Args:
        html_path: Path to HTML file
        output_video: Path to save video file
        duration_seconds: Duration of recording in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not available")
        sys.exit(1)

    print(f"  Recording {html_path.name} to video...")

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)

        # Create context with viewport size
        context = browser.new_context(
            viewport={'width': resolution[0], 'height': resolution[1]},
            record_video_dir=str(output_video.parent),
            record_video_size={'width': resolution[0], 'height': resolution[1]}
        )

        # Create page
        page = context.new_page()

        # Load HTML
        page.goto(f'file://{html_path.absolute()}')

        # Wait for animation to complete
        # Add extra 2 seconds for loading
        time.sleep(duration_seconds + 2)

        # Close to finalize video
        page.close()
        context.close()
        browser.close()

    # Playwright saves video as webm - need to convert to mp4
    # Find the webm video file
    video_files = list(output_video.parent.glob("*.webm"))
    if video_files:
        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)

        # Convert webm to mp4
        convert_cmd = [
            'ffmpeg', '-y',
            '-i', str(latest_video),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        subprocess.run(convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Remove the webm file
        latest_video.unlink()

    print(f"  ✅ Saved to {output_video}")


def record_html_to_video_ffmpeg(
    html_path: Path,
    output_video: Path,
    duration_seconds: float,
    fps: int = 30,
    resolution: tuple = (1920, 1080)
):
    """
    Record HTML animation to video using ffmpeg with headless Chrome.

    This method uses Chrome's screenshot capability and ffmpeg to create video.
    Falls back method if Playwright is not available.
    """
    print(f"  Recording {html_path.name} to video (using Chrome + ffmpeg)...")
    print("  ⚠️  Note: This requires Chrome with --headless support")
    print("  For best results, use Playwright (pip install playwright)")

    # This is a placeholder - full implementation would require:
    # 1. Chrome DevTools Protocol to control browser
    # 2. Taking screenshots at each frame
    # 3. Using ffmpeg to stitch screenshots into video

    print("  ❌ Chrome+ffmpeg method not yet implemented")
    print("  Please install Playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


def create_title_card(
    text: str,
    output_path: Path,
    duration: float = 3.0,
    fps: int = 30,
    resolution: tuple = (1920, 1080)
):
    """
    Create a title card video using ffmpeg.

    Args:
        text: Title text to display
        output_path: Path to save video file
        duration: Duration in seconds
        fps: Frames per second
        resolution: Video resolution
    """
    print(f"  Creating title card: '{text}'")

    # Use ffmpeg to create a simple colored background with text
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'color=c=0x1a1a2e:s={resolution[0]}x{resolution[1]}:d={duration}:r={fps}',
        '-vf', f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
               f"text='{text}':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✅ Title card saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Warning: Failed to create title card: {e}")
        print("  Continuing without title card...")


def concatenate_videos(
    video_paths: List[Path],
    output_path: Path
):
    """
    Concatenate multiple videos into a single montage.

    Args:
        video_paths: List of video file paths
        output_path: Path to save final montage
    """
    print(f"\n📹 Concatenating {len(video_paths)} videos...")

    # Create a temporary file list for ffmpeg concat
    concat_file = output_path.parent / "concat_list.txt"

    with open(concat_file, 'w') as f:
        for video_path in video_paths:
            f.write(f"file '{video_path.absolute()}'\n")

    # Concatenate with ffmpeg
    # Use copy since all clips are now h264/mp4 (converted from webm)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Montage saved to {output_path}")

        # Cleanup
        concat_file.unlink()

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to concatenate videos: {e}")
        print(f"stderr: {e.stderr.decode()}")
        sys.exit(1)


def create_montage(
    run_dir: Path,
    output_path: Path = None,
    playback_speed: float = 3.0,
    fps: int = 30,
    resolution: tuple = (1920, 1080),
    use_playwright: bool = True
):
    """
    Create a training montage from saved replays.

    Args:
        run_dir: Training run directory (contains replays/ and replay_index.json)
        output_path: Path to save final montage video
        playback_speed: Playback speed multiplier (3.0 = 3x faster)
        fps: Frames per second
        resolution: Video resolution
        use_playwright: Use Playwright for recording (recommended)
    """
    print("\n" + "="*80)
    print("CREATING TRAINING MONTAGE")
    print("="*80)

    # Check dependencies
    check_ffmpeg()

    if use_playwright:
        if not check_playwright():
            print("\n❌ Playwright not available. Please install:")
            print("  pip install playwright")
            print("  playwright install chromium")
            sys.exit(1)
    else:
        chrome_path = check_chrome()
        if not chrome_path:
            print("\n❌ Chrome not available. Please install Chrome or use --use-playwright")
            sys.exit(1)

    # Paths
    temp_dir = run_dir / "montage_temp"
    temp_dir.mkdir(exist_ok=True)

    if output_path is None:
        output_path = run_dir / "training_montage.mp4"

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

    # Separate curriculum and population replays
    curriculum_replays = [r for r in replays if r['stage_type'] == 'curriculum']
    population_replays = [r for r in replays if r['stage_type'] == 'population']

    print(f"  - Curriculum: {len(curriculum_replays)} replays")
    print(f"  - Population: {len(population_replays)} replays")

    # Sort replays
    curriculum_replays.sort(key=lambda r: (r['stage'], r['spectacle_rank']))
    population_replays.sort(key=lambda r: (r['stage'], r['spectacle_rank']))

    video_clips = []

    # Part 1: Curriculum montage
    if curriculum_replays:
        print("\n" + "="*80)
        print("PART 1: CURRICULUM TRAINING (From Zero to Graduate)")
        print("="*80)

        # Title card
        title_card_path = temp_dir / "title_curriculum.mp4"
        create_title_card(
            "Part 1: From Zero to Graduate",
            title_card_path,
            duration=3.0,
            fps=fps,
            resolution=resolution
        )
        video_clips.append(title_card_path)

        # Render each curriculum replay
        for i, replay_meta in enumerate(curriculum_replays, 1):
            print(f"\n[{i}/{len(curriculum_replays)}] Processing {replay_meta['stage']} ({replay_meta['spectacle_rank']})")

            # Find replay file in correct subdirectory
            source_dir = replay_meta.get('_source_dir', 'curriculum')
            replays_dir = run_dir / source_dir / "replays"
            replay_files = list(replays_dir.glob(f"{replay_meta['stage']}_{replay_meta['spectacle_rank']}_*.json.gz"))
            if not replay_files:
                print(f"  ⚠️  Warning: Replay file not found in {replays_dir}, skipping")
                continue

            replay_path = replay_files[0]

            # Render to HTML
            html_path = temp_dir / f"curriculum_{i}.html"
            render_replay_to_html(replay_path, html_path, playback_speed)

            # Record to video
            video_path = temp_dir / f"curriculum_{i}.mp4"

            # Estimate duration (ticks / 12 ticks per second / playback_speed)
            replay_data = load_replay(str(replay_path))
            duration = (replay_data['result']['total_ticks'] / 12.0) / playback_speed

            if use_playwright:
                record_html_to_video_playwright(html_path, video_path, duration, fps, resolution)
            else:
                record_html_to_video_ffmpeg(html_path, video_path, duration, fps, resolution)

            video_clips.append(video_path)

    # Part 2: Population montage
    if population_replays:
        print("\n" + "="*80)
        print("PART 2: POPULATION EVOLUTION (Natural Selection)")
        print("="*80)

        # Title card
        title_card_path = temp_dir / "title_population.mp4"
        create_title_card(
            "Part 2: Natural Selection",
            title_card_path,
            duration=3.0,
            fps=fps,
            resolution=resolution
        )
        video_clips.append(title_card_path)

        # Render each population replay
        for i, replay_meta in enumerate(population_replays, 1):
            print(f"\n[{i}/{len(population_replays)}] Processing {replay_meta['stage']} ({replay_meta['spectacle_rank']})")

            # Find replay file in correct subdirectory
            source_dir = replay_meta.get('_source_dir', 'population')
            replays_dir = run_dir / source_dir / "replays"
            replay_files = list(replays_dir.glob(f"{replay_meta['stage']}_{replay_meta['spectacle_rank']}_*.json.gz"))
            if not replay_files:
                print(f"  ⚠️  Warning: Replay file not found in {replays_dir}, skipping")
                continue

            replay_path = replay_files[0]

            # Render to HTML
            html_path = temp_dir / f"population_{i}.html"
            render_replay_to_html(replay_path, html_path, playback_speed)

            # Record to video
            video_path = temp_dir / f"population_{i}.mp4"

            # Estimate duration
            replay_data = load_replay(str(replay_path))
            duration = (replay_data['result']['total_ticks'] / 12.0) / playback_speed

            if use_playwright:
                record_html_to_video_playwright(html_path, video_path, duration, fps, resolution)
            else:
                record_html_to_video_ffmpeg(html_path, video_path, duration, fps, resolution)

            video_clips.append(video_path)

    # Concatenate all clips
    if video_clips:
        concatenate_videos(video_clips, output_path)

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)

        print("\n" + "="*80)
        print("✅ MONTAGE COMPLETE!")
        print("="*80)
        print(f"Output: {output_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Clips: {len(video_clips)}")
        print("\nYou can now:")
        print(f"  - Play: mpv {output_path}")
        print(f"  - Upload to YouTube/Twitter/etc.")
        print(f"  - Share: {output_path}")

        # Optionally clean up temp files
        print(f"\nTemp files saved in: {temp_dir}")
        print("Run: rm -rf {} to clean up".format(temp_dir))
    else:
        print("\n⚠️  No video clips created - check for errors above")


def main():
    parser = argparse.ArgumentParser(
        description="Create training montage from recorded replays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create montage from training run
  python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000

  # Faster playback (5x speed)
  python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000 --speed 5.0

  # Custom output path
  python scripts/montage/create_montage.py --run-dir outputs/progressive_20251114_120000 --output my_montage.mp4
        """
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing replays/ and replay_index.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: RUN_DIR/training_montage.mp4)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="Playback speed multiplier (default: 3.0 = 3x faster)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video framerate (default: 30)"
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Video resolution WIDTHxHEIGHT (default: 1920x1080)"
    )

    parser.add_argument(
        "--use-chrome",
        action="store_true",
        help="Use Chrome instead of Playwright (not recommended)"
    )

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"ERROR: Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1920x1080)")
        sys.exit(1)

    # Create montage
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else None

    create_montage(
        run_dir=run_dir,
        output_path=output_path,
        playback_speed=args.speed,
        fps=args.fps,
        resolution=resolution,
        use_playwright=not args.use_chrome
    )


if __name__ == "__main__":
    main()
