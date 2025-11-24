#!/usr/bin/env python3
"""
Create HTML5 Training Montage from Recorded Replays

Generates a single HTML file with all fights embedded, showing
the progression from novice to expert across all curriculum levels.
"""

import json
import gzip
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys
from datetime import datetime

from src.telemetry.replay_store import load_replay
from src.renderer.html_renderer import HtmlRenderer


def load_progressive_replays(run_dir: Path) -> Dict[str, Any]:
    """Load progressive replay index and replays."""
    # Check for progressive replay index
    progressive_index_path = run_dir / "curriculum" / "progressive_replay_index.json"
    if not progressive_index_path.exists():
        progressive_index_path = run_dir / "progressive_replay_index.json"

    if not progressive_index_path.exists():
        # Try to build index from existing replays
        print("⚠️  No progressive replay index found, building from replays...")

        # Look for replay directory
        replays_dir = run_dir / "curriculum" / "progressive_replays"
        if not replays_dir.exists():
            replays_dir = run_dir / "progressive_replays"

        if not replays_dir.exists() or not any(replays_dir.glob("*.json.gz")):
            print(f"❌ No progressive replays found in {run_dir}")
            print("Please run training with --record-replays flag")
            sys.exit(1)

        # Build index from replay files
        index_data = build_index_from_replays(replays_dir)
        print(f"✅ Built index from {index_data['total_replays']} replays")
    else:
        with open(progressive_index_path) as f:
            index_data = json.load(f)
        print(f"✅ Found {index_data['total_replays']} progressive replays")

    return index_data


def build_index_from_replays(replays_dir: Path) -> Dict[str, Any]:
    """Build progressive replay index from replay files."""
    import re

    replay_files = sorted(replays_dir.glob("*.json.gz"))
    replays = []

    for replay_path in replay_files:
        # Parse filename to extract metadata
        # Format: level_{num}_{name}_ep_{episode:05d}_wr_{winrate:03d}.json.gz
        filename = replay_path.stem.replace('.json', '')
        match = re.match(r'level_(\d+)_(.+)_ep_(\d+)_wr_(\d+)', filename)

        if match:
            level_num = int(match.group(1))
            level_name = match.group(2).replace('_', ' ').title()
            episode = int(match.group(3))
            win_rate = int(match.group(4)) / 100.0

            # Load replay to get additional metadata
            try:
                with gzip.open(replay_path, 'rt') as f:
                    data = json.load(f)

                replays.append({
                    'level_name': level_name,
                    'level_num': level_num,
                    'episode': episode,
                    'total_episodes': 1000,  # Estimate
                    'win_rate': win_rate,
                    'recent_rewards': [],
                    'timestamp': datetime.now().isoformat(),
                    'fighter_a': data['result'].get('fighter_a_name', 'AI_Fighter'),
                    'fighter_b': data['result'].get('fighter_b_name', 'Opponent'),
                    'winner': data['result']['winner'],
                    'duration_ticks': data['result']['total_ticks'],
                    'notes': f"Level {level_num} - Episode {episode}"
                })
            except Exception as e:
                print(f"  ⚠️  Could not parse {replay_path.name}: {e}")

    return {
        'total_replays': len(replays),
        'recording_strategy': {
            'early_phase_interval': 10,
            'mid_phase_interval': 50,
            'late_phase_interval': 100,
            'early_phase_end': 0.2,
            'mid_phase_end': 0.8
        },
        'replays': replays
    }


def load_replay_telemetry(replay_path: Path) -> Dict[str, Any]:
    """Load telemetry data from a replay file."""
    try:
        with gzip.open(replay_path, 'rt') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Could not load replay {replay_path}: {e}")
        return None


def generate_html_montage(
    run_dir: Path,
    output_path: Path,
    playback_speed: float = 2.0,
    verbose: bool = True
):
    """
    Generate a single HTML file with all fights embedded.

    Args:
        run_dir: Training run directory
        output_path: Path to save HTML file
        playback_speed: Playback speed multiplier
        verbose: Print progress
    """
    print("\n" + "="*80)
    print("CREATING HTML5 TRAINING MONTAGE")
    print("="*80)

    # Load progressive replay index
    index_data = load_progressive_replays(run_dir)
    replays = index_data['replays']

    # Group replays by level
    levels = {}
    for replay in replays:
        level_num = replay['level_num']
        if level_num not in levels:
            levels[level_num] = []
        levels[level_num].append(replay)

    # Sort levels and replays within each level by episode
    sorted_levels = sorted(levels.keys())
    for level_num in sorted_levels:
        levels[level_num].sort(key=lambda r: r['episode'])

    if verbose:
        print(f"\nFound replays for {len(levels)} levels:")
        for level_num in sorted_levels:
            level_replays = levels[level_num]
            level_name = level_replays[0]['level_name'] if level_replays else "Unknown"
            print(f"  Level {level_num} ({level_name}): {len(level_replays)} replays")

    # Load actual replay telemetry data
    all_replay_data = []
    replays_dir = run_dir / "curriculum" / "progressive_replays"
    if not replays_dir.exists():
        replays_dir = run_dir / "progressive_replays"

    print(f"\nLoading replay telemetry from {replays_dir}...")
    loaded_count = 0

    for level_num in sorted_levels:
        for replay_meta in levels[level_num]:
            # Construct replay filename
            filename = (
                f"level_{replay_meta['level_num']}_{replay_meta['level_name'].lower().replace(' ', '_')}_"
                f"ep_{replay_meta['episode']:05d}_wr_{int(replay_meta['win_rate']*100):03d}.json.gz"
            )
            replay_path = replays_dir / filename

            if replay_path.exists():
                replay_data = load_replay_telemetry(replay_path)
                if replay_data:
                    # Add metadata to replay data
                    replay_data['meta'] = replay_meta
                    all_replay_data.append(replay_data)
                    loaded_count += 1
                    if verbose and loaded_count % 10 == 0:
                        print(f"  Loaded {loaded_count}/{len(replays)} replays...")
            else:
                if verbose:
                    print(f"  ⚠️  Replay file not found: {filename}")

    print(f"✅ Loaded {loaded_count} replays successfully")

    # Generate the HTML file
    html_content = generate_html_content(
        all_replay_data,
        index_data.get('recording_strategy', {}),
        playback_speed
    )

    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "="*80)
    print("✅ HTML MONTAGE COMPLETE!")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Replays: {loaded_count}")
    print(f"\nTo view: Open {output_path} in your browser")
    print("Controls:")
    print("  - Space: Play/Pause")
    print("  - Left/Right: Previous/Next fight")
    print("  - 1/2/3/4: Playback speed (1x/2x/4x/8x)")


def generate_html_content(
    replay_data: List[Dict[str, Any]],
    recording_strategy: Dict[str, Any],
    playback_speed: float
) -> str:
    """Generate the HTML content with embedded replay data and the existing sophisticated player."""

    # Get the base template from the existing renderer
    renderer = HtmlRenderer()
    base_template = renderer._get_html_template()

    # Extract the JavaScript rendering code from the template
    # Find the part between <script> tags
    script_start = base_template.find('<script>')
    script_end = base_template.find('</script>')

    if script_start == -1 or script_end == -1:
        print("⚠️  Could not extract JavaScript from template")
        return ""

    # Get the JavaScript code (without the script tags)
    js_code = base_template[script_start + 8:script_end]

    # We need to modify the original code to work with dynamic replay loading
    # Remove the original const declarations - we'll declare them as let later
    js_code = js_code.replace('const ticks = REPLAY_DATA.telemetry.ticks;', '')
    js_code = js_code.replace('const config = REPLAY_DATA.telemetry.config;', '')
    js_code = js_code.replace('const arenaWidth = config.arena_width;', '')
    js_code = js_code.replace('const dt = config.dt;', '')
    js_code = js_code.replace('let playbackSpeed = REPLAY_DATA.playback_speed || 1.0;', 'let playbackSpeed = 1.0;')

    # Convert replay data to JSON string
    replays_json = json.dumps(replay_data)

    # Build the full HTML with montage-specific features
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Montage - AI Combat Learning Journey</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }}

        h1 {{
            color: #ff6b6b;
            margin-bottom: 10px;
            text-align: center;
        }}

        .montage-header {{
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: #0f3460;
            border-radius: 8px;
            width: 100%;
            max-width: 1200px;
        }}

        .replay-selector {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }}

        .level-section {{
            background: #16213e;
            border: 2px solid #0f3460;
            border-radius: 5px;
            padding: 10px;
        }}

        .level-title {{
            color: #4ecca3;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }}

        .replay-button {{
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            transition: all 0.3s;
            margin: 2px;
        }}

        .replay-button:hover {{
            background: #ff5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
        }}

        .replay-button.active {{
            background: #4ecca3;
            box-shadow: 0 0 10px rgba(78, 204, 163, 0.5);
        }}

        .container {{
            max-width: 1200px;
            width: 100%;
        }}

        #canvas {{
            background: #16213e;
            border: 3px solid #0f3460;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }}

        .controls {{
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}

        button {{
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
        }}

        button:hover {{
            background: #ff5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
        }}

        button:active {{
            transform: translateY(0);
        }}

        button:disabled {{
            background: #555;
            cursor: not-allowed;
            transform: none;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}

        .stat-box {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff6b6b;
        }}

        .stat-box h3 {{
            color: #ff6b6b;
            margin-bottom: 10px;
            font-size: 14px;
        }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 13px;
        }}

        .stat-value {{
            color: #4ecca3;
            font-weight: bold;
        }}

        #tick-info {{
            text-align: center;
            font-size: 18px;
            color: #4ecca3;
            margin: 10px 0;
        }}

        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .speed-control input {{
            width: 100px;
        }}

        .speed-control span {{
            min-width: 40px;
            color: #4ecca3;
        }}

        .navigation-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            background: #0f3460;
            padding: 10px;
            border-radius: 5px;
        }}

        .current-replay-info {{
            color: #4ecca3;
            font-weight: bold;
            margin: 0 20px;
        }}

        .no-replay {{
            text-align: center;
            color: #ff6b6b;
            font-size: 24px;
            padding: 100px 20px;
        }}
    </style>
</head>
<body>
    <div class="montage-header">
        <h1>⚔️ ATOM COMBAT - TRAINING MONTAGE ⚔️</h1>
        <p style="color: #4ecca3; margin: 10px 0;">Watch the AI's journey from novice to expert</p>

        <div class="replay-selector" id="replaySelector">
            <!-- Will be populated with replay buttons -->
        </div>

        <div class="navigation-controls">
            <button id="prevReplay">◀ Previous</button>
            <span class="current-replay-info" id="currentReplayInfo">Loading...</span>
            <button id="nextReplay">Next ▶</button>
        </div>
    </div>

    <div class="container">
        <div class="controls">
            <button id="playPauseBtn">▶️ Play</button>
            <button id="restartBtn">🔄 Restart</button>
            <button id="stepBtn">⏭️ Step</button>
            <button id="autoPlayBtn" style="background: #4ecca3;">🔄 Auto-Play: ON</button>
            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="0.25" max="10" step="0.25" value="{playback_speed}">
                <span id="speedValue">{playback_speed}x</span>
            </div>
        </div>

        <div id="tick-info">Tick 0 / 0</div>

        <canvas id="canvas" width="1200" height="600"></canvas>

        <div class="stats" id="stats"></div>
    </div>

    <script>
        // All replay data
        const ALL_REPLAYS = {replays_json};
        let currentReplayIndex = 0;
        let REPLAY_DATA = null;

        // Initialize replay selector
        function initReplaySelector() {{
            const selector = document.getElementById('replaySelector');
            selector.innerHTML = '';

            // Group replays by level
            const levels = {{}};
            ALL_REPLAYS.forEach((replay, index) => {{
                const levelNum = replay.meta?.level_num || 1;
                if (!levels[levelNum]) {{
                    levels[levelNum] = [];
                }}
                levels[levelNum].push({{replay, index}});
            }});

            // Create buttons for each level
            Object.keys(levels).sort((a, b) => a - b).forEach(levelNum => {{
                const levelDiv = document.createElement('div');
                levelDiv.className = 'level-section';

                const title = document.createElement('div');
                title.className = 'level-title';
                title.textContent = `Level ${{levelNum}}: ${{levels[levelNum][0].replay.meta?.level_name || 'Unknown'}}`;
                levelDiv.appendChild(title);

                levels[levelNum].forEach(item => {{
                    const btn = document.createElement('button');
                    btn.className = 'replay-button';
                    btn.textContent = `Ep ${{item.replay.meta?.episode || '?'}}`;
                    btn.title = `Episode ${{item.replay.meta?.episode}}, Win Rate: ${{(item.replay.meta?.win_rate * 100 || 0).toFixed(0)}}%`;
                    btn.onclick = () => loadReplay(item.index);
                    btn.id = `replay-btn-${{item.index}}`;
                    levelDiv.appendChild(btn);
                }});

                selector.appendChild(levelDiv);
            }});

            // Load first replay
            if (ALL_REPLAYS.length > 0) {{
                loadReplay(0);
            }} else {{
                showNoReplays();
            }}
        }}

        function showNoReplays() {{
            const canvas = document.getElementById('canvas');
            canvas.style.display = 'none';
            const container = canvas.parentElement;
            const message = document.createElement('div');
            message.className = 'no-replay';
            message.innerHTML = '❌ No replay data available<br><br>The replays have no telemetry data.';
            container.appendChild(message);
        }}

        function loadReplay(index) {{
            if (index < 0 || index >= ALL_REPLAYS.length) return;

            currentReplayIndex = index;
            REPLAY_DATA = ALL_REPLAYS[index];

            // Update UI
            document.querySelectorAll('.replay-button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            const activeBtn = document.getElementById(`replay-btn-${{index}}`);
            if (activeBtn) activeBtn.classList.add('active');

            // Update info
            const meta = REPLAY_DATA.meta || {{}};
            document.getElementById('currentReplayInfo').textContent =
                `Level ${{meta.level_num || '?'}} - Episode ${{meta.episode || '?'}} (${{(meta.win_rate * 100 || 0).toFixed(0)}}%)`;

            // Update navigation buttons
            document.getElementById('prevReplay').disabled = index === 0;
            document.getElementById('nextReplay').disabled = index === ALL_REPLAYS.length - 1;

            // Reinitialize the replay data
            initializeReplayData();
            restart();
        }}

        function resetReplay() {{
            currentTick = 0;
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶️ Play';

            // Check if replay has telemetry
            if (!REPLAY_DATA || !REPLAY_DATA.telemetry || !REPLAY_DATA.telemetry.ticks ||
                REPLAY_DATA.telemetry.ticks.length === 0) {{
                showNoTelemetry();
                return;
            }}

            // Set up replay data
            ticks = REPLAY_DATA.telemetry.ticks;
            config = REPLAY_DATA.telemetry.config;
            arenaWidth = config.arena_width;
            dt = config.dt;

            render();
        }}

        function showNoTelemetry() {{
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#ff6b6b';
            ctx.font = 'bold 24px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('No telemetry data in this replay', canvas.width / 2, canvas.height / 2 - 20);

            ctx.fillStyle = '#aaa';
            ctx.font = '16px Courier New';
            ctx.fillText('The fight was recorded but has 0 ticks', canvas.width / 2, canvas.height / 2 + 20);
        }}

        // Navigation handlers
        document.getElementById('prevReplay').addEventListener('click', () => {{
            if (currentReplayIndex > 0) {{
                loadReplay(currentReplayIndex - 1);
            }}
        }});

        document.getElementById('nextReplay').addEventListener('click', () => {{
            if (currentReplayIndex < ALL_REPLAYS.length - 1) {{
                loadReplay(currentReplayIndex + 1);
            }}
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'ArrowLeft':
                    if (currentReplayIndex > 0) loadReplay(currentReplayIndex - 1);
                    break;
                case 'ArrowRight':
                    if (currentReplayIndex < ALL_REPLAYS.length - 1) loadReplay(currentReplayIndex + 1);
                    break;
                case ' ':
                    e.preventDefault();
                    togglePlayPause();
                    break;
                case 'r':
                    restart();
                    break;
            }}
        }});

        // Variables that need to be initialized per replay
        let ticks = null;
        let config = null;
        let arenaWidth = null;
        let dt = null;

        function initializeReplayData() {{
            if (!REPLAY_DATA || !REPLAY_DATA.telemetry) {{
                console.error('No replay data available');
                return;
            }}

            // Set up the variables expected by the original renderer
            ticks = REPLAY_DATA.telemetry.ticks;
            config = REPLAY_DATA.telemetry.config;
            arenaWidth = config.arena_width;
            dt = config.dt;

            // Reset playback state
            currentTick = 0;
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶️ Play';
        }}

        // Auto-play settings
        let autoPlayEnabled = true;
        let autoPlayDelay = 2000; // 2 seconds between fights

        function toggleAutoPlay() {{
            autoPlayEnabled = !autoPlayEnabled;
            const btn = document.getElementById('autoPlayBtn');
            if (autoPlayEnabled) {{
                btn.textContent = '🔄 Auto-Play: ON';
                btn.style.background = '#4ecca3';
            }} else {{
                btn.textContent = '🔄 Auto-Play: OFF';
                btn.style.background = '#888';
            }}
        }}

        // Event handlers for buttons
        document.getElementById('playPauseBtn').onclick = togglePlayPause;
        document.getElementById('restartBtn').onclick = restart;
        document.getElementById('stepBtn').onclick = step;
        document.getElementById('autoPlayBtn').onclick = toggleAutoPlay;
        document.getElementById('speedSlider').oninput = function(e) {{
            playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = playbackSpeed.toFixed(2) + 'x';
        }};

        {js_code}

        // Initialize on load
        initReplaySelector();
        if (ALL_REPLAYS.length > 0) {{
            loadReplay(0);  // Load first replay
        }}
    </script>
</body>
</html>
"""

    return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create HTML training montage from progressive replays"
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to training run directory containing replays"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: run_dir/training_montage.html)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="Default playback speed (default: 2.0)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    # Determine paths
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "training_montage.html"

    # Generate montage
    generate_html_montage(
        run_dir=run_dir,
        output_path=output_path,
        playback_speed=args.speed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()