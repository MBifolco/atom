#!/usr/bin/env python3
"""
Create HTML5 Training Montage from Recorded Replays - V2
This version properly handles dynamic replay loading.
"""

import json
import gzip
from pathlib import Path
from typing import List, Dict, Any
import argparse
import sys
from datetime import datetime
import re

from src.renderer.html_renderer import HtmlRenderer


def build_index_from_replays(replays_dir: Path) -> Dict[str, Any]:
    """Build progressive replay index from replay files."""
    replay_files = sorted(replays_dir.glob("*.json.gz"))
    replays = []

    for replay_path in replay_files:
        # Parse filename to extract metadata
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
                    'win_rate': win_rate,
                    'winner': data['result']['winner'],
                    'duration_ticks': data['result']['total_ticks'],
                })
            except Exception as e:
                print(f"  ⚠️  Could not parse {replay_path.name}: {e}")

    return replays


def generate_html_montage(run_dir: Path, output_path: Path):
    """Generate HTML montage with properly working dynamic replay loading."""

    print("\n" + "="*80)
    print("CREATING HTML5 TRAINING MONTAGE (V2)")
    print("="*80)

    # Find replay directory
    replays_dir = run_dir / "curriculum" / "progressive_replays"
    if not replays_dir.exists():
        replays_dir = run_dir / "progressive_replays"

    if not replays_dir.exists() or not any(replays_dir.glob("*.json.gz")):
        print(f"❌ No progressive replays found in {run_dir}")
        sys.exit(1)

    # Load all replays
    replay_files = sorted(replays_dir.glob("*.json.gz"))
    print(f"Found {len(replay_files)} replay files")

    # Build index
    index = build_index_from_replays(replays_dir)
    print(f"✅ Loaded metadata for {len(index)} replays")

    # Load actual replay data
    all_replays = []
    for i, replay_path in enumerate(replay_files):
        try:
            with gzip.open(replay_path, 'rt') as f:
                data = json.load(f)

            # Add metadata from index
            if i < len(index):
                data['meta'] = index[i]

            all_replays.append(data)

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(replay_files)} replays...")
        except Exception as e:
            print(f"  ⚠️  Could not load {replay_path.name}: {e}")

    print(f"✅ Successfully loaded {len(all_replays)} complete replays")

    # Get the base HTML renderer template
    renderer = HtmlRenderer()
    base_template = renderer._get_html_template()

    # Extract just the rendering functions (not the variable declarations)
    # We'll grab from "function togglePlayPause" to just before "// Initialize"
    func_start = base_template.find('function togglePlayPause')
    func_end = base_template.find('// Initialize')
    if func_start == -1 or func_end == -1:
        print("❌ Could not extract rendering functions")
        sys.exit(1)

    core_render_functions = base_template[func_start:func_end]

    # We'll include all functions and then override animate with our own

    # Generate the complete HTML
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

        .navigation-controls {{
            margin-top: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
        }}

        .navigation-controls button {{
            background: #4ecca3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            transition: all 0.3s;
        }}

        .navigation-controls button:hover:not(:disabled) {{
            background: #45b393;
            transform: translateY(-2px);
        }}

        .navigation-controls button:disabled {{
            background: #555;
            cursor: not-allowed;
        }}

        .current-replay-info {{
            font-size: 16px;
            color: #ffd93d;
            font-weight: bold;
        }}

        .container {{
            max-width: 1200px;
            width: 100%;
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

        button:hover:not(:disabled) {{
            background: #ff5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
        }}

        button:disabled {{
            background: #555;
            cursor: not-allowed;
        }}

        #canvas {{
            background: #16213e;
            border: 3px solid #0f3460;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
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

        .loading {{
            text-align: center;
            color: #ff6b6b;
            font-size: 24px;
            padding: 100px 20px;
        }}

        .error {{
            color: #ff6b6b;
            background: rgba(255, 0, 0, 0.1);
            border: 2px solid #ff6b6b;
            padding: 20px;
            margin: 20px;
            border-radius: 8px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="montage-header">
        <h1>⚔️ ATOM COMBAT - TRAINING MONTAGE ⚔️</h1>
        <p style="color: #4ecca3; margin: 10px 0;">Watch the AI's journey from novice to expert</p>
        <p style="color: #ffd93d; margin: 10px 0;">Loaded {len(all_replays)} progressive replays</p>

        <div class="navigation-controls">
            <button id="prevReplay" onclick="loadPreviousReplay()">◀ Previous</button>
            <span class="current-replay-info" id="currentReplayInfo">Loading...</span>
            <button id="nextReplay" onclick="loadNextReplay()">Next ▶</button>
        </div>
    </div>

    <div class="container">
        <div class="controls">
            <button id="playPauseBtn" onclick="togglePlayPause()">▶️ Play</button>
            <button id="restartBtn" onclick="restart()">🔄 Restart</button>
            <button id="stepBtn" onclick="step()">⏭️ Step</button>
            <button id="autoPlayBtn" onclick="toggleAutoPlay()" style="background: #4ecca3;">🔄 Auto-Play: ON</button>
            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="0.5" max="10" step="0.5" value="2" oninput="updateSpeed(event)">
                <span id="speedValue">2.00x</span>
            </div>
        </div>

        <div id="tick-info">Tick 0 / 0</div>

        <canvas id="canvas" width="1200" height="600"></canvas>

        <div class="stats" id="stats"></div>
    </div>

    <div id="errors" class="error" style="display:none;"></div>

    <script>
        // All replay data embedded
        const ALL_REPLAYS = {json.dumps(all_replays)};

        // Current state
        let currentReplayIndex = 0;
        let REPLAY_DATA = null;

        // Replay-specific variables (will be set when loading a replay)
        let ticks = null;
        let config = null;
        let arenaWidth = null;
        let dt = null;

        // Animation state
        let currentTick = 0;
        let isPlaying = false;
        let playbackSpeed = 2.0;
        let lastFrameTime = 0;
        let animationId = null;

        // Canvas setup
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // Colors
        const COLORS = {{
            background: '#16213e',
            arena: '#0f3460',
            fighterA: '#ff6b6b',
            fighterB: '#4ecca3',
            hp: '#ff6b6b',
            stamina: '#4ecca3',
            collision: '#ffd93d',
            text: '#eeeeee',
            textDim: '#aaaaaa'
        }};

        // Stance visual styles
        const STANCE_STYLES = {{
            neutral: {{ shape: 'circle', size: 1.0 }},
            extended: {{ shape: 'triangle-right', size: 1.2 }},
            retracted: {{ shape: 'square', size: 0.8 }},
            defending: {{ shape: 'hexagon', size: 1.1 }}
        }};

        // Load a replay by index
        function loadReplay(index) {{
            try {{
                if (index < 0 || index >= ALL_REPLAYS.length) {{
                    console.error('Invalid replay index:', index);
                    return;
                }}

                currentReplayIndex = index;
                REPLAY_DATA = ALL_REPLAYS[index];

                if (!REPLAY_DATA || !REPLAY_DATA.telemetry || !REPLAY_DATA.telemetry.ticks) {{
                    console.error('Invalid replay data at index', index);
                    showError('Invalid replay data');
                    return;
                }}

                // Set up replay-specific variables
                ticks = REPLAY_DATA.telemetry.ticks;
                config = REPLAY_DATA.telemetry.config;
                arenaWidth = config.arena_width;
                dt = config.dt;

                // Reset state
                currentTick = 0;
                isPlaying = false;
                document.getElementById('playPauseBtn').textContent = '▶️ Play';

                // Update UI
                updateReplayInfo();
                document.getElementById('prevReplay').disabled = index === 0;
                document.getElementById('nextReplay').disabled = index === ALL_REPLAYS.length - 1;

                // Clear any errors
                document.getElementById('errors').style.display = 'none';

                // Render first frame
                render();

                console.log('Loaded replay', index, 'with', ticks.length, 'ticks');
            }} catch (error) {{
                console.error('Error loading replay:', error);
                showError('Error loading replay: ' + error.message);
            }}
        }}

        function loadPreviousReplay() {{
            if (currentReplayIndex > 0) {{
                loadReplay(currentReplayIndex - 1);
            }}
        }}

        function loadNextReplay() {{
            if (currentReplayIndex < ALL_REPLAYS.length - 1) {{
                loadReplay(currentReplayIndex + 1);
            }}
        }}

        function updateReplayInfo() {{
            const meta = REPLAY_DATA.meta || {{}};
            const info = `Level ${{meta.level_num || '?'}} - Episode ${{meta.episode || '?'}} ` +
                        `(Win Rate: ${{((meta.win_rate || 0) * 100).toFixed(0)}}%)`;
            document.getElementById('currentReplayInfo').textContent = info;
        }}

        function showError(message) {{
            const errorDiv = document.getElementById('errors');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }}

        // Control functions are included in core_render_functions
        // updateSpeed is already in core functions, just need toggleAutoPlay

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

        // Auto-play settings
        let autoPlayEnabled = true;
        let autoPlayDelay = 2000; // 2 seconds between fights

        // Core rendering functions from the original renderer
        {core_render_functions}

        // Override the animate function with our auto-play version
        function animate(timestamp) {{
            if (isPlaying && ticks && currentTick < ticks.length - 1) {{
                const elapsed = timestamp - lastFrameTime;
                const frameTime = (dt * 1000) / playbackSpeed;

                if (elapsed >= frameTime) {{
                    currentTick++;
                    lastFrameTime = timestamp;
                    render();
                }}
            }} else if (ticks && currentTick >= ticks.length - 1 && isPlaying) {{
                // Fight just ended
                isPlaying = false;
                document.getElementById('playPauseBtn').textContent = '▶️ Play';
                console.log(`Fight ended at tick ${{currentTick}} of ${{ticks.length}}`);

                // Auto-play next fight if enabled and not at the last replay
                if (autoPlayEnabled && currentReplayIndex < ALL_REPLAYS.length - 1) {{
                    console.log(`Auto-playing next fight in ${{autoPlayDelay}}ms...`);
                    setTimeout(() => {{
                        console.log(`Loading replay ${{currentReplayIndex + 1}} of ${{ALL_REPLAYS.length}}`);
                        loadNextReplay();
                        // Auto-start the next replay
                        setTimeout(() => {{
                            isPlaying = true;
                            document.getElementById('playPauseBtn').textContent = '⏸️ Pause';
                            console.log('Started playing next fight');
                        }}, 500);
                    }}, autoPlayDelay);
                }} else if (autoPlayEnabled && currentReplayIndex >= ALL_REPLAYS.length - 1) {{
                    console.log('Reached last replay, stopping auto-play');
                }}
            }}

            animationId = requestAnimationFrame(animate);
        }}

        // Start animation loop
        requestAnimationFrame(animate);

        // Initialize with first replay and start playing if auto-play is on
        if (ALL_REPLAYS.length > 0) {{
            loadReplay(0);
            // Auto-start first replay
            if (autoPlayEnabled) {{
                setTimeout(() => {{
                    isPlaying = true;
                    document.getElementById('playPauseBtn').textContent = '⏸️ Pause';
                }}, 500);
            }}
        }} else {{
            showError('No replays available');
        }}
    </script>
</body>
</html>"""

    # Save the HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "="*80)
    print("✅ HTML MONTAGE COMPLETE!")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Replays: {len(all_replays)}")
    print(f"\nTo view: Open {output_path} in your browser")
    print("Controls:")
    print("  - Space: Play/Pause")
    print("  - Left/Right: Previous/Next fight")
    print("  - Slider: Adjust playback speed")


def main():
    parser = argparse.ArgumentParser(description='Create HTML5 training montage')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Training run directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file path')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "training_montage.html"

    generate_html_montage(run_dir, output_path)


if __name__ == "__main__":
    main()