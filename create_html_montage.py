#!/usr/bin/env python3
"""
Create HTML training montage from progressive replays.
This version uses the proven working HTML template directly.
"""

import json
import gzip
from pathlib import Path
import argparse
import sys


def load_replay_data(replay_dir: Path, verbose: bool = False) -> list:
    """Load all replay data from progressive replays directory."""
    replay_files = sorted(replay_dir.glob("*.json.gz"))

    if verbose:
        print(f"Loading replay telemetry from {replay_dir}...")

    replay_data = []
    for replay_file in replay_files:
        try:
            with gzip.open(replay_file, 'rt') as f:
                data = json.load(f)
                # Add metadata if available
                if 'metadata' in data:
                    data['meta'] = data['metadata']
                replay_data.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {replay_file.name}: {e}")
            continue

    if verbose:
        print(f"✅ Loaded {len(replay_data)} replays successfully")

    return replay_data


def generate_html_montage(
    run_dir: Path,
    output_path: Path,
    playback_speed: float = 2.0,
    verbose: bool = False
):
    """Generate HTML montage file from progressive replays."""

    print("\n" + "="*80)
    print("CREATING HTML5 TRAINING MONTAGE")
    print("="*80)

    # Find replay directory
    curriculum_dir = run_dir / "curriculum"
    progressive_replay_dir = curriculum_dir / "progressive_replays"

    if not progressive_replay_dir.exists():
        print(f"❌ No progressive replays found at: {progressive_replay_dir}")
        sys.exit(1)

    # Load all replay data
    replay_files = sorted(progressive_replay_dir.glob("*.json.gz"))
    print(f"✅ Found {len(replay_files)} progressive replays")

    if len(replay_files) == 0:
        print("❌ No replay files found!")
        sys.exit(1)

    # Load replay data
    replay_data = load_replay_data(progressive_replay_dir, verbose)

    if len(replay_data) == 0:
        print("❌ Failed to load any replays!")
        sys.exit(1)

    # Convert replay data to JSON string
    replays_json = json.dumps(replay_data)

    # The complete working HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Montage - Progressive Combat Training</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #0a0e27;
            color: #e0e6ed;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            overflow-x: hidden;
        }

        .header {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #4ecca3;
        }

        .header h1 {
            margin: 0;
            font-size: 32px;
            color: #4ecca3;
            text-shadow: 0 0 10px rgba(78, 204, 163, 0.5);
        }

        .header p {
            margin: 10px 0 0 0;
            color: #b0b8c4;
            font-size: 14px;
        }

        .container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
        }

        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }

        .controls button {
            padding: 10px 20px;
            font-size: 14px;
            background: #16213e;
            color: #4ecca3;
            border: 1px solid #4ecca3;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .controls button:hover {
            background: #4ecca3;
            color: #0a0e27;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(78, 204, 163, 0.3);
        }

        #canvas {
            display: block;
            margin: 0 auto;
            border: 2px solid #4ecca3;
            background: #000;
            box-shadow: 0 4px 20px rgba(78, 204, 163, 0.2);
        }

        .stats {
            margin-top: 20px;
            padding: 15px;
            background: #16213e;
            border-radius: 5px;
            border: 1px solid #4ecca3;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 13px;
        }

        .stat-value {
            color: #4ecca3;
            font-weight: bold;
        }

        #tick-info {
            text-align: center;
            font-size: 18px;
            color: #4ecca3;
            margin: 10px 0;
        }

        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .speed-control input {
            width: 120px;
        }

        .speed-control span {
            min-width: 50px;
            color: #4ecca3;
        }

        .navigation-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            background: #0f3460;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }

        .navigation-controls button {
            padding: 5px 15px;
            font-size: 12px;
            background: #16213e;
            color: #4ecca3;
            border: 1px solid #4ecca3;
            border-radius: 3px;
            cursor: pointer;
        }

        .navigation-controls button:hover {
            background: #4ecca3;
            color: #0a0e27;
        }

        .current-replay-info {
            flex: 1;
            text-align: center;
            color: #4ecca3;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🥊 Training Montage</h1>
        <p>Progressive Training Replays - Watch AI Learn Combat</p>
    </div>

    <div class="container">
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
                <input type="range" id="speedSlider" min="0.25" max="10" step="0.25" value="PLAYBACK_SPEED_VALUE" oninput="updateSpeed(event)">
                <span id="speedValue">PLAYBACK_SPEED_VALUEx</span>
            </div>
        </div>

        <div id="tick-info">Tick 0 / 0</div>

        <canvas id="canvas" width="1200" height="600"></canvas>

        <div class="stats" id="stats"></div>
    </div>

    <script>
        // All replay data
        const ALL_REPLAYS = REPLAYS_JSON_DATA;
        let currentReplayIndex = 0;
        let REPLAY_DATA = null;

        // Variables for rendering
        let ticks = null;
        let config = null;
        let arenaWidth = null;
        let dt = null;

        // Playback state
        let currentTick = 0;
        let isPlaying = false;
        let playbackSpeed = PLAYBACK_SPEED_VALUE;
        let lastFrameTime = 0;

        // Auto-play settings
        let autoPlayEnabled = true;
        let autoPlayDelay = 2000; // 2 seconds between fights

        // Canvas setup
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const scale = 96; // pixels per meter

        function loadReplay(index) {
            currentReplayIndex = index;
            REPLAY_DATA = ALL_REPLAYS[index];

            if (!REPLAY_DATA || !REPLAY_DATA.telemetry) {
                console.error('No replay data available');
                return;
            }

            // Initialize data
            ticks = REPLAY_DATA.telemetry.ticks;
            config = REPLAY_DATA.telemetry.config;
            arenaWidth = config.arena_width;
            dt = config.dt;

            // Reset playback
            currentTick = 0;
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶️ Play';

            // Update UI
            updateReplayInfo();
            updateNavigationButtons();
            render();
        }

        function updateReplayInfo() {
            const meta = REPLAY_DATA.meta || REPLAY_DATA.metadata || {};
            const levelName = meta.level_name || 'Unknown';
            const episode = meta.episode || 0;
            const winRate = ((meta.win_rate || 0) * 100).toFixed(1);
            const winner = meta.winner || 'Unknown';

            document.getElementById('currentReplayInfo').textContent =
                `Level ${meta.level_num || '?'}: ${levelName} | Episode ${episode} | Win Rate: ${winRate}% | Winner: ${winner}`;
        }

        function updateNavigationButtons() {
            document.getElementById('prevReplay').disabled = currentReplayIndex === 0;
            document.getElementById('nextReplay').disabled = currentReplayIndex === ALL_REPLAYS.length - 1;
        }

        function loadPreviousReplay() {
            if (currentReplayIndex > 0) {
                loadReplay(currentReplayIndex - 1);
            }
        }

        function loadNextReplay() {
            if (currentReplayIndex < ALL_REPLAYS.length - 1) {
                loadReplay(currentReplayIndex + 1);
            }
        }

        function togglePlayPause() {
            isPlaying = !isPlaying;
            document.getElementById('playPauseBtn').textContent = isPlaying ? '⏸️ Pause' : '▶️ Play';
        }

        function restart() {
            currentTick = 0;
            isPlaying = false;
            document.getElementById('playPauseBtn').textContent = '▶️ Play';
            render();
        }

        function step() {
            if (currentTick < ticks.length - 1) {
                currentTick++;
                render();
            }
        }

        function toggleAutoPlay() {
            autoPlayEnabled = !autoPlayEnabled;
            const btn = document.getElementById('autoPlayBtn');
            if (autoPlayEnabled) {
                btn.textContent = '🔄 Auto-Play: ON';
                btn.style.background = '#4ecca3';
            } else {
                btn.textContent = '🔄 Auto-Play: OFF';
                btn.style.background = '#888';
            }
        }

        function updateSpeed(e) {
            playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = playbackSpeed.toFixed(2) + 'x';
        }

        function render() {
            if (!ticks || currentTick >= ticks.length) return;

            const tick = ticks[currentTick];

            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw arena floor
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();

            // Draw walls
            ctx.strokeStyle = '#4ecca3';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2 - 50);
            ctx.lineTo(0, canvas.height / 2 + 50);
            ctx.moveTo(canvas.width, canvas.height / 2 - 50);
            ctx.lineTo(canvas.width, canvas.height / 2 + 50);
            ctx.stroke();

            // Draw fighters
            const fighter_a = tick.fighter_a;
            const fighter_b = tick.fighter_b;

            // Fighter A (Learning AI)
            const aX = fighter_a.position * scale;
            const aY = canvas.height / 2;

            ctx.fillStyle = '#ff6b6b';
            ctx.fillRect(aX - 15, aY - 25, 30, 50);

            // Fighter A health bar
            ctx.fillStyle = '#ff6b6b';
            ctx.fillRect(aX - 20, aY - 40, 40 * (fighter_a.hp / fighter_a.max_hp), 5);
            ctx.strokeStyle = '#fff';
            ctx.strokeRect(aX - 20, aY - 40, 40, 5);

            // Fighter B (Opponent)
            const bX = fighter_b.position * scale;
            const bY = canvas.height / 2;

            ctx.fillStyle = '#4ecca3';
            ctx.fillRect(bX - 15, bY - 25, 30, 50);

            // Fighter B health bar
            ctx.fillStyle = '#4ecca3';
            ctx.fillRect(bX - 20, bY - 40, 40 * (fighter_b.hp / fighter_b.max_hp), 5);
            ctx.strokeStyle = '#fff';
            ctx.strokeRect(bX - 20, bY - 40, 40, 5);

            // Draw collision indicator
            if (tick.events && tick.events.some(e => e.type === 'collision')) {
                ctx.strokeStyle = '#ffd93d';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc((aX + bX) / 2, canvas.height / 2, 30, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Update tick info
            document.getElementById('tick-info').textContent = `Tick ${currentTick} / ${ticks.length - 1}`;

            // Update stats
            updateStats(tick);
        }

        function updateStats(tick) {
            const stats = document.getElementById('stats');
            const fighter_a = tick.fighter_a;
            const fighter_b = tick.fighter_b;

            stats.innerHTML = `
                <div class="stat-item">
                    <span>AI Fighter HP:</span>
                    <span class="stat-value">${fighter_a.hp.toFixed(1)} / ${fighter_a.max_hp}</span>
                </div>
                <div class="stat-item">
                    <span>AI Fighter Stamina:</span>
                    <span class="stat-value">${fighter_a.stamina.toFixed(1)} / ${fighter_a.max_stamina}</span>
                </div>
                <div class="stat-item">
                    <span>AI Fighter Stance:</span>
                    <span class="stat-value">${fighter_a.stance}</span>
                </div>
                <div class="stat-item">
                    <span>Opponent HP:</span>
                    <span class="stat-value">${fighter_b.hp.toFixed(1)} / ${fighter_b.max_hp}</span>
                </div>
                <div class="stat-item">
                    <span>Opponent Stamina:</span>
                    <span class="stat-value">${fighter_b.stamina.toFixed(1)} / ${fighter_b.max_stamina}</span>
                </div>
                <div class="stat-item">
                    <span>Opponent Stance:</span>
                    <span class="stat-value">${fighter_b.stance}</span>
                </div>
                <div class="stat-item">
                    <span>Distance:</span>
                    <span class="stat-value">${Math.abs(fighter_b.position - fighter_a.position).toFixed(2)}m</span>
                </div>
            `;
        }

        // Animation loop
        function animate(timestamp) {
            if (!lastFrameTime) lastFrameTime = timestamp;
            const deltaTime = timestamp - lastFrameTime;

            if (isPlaying && ticks) {
                const msPerTick = (dt * 1000) / playbackSpeed;

                if (deltaTime >= msPerTick) {
                    if (currentTick < ticks.length - 1) {
                        currentTick++;
                        render();
                        lastFrameTime = timestamp;
                    } else {
                        // Fight ended
                        isPlaying = false;
                        document.getElementById('playPauseBtn').textContent = '▶️ Play';
                        console.log(`Fight ended - Episode ${REPLAY_DATA.meta?.episode || '?'}`);

                        // Auto-play next fight if enabled
                        if (autoPlayEnabled && currentReplayIndex < ALL_REPLAYS.length - 1) {
                            console.log('Auto-playing next fight in ' + autoPlayDelay + 'ms...');
                            setTimeout(() => {
                                loadNextReplay();
                                // Auto-start playback
                                setTimeout(() => {
                                    isPlaying = true;
                                    document.getElementById('playPauseBtn').textContent = '⏸️ Pause';
                                }, 500);
                            }, autoPlayDelay);
                        } else if (autoPlayEnabled && currentReplayIndex >= ALL_REPLAYS.length - 1) {
                            console.log('Reached last replay. Auto-play complete.');
                        }
                    }
                }
            }

            requestAnimationFrame(animate);
        }

        // Initialize
        if (ALL_REPLAYS.length > 0) {
            loadReplay(0);
            // Auto-start if auto-play is enabled
            if (autoPlayEnabled) {
                setTimeout(() => {
                    isPlaying = true;
                    document.getElementById('playPauseBtn').textContent = '⏸️ Pause';
                }, 1000);
            }
        } else {
            document.getElementById('currentReplayInfo').textContent = 'No replays available';
        }

        // Start animation loop
        requestAnimationFrame(animate);
    </script>
</body>
</html>"""

    # Replace placeholders
    html = html_template.replace('REPLAYS_JSON_DATA', replays_json)
    html = html.replace('PLAYBACK_SPEED_VALUE', str(playback_speed))

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    # Print summary
    print("\n" + "="*80)
    print("✅ HTML MONTAGE COMPLETE!")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Replays: {len(replay_data)}")
    print(f"\nTo view: Open {output_path} in your browser")
    print("Controls:")
    print("  - Space/Click: Play/Pause")
    print("  - Left/Right: Previous/Next fight")
    print("  - Speed Slider: 0.25x to 10x speed")
    print("  - Auto-Play: Automatically advances through all fights")


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
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate inputs
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