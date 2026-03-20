#!/usr/bin/env python3
"""
Create HTML5 Montage from Existing Spectacle-Based Replays

Works with the existing replay structure until new progressive replays are available.
"""

import json
import gzip
from pathlib import Path
import argparse
import sys


def create_html_from_existing_replays(run_dir: Path, output_path: Path):
    """Create HTML montage from existing spectacle-based replays."""

    print("\n" + "="*80)
    print("CREATING HTML5 MONTAGE FROM EXISTING REPLAYS")
    print("="*80)

    # Find replay index
    replay_index_path = run_dir / "curriculum" / "replay_index.json"
    if not replay_index_path.exists():
        print(f"❌ No replay index found at {replay_index_path}")
        sys.exit(1)

    with open(replay_index_path) as f:
        index_data = json.load(f)

    replays = index_data['replays']
    print(f"✅ Found {len(replays)} replays")

    # Group by level
    levels = {}
    for replay in replays:
        # Extract level number from stage
        stage = replay['stage']
        if 'level_1' in stage:
            level_num = 1
            level_name = "Fundamentals"
        elif 'level_2' in stage:
            level_num = 2
            level_name = "Basic Skills"
        elif 'level_3' in stage:
            level_num = 3
            level_name = "Intermediate"
        elif 'level_4' in stage:
            level_num = 4
            level_name = "Advanced"
        elif 'level_5' in stage:
            level_num = 5
            level_name = "Expert"
        else:
            continue

        if level_num not in levels:
            levels[level_num] = []
        levels[level_num].append({
            **replay,
            'level_num': level_num,
            'level_name': level_name
        })

    # Load telemetry data
    replays_dir = run_dir / "curriculum" / "replays"
    all_replay_data = []

    for level_num in sorted(levels.keys()):
        level_replays = sorted(levels[level_num], key=lambda r: r['spectacle_score'])

        for i, replay in enumerate(level_replays):
            # Find the replay file
            pattern = f"{replay['stage']}_{replay['spectacle_rank']}_*.json.gz"
            replay_files = list(replays_dir.glob(pattern))

            if replay_files:
                replay_path = replay_files[0]
                print(f"  Loading {replay_path.name}...")

                try:
                    with gzip.open(replay_path, 'rt') as f:
                        data = json.load(f)

                    # Simulate progressive metadata
                    # Estimate episode based on spectacle rank
                    if replay['spectacle_rank'] == 'bottom':
                        episode = 100  # Early learning
                        win_rate = 0.3
                        notes = "Early learning - struggling"
                    elif replay['spectacle_rank'] == 'middle':
                        episode = 500  # Mid training
                        win_rate = 0.6
                        notes = "Skill development - competitive"
                    else:  # top
                        episode = 900  # Late training
                        win_rate = 0.85
                        notes = "Mastery - dominant"

                    # Add simulated progressive metadata
                    data['meta'] = {
                        'level_num': level_num,
                        'level_name': level_name,
                        'episode': episode,
                        'total_episodes': 1000,
                        'win_rate': win_rate,
                        'recent_rewards': [],
                        'fighter_a': replay['fighter_a'],
                        'fighter_b': replay['fighter_b'],
                        'winner': replay['winner'],
                        'notes': notes
                    }

                    all_replay_data.append(data)
                except Exception as e:
                    print(f"  ⚠️  Error loading {replay_path.name}: {e}")

    print(f"\n✅ Loaded {len(all_replay_data)} replays")

    # Generate HTML
    html_content = generate_simple_html(all_replay_data)

    # Save
    with open(output_path, 'w') as f:
        f.write(html_content)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✅ HTML montage created: {output_path} ({size_mb:.1f} MB)")
    print(f"Open in browser to view the training progression!")


def generate_simple_html(replay_data):
    """Generate simplified HTML for existing replays."""

    replays_json = json.dumps(replay_data)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Montage</title>
    <style>
        body {{ font-family: monospace; background: #000; color: #0f0; margin: 0; }}
        #container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; }}
        #arena {{
            width: 100%;
            height: 300px;
            background: #111;
            border: 2px solid #0f0;
            position: relative;
            margin: 20px 0;
        }}
        .fighter {{
            position: absolute;
            bottom: 0;
            width: 30px;
            height: 80px;
        }}
        #fighter-a {{ background: #0f0; }}
        #fighter-b {{ background: #f00; }}
        #controls {{ text-align: center; margin: 20px 0; }}
        button {{
            padding: 10px 20px;
            margin: 0 10px;
            background: #111;
            color: #0f0;
            border: 1px solid #0f0;
            cursor: pointer;
        }}
        button:hover {{ background: #0f0; color: #000; }}
        #info {{
            background: #111;
            padding: 15px;
            border: 1px solid #0f0;
            margin: 20px 0;
        }}
        .stat {{ display: inline-block; margin-right: 30px; }}
    </style>
</head>
<body>
    <div id="container">
        <h1>AI Combat Training Montage</h1>

        <div id="info">
            <div class="stat">Level: <span id="level">1</span></div>
            <div class="stat">Episode: <span id="episode">1</span></div>
            <div class="stat">Win Rate: <span id="winrate">0%</span></div>
            <div class="stat">Fight: <span id="current">1</span> / <span id="total">1</span></div>
        </div>

        <div id="arena">
            <div id="fighter-a" class="fighter"></div>
            <div id="fighter-b" class="fighter"></div>
        </div>

        <div id="controls">
            <button onclick="previousFight()">◀ Previous</button>
            <button onclick="togglePlay()" id="play-btn">▶ Play</button>
            <button onclick="nextFight()">Next ▶</button>
            <button onclick="changeSpeed()">Speed: <span id="speed">2x</span></button>
        </div>
    </div>

    <script>
        const replays = {replays_json};
        let currentReplay = 0;
        let currentTick = 0;
        let playing = false;
        let speed = 2;
        let interval = null;

        const ARENA_WIDTH = 1200;
        const SCALE = ARENA_WIDTH / 12.476;

        function update() {{
            if (!replays[currentReplay]) return;

            const replay = replays[currentReplay];
            const meta = replay.meta || {{}};
            const ticks = replay.telemetry?.ticks || [];

            if (currentTick >= ticks.length) {{
                if (playing && currentReplay < replays.length - 1) {{
                    nextFight();
                }} else {{
                    stop();
                }}
                return;
            }}

            const tick = ticks[currentTick];
            if (tick) {{
                document.getElementById('fighter-a').style.left = (tick.fighter_a.position * SCALE) + 'px';
                document.getElementById('fighter-b').style.left = (tick.fighter_b.position * SCALE) + 'px';
            }}

            document.getElementById('level').textContent = meta.level_num + ': ' + meta.level_name;
            document.getElementById('episode').textContent = meta.episode || '?';
            document.getElementById('winrate').textContent = ((meta.win_rate || 0) * 100).toFixed(0) + '%';
            document.getElementById('current').textContent = currentReplay + 1;
            document.getElementById('total').textContent = replays.length;

            if (playing) currentTick++;
        }}

        function togglePlay() {{
            playing = !playing;
            document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';

            if (playing) {{
                interval = setInterval(update, 1000 / (12 * speed));
            }} else {{
                stop();
            }}
        }}

        function stop() {{
            playing = false;
            document.getElementById('play-btn').textContent = '▶ Play';
            if (interval) clearInterval(interval);
        }}

        function nextFight() {{
            if (currentReplay < replays.length - 1) {{
                currentReplay++;
                currentTick = 0;
                update();
            }}
        }}

        function previousFight() {{
            if (currentReplay > 0) {{
                currentReplay--;
                currentTick = 0;
                update();
            }}
        }}

        function changeSpeed() {{
            speed = speed === 1 ? 2 : speed === 2 ? 4 : speed === 4 ? 8 : 1;
            document.getElementById('speed').textContent = speed + 'x';
            if (playing) {{
                stop();
                togglePlay();
            }}
        }}

        // Keyboard controls
        document.addEventListener('keydown', e => {{
            if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
            else if (e.key === 'ArrowLeft') previousFight();
            else if (e.key === 'ArrowRight') nextFight();
        }});

        update();
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Create HTML montage from existing replays")
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--output", help="Output HTML path")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "montage.html"

    create_html_from_existing_replays(run_dir, output_path)


if __name__ == "__main__":
    main()