"""
Atom Combat - HTML5 Canvas Renderer

Generates standalone HTML files with animated match replays.
"""

import json
from pathlib import Path
from typing import Dict, Any


class HtmlRenderer:
    """
    Generates HTML5 canvas animations from match telemetry.

    Creates standalone .html files that can be:
    - Opened in any browser
    - Shared easily
    - Embedded in web pages
    """

    def __init__(self):
        self.template = self._get_html_template()

    def generate_replay_html(
        self,
        telemetry: Dict[str, Any],
        match_result: Any,
        output_path: str,
        spectacle_score: Any = None,
        playback_speed: float = 1.0
    ):
        """
        Generate standalone HTML file with animated replay.

        Args:
            telemetry: Match telemetry data
            match_result: MatchResult object
            output_path: Where to save .html file
            spectacle_score: Optional SpectacleScore for display
            playback_speed: Animation speed multiplier (1.0 = realtime)
        """
        # Prepare data for JavaScript
        replay_data = {
            "telemetry": telemetry,
            "result": {
                "winner": match_result.winner,
                "total_ticks": match_result.total_ticks,
                "final_hp_a": match_result.final_hp_a,
                "final_hp_b": match_result.final_hp_b
            },
            "events": match_result.events,
            "spectacle_score": spectacle_score.to_dict() if spectacle_score else None,
            "playback_speed": playback_speed
        }

        # Generate HTML
        html_content = self.template.replace(
            "// REPLAY_DATA_PLACEHOLDER",
            f"const REPLAY_DATA = {json.dumps(replay_data, indent=2)};"
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(html_content)

        return output_file

    def _get_html_template(self) -> str:
        """Get the HTML5 canvas template."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atom Combat - Replay Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        h1 {
            color: #ff6b6b;
            margin-bottom: 10px;
            text-align: center;
        }

        .container {
            max-width: 1200px;
            width: 100%;
        }

        #canvas {
            background: #16213e;
            border: 3px solid #0f3460;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }

        .controls {
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        button {
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
        }

        button:hover {
            background: #ff5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .stat-box {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff6b6b;
        }

        .stat-box h3 {
            color: #ff6b6b;
            margin-bottom: 10px;
            font-size: 14px;
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
            width: 100px;
        }

        .speed-control span {
            min-width: 40px;
            color: #4ecca3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚔️ ATOM COMBAT - REPLAY VIEWER ⚔️</h1>

        <div class="controls">
            <button id="playPauseBtn">▶️ Play</button>
            <button id="restartBtn">🔄 Restart</button>
            <button id="stepBtn">⏭️ Step</button>
            <div class="speed-control">
                <label>Speed:</label>
                <input type="range" id="speedSlider" min="0.25" max="5" step="0.25" value="1">
                <span id="speedValue">1.0x</span>
            </div>
        </div>

        <div id="tick-info">Tick 0 / 0</div>

        <canvas id="canvas" width="1200" height="600"></canvas>

        <div class="stats" id="stats"></div>
    </div>

    <script>
        // REPLAY_DATA_PLACEHOLDER

        // Canvas setup
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // State
        let currentTick = 0;
        let isPlaying = false;
        let playbackSpeed = REPLAY_DATA.playback_speed || 1.0;
        let lastFrameTime = 0;

        const ticks = REPLAY_DATA.telemetry.ticks;
        const config = REPLAY_DATA.telemetry.config;
        const arenaWidth = config.arena_width;
        const dt = config.dt;

        // Colors
        const COLORS = {
            background: '#16213e',
            arena: '#0f3460',
            fighterA: '#ff6b6b',
            fighterB: '#4ecca3',
            hp: '#ff6b6b',
            stamina: '#4ecca3',
            collision: '#ffd93d',
            text: '#eeeeee',
            textDim: '#aaaaaa'
        };

        // Stance visual styles
        const STANCE_STYLES = {
            neutral: { shape: 'circle', size: 1.0 },
            extended: { shape: 'triangle-right', size: 1.2 },
            retracted: { shape: 'square', size: 0.8 },
            defending: { shape: 'hexagon', size: 1.1 }
        };

        // Controls
        document.getElementById('playPauseBtn').addEventListener('click', togglePlayPause);
        document.getElementById('restartBtn').addEventListener('click', restart);
        document.getElementById('stepBtn').addEventListener('click', step);
        document.getElementById('speedSlider').addEventListener('input', updateSpeed);

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

        function updateSpeed(e) {
            playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = playbackSpeed.toFixed(2) + 'x';
        }

        // Animation loop
        function animate(timestamp) {
            if (isPlaying && currentTick < ticks.length - 1) {
                const elapsed = timestamp - lastFrameTime;
                const frameTime = (dt * 1000) / playbackSpeed; // Convert to ms

                if (elapsed >= frameTime) {
                    currentTick++;
                    lastFrameTime = timestamp;
                    render();
                }
            } else if (currentTick >= ticks.length - 1) {
                isPlaying = false;
                document.getElementById('playPauseBtn').textContent = '▶️ Play';
            }

            requestAnimationFrame(animate);
        }

        // Rendering
        function render() {
            const tick = ticks[currentTick];

            // Clear canvas
            ctx.fillStyle = COLORS.background;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Check if match is over
            if (currentTick === ticks.length - 1) {
                drawMatchEndScreen(tick);
            } else {
                // Draw arena
                drawArena();

                // Draw fighters
                drawFighter(tick.fighter_a, COLORS.fighterA, 'left');
                drawFighter(tick.fighter_b, COLORS.fighterB, 'right');

                // Draw stats overlay
                drawStatsOverlay(tick);
            }

            // Update tick info
            document.getElementById('tick-info').textContent =
                `Tick ${tick.tick} / ${ticks.length - 1} (${(tick.tick * dt).toFixed(2)}s)`;

            // Update stats panel
            updateStatsPanel(tick);
        }

        function drawArena() {
            const arenaY = canvas.height * 0.5;
            const arenaVisualWidth = canvas.width * 0.8;
            const arenaX = canvas.width * 0.1;

            // Arena line
            ctx.strokeStyle = COLORS.arena;
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(arenaX, arenaY);
            ctx.lineTo(arenaX + arenaVisualWidth, arenaY);
            ctx.stroke();

            // Arena bounds
            ctx.fillStyle = COLORS.textDim;
            ctx.font = '12px Courier New';
            ctx.fillText('0m', arenaX - 15, arenaY + 20);
            ctx.fillText(arenaWidth.toFixed(1) + 'm', arenaX + arenaVisualWidth - 20, arenaY + 20);
        }

        function drawFighter(fighter, color, side) {
            const arenaVisualWidth = canvas.width * 0.8;
            const arenaX = canvas.width * 0.1;
            const arenaY = canvas.height * 0.5;

            // Map position to canvas
            const x = arenaX + (fighter.position / arenaWidth) * arenaVisualWidth;
            const y = arenaY;

            // Get stance style
            const stanceStyle = STANCE_STYLES[fighter.stance] || STANCE_STYLES.neutral;
            const baseSize = 25 * stanceStyle.size;

            // Draw fighter shape
            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;

            switch (stanceStyle.shape) {
                case 'circle':
                    ctx.beginPath();
                    ctx.arc(x, y, baseSize, 0, Math.PI * 2);
                    ctx.fill();
                    break;
                case 'triangle-right':
                    ctx.beginPath();
                    ctx.moveTo(x + baseSize, y);
                    ctx.lineTo(x - baseSize/2, y - baseSize);
                    ctx.lineTo(x - baseSize/2, y + baseSize);
                    ctx.closePath();
                    ctx.fill();
                    break;
                case 'square':
                    ctx.fillRect(x - baseSize, y - baseSize, baseSize * 2, baseSize * 2);
                    break;
                case 'hexagon':
                    drawHexagon(x, y, baseSize);
                    ctx.fill();
                    break;
            }

            // Draw velocity indicator
            if (Math.abs(fighter.velocity) > 0.1) {
                const velLength = fighter.velocity * 30;
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.globalAlpha = 0.7;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + velLength, y);
                ctx.stroke();
                ctx.globalAlpha = 1.0;

                // Arrowhead
                const arrowSize = 8;
                const dir = Math.sign(velLength);
                ctx.beginPath();
                ctx.moveTo(x + velLength, y);
                ctx.lineTo(x + velLength - dir * arrowSize, y - arrowSize/2);
                ctx.lineTo(x + velLength - dir * arrowSize, y + arrowSize/2);
                ctx.closePath();
                ctx.fill();
            }

            // Draw HP/Stamina bars above fighter
            const barWidth = 80;
            const barHeight = 8;
            const barY = side === 'left' ? y - 80 : y - 80;

            // HP bar
            drawBar(x - barWidth/2, barY, barWidth, barHeight,
                    fighter.hp / fighter.max_hp, COLORS.hp);

            // Stamina bar
            drawBar(x - barWidth/2, barY + barHeight + 4, barWidth, barHeight,
                    fighter.stamina / fighter.max_stamina, COLORS.stamina);

            // Name and stance
            ctx.fillStyle = COLORS.text;
            ctx.font = 'bold 14px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText(fighter.name, x, barY - 10);

            ctx.font = '11px Courier New';
            ctx.fillStyle = COLORS.textDim;
            ctx.fillText(fighter.stance, x, barY + barHeight * 2 + 20);
        }

        function drawBar(x, y, width, height, percentage, color) {
            // Background
            ctx.fillStyle = '#333';
            ctx.fillRect(x, y, width, height);

            // Fill
            ctx.fillStyle = color;
            ctx.fillRect(x, y, width * percentage, height);

            // Border
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, width, height);
        }

        function drawHexagon(x, y, size) {
            ctx.beginPath();
            for (let i = 0; i < 6; i++) {
                const angle = (Math.PI / 3) * i;
                const hx = x + size * Math.cos(angle);
                const hy = y + size * Math.sin(angle);
                if (i === 0) ctx.moveTo(hx, hy);
                else ctx.lineTo(hx, hy);
            }
            ctx.closePath();
        }

        function drawStatsOverlay(tick) {
            // Check for collision
            const hasCollision = tick.events && tick.events.some(e => e.type === 'COLLISION');

            if (hasCollision) {
                ctx.fillStyle = COLORS.collision;
                ctx.font = 'bold 36px Courier New';
                ctx.textAlign = 'center';
                ctx.fillText('💥 COLLISION!', canvas.width / 2, 60);
            }
        }

        function drawMatchEndScreen(finalTick) {
            const result = REPLAY_DATA.result;
            const spectacle = REPLAY_DATA.spectacle_score;
            const collisions = REPLAY_DATA.events.filter(e => e.type === 'COLLISION').length;

            // Determine winner and loser
            const fighter_a = finalTick.fighter_a;
            const fighter_b = finalTick.fighter_b;
            let winner, loser, winnerColor, loserColor;

            if (result.winner.includes(fighter_a.name)) {
                winner = fighter_a;
                loser = fighter_b;
                winnerColor = COLORS.fighterA;
                loserColor = COLORS.fighterB;
            } else {
                winner = fighter_b;
                loser = fighter_a;
                winnerColor = COLORS.fighterB;
                loserColor = COLORS.fighterA;
            }

            // Background gradient
            const gradient = ctx.createRadialGradient(
                canvas.width / 2, canvas.height / 2, 100,
                canvas.width / 2, canvas.height / 2, 600
            );
            gradient.addColorStop(0, '#1a1a2e');
            gradient.addColorStop(1, '#0f0f1e');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Winner banner
            ctx.fillStyle = winnerColor;
            ctx.globalAlpha = 0.2;
            ctx.fillRect(0, 80, canvas.width, 120);
            ctx.globalAlpha = 1.0;

            // "MATCH COMPLETE" header
            ctx.fillStyle = COLORS.text;
            ctx.font = 'bold 28px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('⚔️ MATCH COMPLETE ⚔️', canvas.width / 2, 50);

            // Winner announcement
            ctx.fillStyle = winnerColor;
            ctx.font = 'bold 56px Courier New';
            ctx.fillText(`🏆 ${winner.name.toUpperCase()} WINS!`, canvas.width / 2, 140);

            // Subtitle
            ctx.fillStyle = COLORS.textDim;
            ctx.font = '20px Courier New';
            ctx.fillText(result.winner, canvas.width / 2, 175);

            // Fighter comparison box
            const boxY = 220;
            const boxHeight = 140;

            // Winner box (left)
            ctx.fillStyle = 'rgba(255, 107, 107, 0.1)';
            ctx.fillRect(150, boxY, 400, boxHeight);
            ctx.strokeStyle = winnerColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(150, boxY, 400, boxHeight);

            ctx.fillStyle = winnerColor;
            ctx.font = 'bold 24px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('WINNER', 350, boxY + 30);

            ctx.font = '18px Courier New';
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = 'left';
            ctx.fillText(`Fighter: ${winner.name}`, 170, boxY + 60);
            ctx.fillText(`Mass: ${winner.mass}kg`, 170, boxY + 85);
            ctx.fillText(`Final HP: ${winner.hp.toFixed(1)} / ${winner.max_hp.toFixed(1)}`, 170, boxY + 110);

            const winnerHpPct = (winner.hp / winner.max_hp * 100).toFixed(0);
            ctx.fillStyle = winnerColor;
            ctx.font = 'bold 20px Courier New';
            ctx.fillText(`${winnerHpPct}%`, 340, boxY + 110);

            // Loser box (right)
            ctx.fillStyle = 'rgba(78, 204, 163, 0.05)';
            ctx.fillRect(650, boxY, 400, boxHeight);
            ctx.strokeStyle = loserColor;
            ctx.lineWidth = 2;
            ctx.strokeRect(650, boxY, 400, boxHeight);

            ctx.fillStyle = loserColor;
            ctx.font = 'bold 24px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('DEFEATED', 850, boxY + 30);

            ctx.font = '18px Courier New';
            ctx.fillStyle = COLORS.textDim;
            ctx.textAlign = 'left';
            ctx.fillText(`Fighter: ${loser.name}`, 670, boxY + 60);
            ctx.fillText(`Mass: ${loser.mass}kg`, 670, boxY + 85);
            ctx.fillText(`Final HP: ${loser.hp.toFixed(1)} / ${loser.max_hp.toFixed(1)}`, 670, boxY + 110);

            // Match Stats section
            const statsY = 390;
            ctx.fillStyle = COLORS.text;
            ctx.font = 'bold 22px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('📊 MATCH STATISTICS', canvas.width / 2, statsY);

            // Stats grid
            const statBoxWidth = 180;
            const statBoxHeight = 100;
            const statY = statsY + 30;
            const statSpacing = 20;

            const stats = [
                { label: 'Duration', value: `${result.total_ticks} ticks`, sub: `${(result.total_ticks * dt).toFixed(1)}s` },
                { label: 'Collisions', value: collisions.toString(), sub: `${(collisions / result.total_ticks * 100).toFixed(1)}% of ticks` },
                { label: 'Damage Dealt', value: `${(loser.max_hp - loser.hp).toFixed(1)}`, sub: 'to loser' },
                { label: 'Damage Taken', value: `${(winner.max_hp - winner.hp).toFixed(1)}`, sub: 'by winner' }
            ];

            const totalWidth = stats.length * statBoxWidth + (stats.length - 1) * statSpacing;
            const startX = (canvas.width - totalWidth) / 2;

            stats.forEach((stat, i) => {
                const x = startX + i * (statBoxWidth + statSpacing);

                // Box
                ctx.fillStyle = 'rgba(15, 52, 96, 0.6)';
                ctx.fillRect(x, statY, statBoxWidth, statBoxHeight);
                ctx.strokeStyle = '#0f3460';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, statY, statBoxWidth, statBoxHeight);

                // Label
                ctx.fillStyle = COLORS.textDim;
                ctx.font = '14px Courier New';
                ctx.textAlign = 'center';
                ctx.fillText(stat.label, x + statBoxWidth / 2, statY + 25);

                // Value
                ctx.fillStyle = COLORS.collision;
                ctx.font = 'bold 28px Courier New';
                ctx.fillText(stat.value, x + statBoxWidth / 2, statY + 60);

                // Sub
                ctx.fillStyle = COLORS.textDim;
                ctx.font = '11px Courier New';
                ctx.fillText(stat.sub, x + statBoxWidth / 2, statY + 82);
            });

            // Spectacle Score
            if (spectacle) {
                const spectacleY = 520;

                ctx.fillStyle = COLORS.text;
                ctx.font = 'bold 22px Courier New';
                ctx.textAlign = 'center';
                ctx.fillText('⭐ SPECTACLE SCORE', canvas.width / 2, spectacleY);

                // Overall score
                ctx.font = 'bold 48px Courier New';
                const scoreColor = spectacle.overall >= 0.8 ? '#4ecca3' :
                                   spectacle.overall >= 0.6 ? '#ffd93d' :
                                   spectacle.overall >= 0.4 ? '#ff9a3c' : '#ff6b6b';
                ctx.fillStyle = scoreColor;
                ctx.fillText(spectacle.overall.toFixed(3), canvas.width / 2, spectacleY + 50);

                // Star rating
                const stars = spectacle.overall >= 0.8 ? '⭐⭐⭐⭐⭐' :
                              spectacle.overall >= 0.6 ? '⭐⭐⭐⭐' :
                              spectacle.overall >= 0.4 ? '⭐⭐⭐' : '⭐⭐';
                ctx.font = '24px Courier New';
                ctx.fillStyle = COLORS.collision;
                ctx.fillText(stars, canvas.width / 2, spectacleY + 85);

                // Top metrics
                const topMetrics = [
                    { name: 'Close Finish', value: spectacle.close_finish },
                    { name: 'Collision Drama', value: spectacle.collision_drama },
                    { name: 'Pacing Variety', value: spectacle.pacing_variety }
                ].sort((a, b) => b.value - a.value);

                ctx.font = '14px Courier New';
                ctx.fillStyle = COLORS.textDim;
                ctx.textAlign = 'center';
                topMetrics.forEach((metric, i) => {
                    const barWidth = 200;
                    const barX = canvas.width / 2 - barWidth / 2;
                    const barY = spectacleY + 110 + i * 25;

                    // Metric name
                    ctx.textAlign = 'right';
                    ctx.fillText(metric.name + ':', barX - 10, barY + 12);

                    // Bar background
                    ctx.fillStyle = '#333';
                    ctx.fillRect(barX, barY, barWidth, 15);

                    // Bar fill
                    const barColor = metric.value >= 0.8 ? '#4ecca3' :
                                    metric.value >= 0.5 ? '#ffd93d' : '#ff6b6b';
                    ctx.fillStyle = barColor;
                    ctx.fillRect(barX, barY, barWidth * metric.value, 15);

                    // Value text
                    ctx.fillStyle = COLORS.text;
                    ctx.textAlign = 'left';
                    ctx.font = 'bold 12px Courier New';
                    ctx.fillText(metric.value.toFixed(2), barX + barWidth + 10, barY + 12);

                    ctx.font = '14px Courier New';
                    ctx.fillStyle = COLORS.textDim;
                });
            }
        }

        function updateStatsPanel(tick) {
            const statsDiv = document.getElementById('stats');
            const fighter_a = tick.fighter_a;
            const fighter_b = tick.fighter_b;

            statsDiv.innerHTML = `
                <div class="stat-box">
                    <h3>${fighter_a.name} (${fighter_a.mass}kg)</h3>
                    <div class="stat-item">
                        <span>HP:</span>
                        <span class="stat-value">${fighter_a.hp.toFixed(1)} / ${fighter_a.max_hp.toFixed(1)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Stamina:</span>
                        <span class="stat-value">${fighter_a.stamina.toFixed(1)} / ${fighter_a.max_stamina.toFixed(1)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Velocity:</span>
                        <span class="stat-value">${fighter_a.velocity.toFixed(2)} m/s</span>
                    </div>
                    <div class="stat-item">
                        <span>Position:</span>
                        <span class="stat-value">${fighter_a.position.toFixed(2)}m</span>
                    </div>
                </div>

                <div class="stat-box">
                    <h3>Match Info</h3>
                    <div class="stat-item">
                        <span>Winner:</span>
                        <span class="stat-value">${REPLAY_DATA.result.winner}</span>
                    </div>
                    <div class="stat-item">
                        <span>Duration:</span>
                        <span class="stat-value">${REPLAY_DATA.result.total_ticks} ticks</span>
                    </div>
                    <div class="stat-item">
                        <span>Collisions:</span>
                        <span class="stat-value">${REPLAY_DATA.events.filter(e => e.type === 'COLLISION').length}</span>
                    </div>
                    ${REPLAY_DATA.spectacle_score ? `
                    <div class="stat-item">
                        <span>Spectacle:</span>
                        <span class="stat-value">${REPLAY_DATA.spectacle_score.overall.toFixed(3)}</span>
                    </div>
                    ` : ''}
                </div>

                <div class="stat-box">
                    <h3>${fighter_b.name} (${fighter_b.mass}kg)</h3>
                    <div class="stat-item">
                        <span>HP:</span>
                        <span class="stat-value">${fighter_b.hp.toFixed(1)} / ${fighter_b.max_hp.toFixed(1)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Stamina:</span>
                        <span class="stat-value">${fighter_b.stamina.toFixed(1)} / ${fighter_b.max_stamina.toFixed(1)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Velocity:</span>
                        <span class="stat-value">${fighter_b.velocity.toFixed(2)} m/s</span>
                    </div>
                    <div class="stat-item">
                        <span>Position:</span>
                        <span class="stat-value">${fighter_b.position.toFixed(2)}m</span>
                    </div>
                </div>
            `;
        }

        // Initialize
        render();
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""
