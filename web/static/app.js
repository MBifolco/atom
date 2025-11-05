// Atom Combat Web Application

class AtomCombatApp {
    constructor() {
        this.fighters = [];
        this.selectedFighterA = null;
        this.selectedFighterB = null;
        this.matchData = null;

        this.init();
    }

    async init() {
        // Load fighters from registry
        await this.loadFighters();

        // Setup event listeners
        this.setupEventListeners();

        // Show selection view
        this.showView('selection-view');
    }

    async loadFighters() {
        try {
            const response = await fetch('/api/fighters');
            if (!response.ok) {
                throw new Error('Failed to load fighters');
            }

            this.fighters = await response.json();
            this.populateFighterSelects();
        } catch (error) {
            console.error('Error loading fighters:', error);
            alert('Failed to load fighters. Please refresh the page.');
        }
    }

    populateFighterSelects() {
        const selectA = document.getElementById('fighter-a-select');
        const selectB = document.getElementById('fighter-b-select');

        // Clear existing options
        selectA.innerHTML = '<option value="">-- Select Fighter A --</option>';
        selectB.innerHTML = '<option value="">-- Select Fighter B --</option>';

        // Add fighters
        this.fighters.forEach(fighter => {
            const optionA = document.createElement('option');
            optionA.value = fighter.id;
            optionA.textContent = `${fighter.name} (${fighter.type})`;
            selectA.appendChild(optionA);

            const optionB = optionA.cloneNode(true);
            selectB.appendChild(optionB);
        });
    }

    setupEventListeners() {
        // Fighter selection
        document.getElementById('fighter-a-select').addEventListener('change', (e) => {
            this.onFighterSelected('a', e.target.value);
        });

        document.getElementById('fighter-b-select').addEventListener('change', (e) => {
            this.onFighterSelected('b', e.target.value);
        });

        // Mass inputs
        document.getElementById('mass-a').addEventListener('change', () => {
            this.checkStartButtonState();
        });

        document.getElementById('mass-b').addEventListener('change', () => {
            this.checkStartButtonState();
        });

        // Start match button
        document.getElementById('start-match-btn').addEventListener('click', () => {
            this.startMatch();
        });

        // New match button
        document.getElementById('new-match-btn').addEventListener('click', () => {
            this.showView('selection-view');
        });

        // Export replay button
        document.getElementById('export-replay-btn').addEventListener('click', () => {
            this.exportReplay();
        });
    }

    onFighterSelected(side, fighterId) {
        if (!fighterId) {
            if (side === 'a') {
                this.selectedFighterA = null;
            } else {
                this.selectedFighterB = null;
            }
            this.clearFighterInfo(side);
            this.checkStartButtonState();
            return;
        }

        const fighter = this.fighters.find(f => f.id === fighterId);
        if (!fighter) return;

        if (side === 'a') {
            this.selectedFighterA = fighter;
            // Set default mass
            document.getElementById('mass-a').value = fighter.mass_default;
        } else {
            this.selectedFighterB = fighter;
            // Set default mass
            document.getElementById('mass-b').value = fighter.mass_default;
        }

        this.displayFighterInfo(side, fighter);
        this.checkStartButtonState();
    }

    displayFighterInfo(side, fighter) {
        const infoDiv = document.getElementById(`fighter-${side}-info`);
        infoDiv.className = 'fighter-info';

        let html = `
            <h3>${fighter.name}</h3>
            <p><strong>Type:</strong> ${fighter.type}</p>
            <p><strong>Creator:</strong> ${fighter.creator}</p>
            <p>${fighter.description}</p>
        `;

        // Add strategy tags if present
        if (fighter.strategy_tags && fighter.strategy_tags.length > 0) {
            html += '<div class="fighter-tags">';
            fighter.strategy_tags.forEach(tag => {
                html += `<span class="tag">${tag}</span>`;
            });
            html += '</div>';
        }

        // Add performance stats if present
        if (fighter.performance_stats) {
            html += '<div class="fighter-stats">';
            if (fighter.performance_stats.elo) {
                html += `
                    <div class="stat-row">
                        <span class="stat-label">ELO Rating:</span>
                        <span class="stat-value">${Math.round(fighter.performance_stats.elo)}</span>
                    </div>
                `;
            }
            if (fighter.performance_stats.win_rate !== undefined) {
                html += `
                    <div class="stat-row">
                        <span class="stat-label">Win Rate:</span>
                        <span class="stat-value">${(fighter.performance_stats.win_rate * 100).toFixed(1)}%</span>
                    </div>
                `;
            }
            if (fighter.performance_stats.wins !== undefined) {
                html += `
                    <div class="stat-row">
                        <span class="stat-label">Record:</span>
                        <span class="stat-value">
                            ${fighter.performance_stats.wins}W -
                            ${fighter.performance_stats.losses}L -
                            ${fighter.performance_stats.draws}D
                        </span>
                    </div>
                `;
            }
            html += '</div>';
        }

        infoDiv.innerHTML = html;
    }

    clearFighterInfo(side) {
        const infoDiv = document.getElementById(`fighter-${side}-info`);
        infoDiv.className = 'fighter-info empty';
        infoDiv.innerHTML = '<p>Select a fighter to see details</p>';
    }

    checkStartButtonState() {
        const startBtn = document.getElementById('start-match-btn');
        const canStart = this.selectedFighterA && this.selectedFighterB;
        startBtn.disabled = !canStart;
    }

    async startMatch() {
        // Get configuration
        const massA = parseFloat(document.getElementById('mass-a').value);
        const massB = parseFloat(document.getElementById('mass-b').value);
        const seed = parseInt(document.getElementById('seed').value);
        const maxTicks = parseInt(document.getElementById('max-ticks').value);

        // Show loading view
        this.showView('loading-view');
        document.getElementById('loading-message').textContent =
            `${this.selectedFighterA.name} vs ${this.selectedFighterB.name}`;

        // Prepare request
        const request = {
            fighter_a_id: this.selectedFighterA.id,
            fighter_b_id: this.selectedFighterB.id,
            mass_a: massA,
            mass_b: massB,
            seed: seed,
            max_ticks: maxTicks
        };

        try {
            const response = await fetch('/api/match', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                throw new Error('Match execution failed');
            }

            const result = await response.json();

            if (result.status === 'error') {
                throw new Error(result.error || 'Unknown error');
            }

            // Store match data for export
            this.matchData = result;

            // Display replay
            this.displayReplay(result);

        } catch (error) {
            console.error('Error running match:', error);
            alert(`Failed to run match: ${error.message}`);
            this.showView('selection-view');
        }
    }

    displayReplay(matchData) {
        // Generate standalone HTML replay content
        const replayHTML = this.generateReplayHTML(matchData);

        // Display in iframe
        const iframe = document.getElementById('replay-frame');
        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
        iframeDoc.open();
        iframeDoc.write(replayHTML);
        iframeDoc.close();

        // Show replay view
        this.showView('replay-view');
    }

    generateReplayHTML(matchData) {
        // This generates a minimal inline HTML for the iframe
        // In production, you might want to fetch the actual HtmlRenderer template
        const telemetry = matchData.telemetry;
        const result = matchData.result;
        const spectacleScore = matchData.spectacle_score;

        return `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Atom Combat Replay</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a2e; color: #e9ecef; font-family: monospace; }
        .summary { text-align: center; padding: 20px; }
        .winner { font-size: 2em; color: #27ae60; margin: 20px 0; }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 600px; margin: 0 auto; }
        .stat { background: #16213e; padding: 15px; border-radius: 8px; }
        .stat-label { color: #adb5bd; font-size: 0.9em; }
        .stat-value { font-size: 1.5em; color: #4a90e2; }
    </style>
</head>
<body>
    <div class="summary">
        <h1>Match Complete!</h1>
        <div class="winner">🏆 Winner: ${result.winner}</div>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">${telemetry.fighter_a_name}</div>
                <div class="stat-value">${result.final_hp_a.toFixed(1)} HP</div>
            </div>
            <div class="stat">
                <div class="stat-label">${telemetry.fighter_b_name}</div>
                <div class="stat-value">${result.final_hp_b.toFixed(1)} HP</div>
            </div>
            <div class="stat">
                <div class="stat-label">Duration</div>
                <div class="stat-value">${result.total_ticks} ticks</div>
            </div>
            <div class="stat">
                <div class="stat-label">Spectacle Score</div>
                <div class="stat-value">${(spectacleScore.overall * 100).toFixed(1)}%</div>
            </div>
        </div>
        <p style="margin-top: 30px; color: #adb5bd;">
            Click "Export Replay" to save a full animated replay with playback controls.
        </p>
    </div>
</body>
</html>
        `;
    }

    async exportReplay() {
        if (!this.matchData) {
            alert('No match data to export');
            return;
        }

        try {
            const response = await fetch('/api/export-replay', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    telemetry: this.matchData.telemetry,
                    result: this.matchData.result,
                    spectacle_score: this.matchData.spectacle_score,
                    filename: `${this.matchData.telemetry.fighter_a_name}_vs_${this.matchData.telemetry.fighter_b_name}.html`
                })
            });

            if (!response.ok) {
                throw new Error('Export failed');
            }

            // Download the file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.matchData.telemetry.fighter_a_name}_vs_${this.matchData.telemetry.fighter_b_name}.html`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Error exporting replay:', error);
            alert(`Failed to export replay: ${error.message}`);
        }
    }

    showView(viewId) {
        // Hide all views
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });

        // Show selected view
        document.getElementById(viewId).classList.add('active');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AtomCombatApp();
});
