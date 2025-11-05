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

    async displayReplay(matchData) {
        // Show loading message while generating replay HTML
        document.getElementById('loading-message').textContent = 'Generating replay...';

        try {
            // Request full replay HTML from server
            const response = await fetch('/api/generate-replay-html', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    telemetry: matchData.telemetry,
                    result: matchData.result,
                    spectacle_score: matchData.spectacle_score,
                    filename: 'replay.html'
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate replay HTML');
            }

            // Get the full HTML content
            const replayHTML = await response.text();

            // Display in iframe
            const iframe = document.getElementById('replay-frame');
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            iframeDoc.open();
            iframeDoc.write(replayHTML);
            iframeDoc.close();

            // Show replay view
            this.showView('replay-view');

        } catch (error) {
            console.error('Error displaying replay:', error);
            alert(`Failed to display replay: ${error.message}`);
            this.showView('selection-view');
        }
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
