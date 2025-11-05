# Atom Combat Web Application

A modern web interface for selecting fighters and running Atom Combat matches.

## Features

- 🎮 **Fighter Selection UI** - Browse and select fighters from the registry
- ⚔️ **Server-Side Execution** - All Python fighters work (rule-based and AI)
- 📊 **Match Configuration** - Configure mass, seed, and other parameters
- 🎬 **Live Replay** - Watch matches with full telemetry
- 💾 **Export Replays** - Save standalone HTML files
- 📈 **Fighter Stats** - View ELO ratings, win rates, and performance metrics

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/JSON
┌────────▼────────┐
│   FastAPI       │
│   (Backend)     │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Fighter │
    │Registry │
    └─────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd web
pip install -r requirements.txt
```

### 2. Build Fighter Registry

```bash
cd ..
python build_registry.py
```

This scans `fighters/` directory and creates `fighters/registry.json`.

### 3. Start the Server

```bash
# Development mode (with auto-reload)
uvicorn web.app:app --reload

# Production mode
uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Open in Browser

Navigate to: http://localhost:8000

## API Endpoints

### GET /api/fighters

List all fighters from the registry.

**Query Parameters:**
- `filter_type` (optional): Filter by type (e.g., "rule-based", "onnx-ai")
- `filter_tags` (optional): Comma-separated strategy tags

**Response:**
```json
[
  {
    "id": "rusher",
    "name": "Rusher",
    "description": "Aggressive rushing fighter...",
    "type": "rule-based",
    "mass_default": 70.0,
    "strategy_tags": ["aggressive", "stamina-aware"],
    ...
  }
]
```

### GET /api/fighters/{fighter_id}

Get detailed information about a specific fighter.

**Response:**
```json
{
  "id": "rusher",
  "name": "Rusher",
  "description": "...",
  "performance_stats": {
    "elo": 1234,
    "win_rate": 0.675
  },
  ...
}
```

### POST /api/match

Execute a match between two fighters.

**Request:**
```json
{
  "fighter_a_id": "rusher",
  "fighter_b_id": "tank",
  "mass_a": 70.0,
  "mass_b": 75.0,
  "seed": 42,
  "max_ticks": 1000
}
```

**Response:**
```json
{
  "status": "complete",
  "telemetry": { /* full tick-by-tick data */ },
  "result": {
    "winner": "Rusher",
    "total_ticks": 456,
    "final_hp_a": 45.2,
    "final_hp_b": 0.0
  },
  "spectacle_score": {
    "overall": 0.67,
    "close_finish": 0.85,
    ...
  }
}
```

### POST /api/export-replay

Export a replay as standalone HTML file.

**Request:**
```json
{
  "telemetry": { /* match telemetry */ },
  "result": { /* match result */ },
  "spectacle_score": { /* optional */ },
  "filename": "my_replay.html"
}
```

**Response:** HTML file download

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "fighters_loaded": 16,
  "registry_path": "/path/to/registry.json"
}
```

## API Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Usage Examples

### Using the Web UI

1. Open http://localhost:8000 in your browser
2. Select Fighter A from the dropdown
3. Select Fighter B from the dropdown
4. Configure match settings (optional)
5. Click "Start Match"
6. Watch the replay
7. Click "Export Replay" to save standalone HTML

### Using the API Directly

```bash
# List all fighters
curl http://localhost:8000/api/fighters

# Run a match
curl -X POST http://localhost:8000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "fighter_a_id": "rusher",
    "fighter_b_id": "tank",
    "seed": 42
  }'
```

### Using Python

```python
import requests

# List fighters
response = requests.get('http://localhost:8000/api/fighters')
fighters = response.json()
print(f"Found {len(fighters)} fighters")

# Run a match
match_request = {
    "fighter_a_id": "rusher",
    "fighter_b_id": "tank",
    "mass_a": 70.0,
    "mass_b": 75.0,
    "seed": 42
}

response = requests.post(
    'http://localhost:8000/api/match',
    json=match_request
)

result = response.json()
print(f"Winner: {result['result']['winner']}")
print(f"Spectacle Score: {result['spectacle_score']['overall']:.2f}")
```

## Development

### Project Structure

```
web/
├── app.py              # FastAPI application
├── models.py           # Pydantic models
├── requirements.txt    # Python dependencies
├── static/
│   ├── index.html     # Main UI
│   ├── app.js         # Frontend logic
│   └── styles.css     # Styling
└── README.md          # This file
```

### Running Tests

```bash
# Start server in one terminal
uvicorn web.app:app --reload

# Test API in another terminal
curl http://localhost:8000/health
curl http://localhost:8000/api/fighters
```

### Adding New Fighters

1. Add your fighter to `fighters/` directory
2. Rebuild registry: `python build_registry.py`
3. Restart the server (or it will auto-reload if using `--reload`)
4. Fighter will appear in the UI automatically

## Configuration

### Port Configuration

Change port in startup command:
```bash
uvicorn web.app:app --port 9000
```

### CORS Configuration

CORS is currently set to allow all origins for development. For production, update in `web/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Registry Location

Default location: `fighters/registry.json`

To use a custom location, modify `web/app.py`:

```python
registry_path = Path("/custom/path/registry.json")
registry = FighterRegistry(registry_path)
```

## Troubleshooting

### Server won't start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`
**Solution:** Install dependencies: `pip install -r web/requirements.txt`

### No fighters showing up

**Error:** Empty fighter list
**Solution:** Build registry: `python build_registry.py`

### Fighter fails to load

**Error:** 500 error when running match
**Solution:**
1. Check fighter file exists: `ls fighters/examples/rusher.py`
2. Validate fighter: `python build_registry.py --validate`
3. Check server logs for detailed error

### Port already in use

**Error:** `Address already in use`
**Solution:** Use a different port: `uvicorn web.app:app --port 8001`

## Performance

- **Match Execution:** ~1-5 seconds depending on match length
- **Registry Load:** <100ms for 100 fighters
- **Replay Export:** ~50-100ms

## Security Notes

- Fighter code is executed server-side (Python sandbox)
- No arbitrary code execution from client
- All fighter files must be in registered directories
- Code hash verification available via registry

## Future Enhancements

- [ ] WebSocket support for live match updates
- [ ] Match history and replay library
- [ ] Tournament bracket system
- [ ] Fighter comparison/statistics
- [ ] Leaderboard/ELO tracking
- [ ] User authentication and fighter uploads
- [ ] Real-time match spectating

## License

Part of Atom Combat platform. See main project LICENSE.
