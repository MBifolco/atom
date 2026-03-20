"""Atom Combat web application backend."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from apps.web.models import ExportReplayRequest, FighterResponse, MatchRequest, MatchResponse
from src.atom.runtime.arena import WorldConfig
from src.atom.runtime.evaluator import SpectacleEvaluator, SpectacleScore
from src.atom.runtime.orchestrator import MatchOrchestrator, MatchResult
from src.atom.registry import FighterRegistry
from src.atom.runtime.renderer import HtmlRenderer

LEGACY_WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = LEGACY_WEB_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Atom Combat API",
    description="Web API for Atom Combat fighter selection and match execution",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

registry_path = PROJECT_ROOT / "fighters" / "registry.json"
registry = FighterRegistry(registry_path)
_fighter_cache: dict[str, object] = {}


def load_fighter_function(fighter_id: str):
    """Load a fighter decide function from the registry."""
    if fighter_id in _fighter_cache:
        return _fighter_cache[fighter_id]

    metadata = registry.get_fighter(fighter_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_id}")

    file_path = PROJECT_ROOT / metadata.file_path
    if not file_path.exists():
        raise HTTPException(status_code=500, detail=f"Fighter file not found: {metadata.file_path}")

    try:
        spec = importlib.util.spec_from_file_location("fighter_module", file_path)
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail=f"Cannot load fighter module: {fighter_id}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "decide"):
            raise HTTPException(status_code=500, detail=f"Fighter missing decide function: {fighter_id}")

        decide_func = module.decide
        _fighter_cache[fighter_id] = decide_func
        return decide_func
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error loading fighter {fighter_id}: {exc}")


@app.get("/")
async def root():
    """Serve the main application page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        content="<h1>Atom Combat Web UI</h1><p>UI not found. Please create web/static/index.html</p>",
        status_code=200,
    )


@app.get("/api/fighters", response_model=List[FighterResponse])
async def list_fighters(filter_type: str = None, filter_tags: str = None):
    """List fighters from the registry with optional filters."""
    tags = filter_tags.split(",") if filter_tags else None
    fighters = registry.list_fighters(filter_type=filter_type, filter_tags=tags)
    return [FighterResponse(**fighter.to_dict()) for fighter in fighters]


@app.get("/api/fighters/{fighter_id}", response_model=FighterResponse)
async def get_fighter(fighter_id: str):
    """Get detailed information about a specific fighter."""
    metadata = registry.get_fighter(fighter_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_id}")
    return FighterResponse(**metadata.to_dict())


@app.post("/api/match", response_model=MatchResponse)
async def run_match(request: MatchRequest):
    """Execute a match between two fighters."""
    try:
        fighter_a_func = load_fighter_function(request.fighter_a_id)
        fighter_b_func = load_fighter_function(request.fighter_b_id)

        metadata_a = registry.get_fighter(request.fighter_a_id)
        metadata_b = registry.get_fighter(request.fighter_b_id)

        mass_a = request.mass_a if request.mass_a is not None else metadata_a.mass_default
        mass_b = request.mass_b if request.mass_b is not None else metadata_b.mass_default

        config = WorldConfig()
        orchestrator = MatchOrchestrator(config=config, max_ticks=request.max_ticks, record_telemetry=True)

        fighter_a_spec = {"name": metadata_a.name, "mass": mass_a, "position": request.position_a}
        fighter_b_spec = {"name": metadata_b.name, "mass": mass_b, "position": request.position_b}

        result = orchestrator.run_match(
            fighter_a_spec,
            fighter_b_spec,
            fighter_a_func,
            fighter_b_func,
            seed=request.seed,
        )

        spectacle_score = SpectacleEvaluator().evaluate(result.telemetry, result)

        return MatchResponse(
            status="complete",
            telemetry=result.telemetry,
            result={
                "winner": result.winner,
                "total_ticks": result.total_ticks,
                "final_hp_a": result.final_hp_a,
                "final_hp_b": result.final_hp_b,
            },
            spectacle_score={
                "duration": spectacle_score.duration,
                "close_finish": spectacle_score.close_finish,
                "stamina_drama": spectacle_score.stamina_drama,
                "comeback_potential": spectacle_score.comeback_potential,
                "positional_exchange": spectacle_score.positional_exchange,
                "pacing_variety": spectacle_score.pacing_variety,
                "collision_drama": spectacle_score.collision_drama,
                "overall": spectacle_score.overall,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        return MatchResponse(status="error", error=str(exc))


@app.get("/api/replay-html/{match_id}")
async def get_replay_html(match_id: str):
    """Placeholder replay HTML endpoint."""
    raise HTTPException(status_code=501, detail="Use POST /api/generate-replay-html instead")


@app.post("/api/generate-replay-html")
async def generate_replay_html_inline(request: ExportReplayRequest):
    """Generate full replay HTML for inline display."""
    try:
        renderer = HtmlRenderer()
        replays_dir = PROJECT_ROOT / "outputs" / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", dir=str(replays_dir), delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        match_result = MatchResult(
            winner=request.result["winner"],
            total_ticks=request.result["total_ticks"],
            final_hp_a=request.result["final_hp_a"],
            final_hp_b=request.result["final_hp_b"],
            telemetry=request.telemetry,
            events=request.telemetry.get("events", []),
        )

        spectacle_score = SpectacleScore(**request.spectacle_score) if request.spectacle_score else None
        output_path = renderer.generate_replay_html(
            request.telemetry,
            match_result,
            str(tmp_path),
            spectacle_score=spectacle_score,
        )

        html_content = output_path.read_text()
        try:
            output_path.unlink()
        except OSError:
            pass

        return PlainTextResponse(content=html_content, media_type="text/html")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating replay HTML: {exc}")


@app.post("/api/export-replay")
async def export_replay(request: ExportReplayRequest):
    """Export a replay as standalone HTML file."""
    try:
        renderer = HtmlRenderer()
        replays_dir = PROJECT_ROOT / "outputs" / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", dir=str(replays_dir), delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        match_result = MatchResult(
            winner=request.result["winner"],
            total_ticks=request.result["total_ticks"],
            final_hp_a=request.result["final_hp_a"],
            final_hp_b=request.result["final_hp_b"],
            telemetry=request.telemetry,
            events=request.telemetry.get("events", []),
        )

        spectacle_score = SpectacleScore(**request.spectacle_score) if request.spectacle_score else None
        output_path = renderer.generate_replay_html(
            request.telemetry,
            match_result,
            str(tmp_path),
            spectacle_score=spectacle_score,
        )

        return FileResponse(path=str(output_path), filename=request.filename, media_type="text/html", background=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating replay: {exc}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "fighters_loaded": len(registry.fighters),
        "registry_path": str(registry.registry_path),
    }


def main() -> None:
    """Run the FastAPI development server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
