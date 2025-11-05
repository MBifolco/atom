"""
Atom Combat Web Application

FastAPI backend for fighter selection and match execution.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import importlib.util
from typing import List
import tempfile

from src.registry import FighterRegistry
from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import HtmlRenderer
from web.models import (
    FighterResponse,
    MatchRequest,
    MatchResponse,
    ExportReplayRequest
)

# Initialize FastAPI app
app = FastAPI(
    title="Atom Combat API",
    description="Web API for Atom Combat fighter selection and match execution",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize registry
registry_path = project_root / "fighters" / "registry.json"
registry = FighterRegistry(registry_path)

# Cache for loaded fighter functions
_fighter_cache = {}


def load_fighter_function(fighter_id: str):
    """
    Load a fighter's decide function from the registry.

    Args:
        fighter_id: Fighter ID

    Returns:
        decide function

    Raises:
        HTTPException if fighter not found or cannot be loaded
    """
    # Check cache first
    if fighter_id in _fighter_cache:
        return _fighter_cache[fighter_id]

    # Get fighter metadata
    metadata = registry.get_fighter(fighter_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_id}")

    # Load the fighter module
    file_path = project_root / metadata.file_path
    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Fighter file not found: {metadata.file_path}"
        )

    try:
        spec = importlib.util.spec_from_file_location("fighter_module", file_path)
        if not spec or not spec.loader:
            raise HTTPException(
                status_code=500,
                detail=f"Cannot load fighter module: {fighter_id}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'decide'):
            raise HTTPException(
                status_code=500,
                detail=f"Fighter missing decide function: {fighter_id}"
            )

        decide_func = module.decide
        _fighter_cache[fighter_id] = decide_func
        return decide_func

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading fighter {fighter_id}: {str(e)}"
        )


@app.get("/")
async def root():
    """Serve the main application page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return HTMLResponse(
            content="<h1>Atom Combat Web UI</h1><p>UI not found. Please create web/static/index.html</p>",
            status_code=200
        )


@app.get("/api/fighters", response_model=List[FighterResponse])
async def list_fighters(
    filter_type: str = None,
    filter_tags: str = None
):
    """
    List all fighters from the registry.

    Args:
        filter_type: Optional filter by type (rule-based, onnx-ai, etc.)
        filter_tags: Optional comma-separated strategy tags

    Returns:
        List of fighter metadata
    """
    # Parse filter tags
    tags = filter_tags.split(",") if filter_tags else None

    # Get fighters from registry
    fighters = registry.list_fighters(
        filter_type=filter_type,
        filter_tags=tags
    )

    # Convert to response models
    return [FighterResponse(**f.to_dict()) for f in fighters]


@app.get("/api/fighters/{fighter_id}", response_model=FighterResponse)
async def get_fighter(fighter_id: str):
    """
    Get detailed information about a specific fighter.

    Args:
        fighter_id: Fighter ID

    Returns:
        Fighter metadata
    """
    metadata = registry.get_fighter(fighter_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_id}")

    return FighterResponse(**metadata.to_dict())


@app.post("/api/match", response_model=MatchResponse)
async def run_match(request: MatchRequest):
    """
    Execute a match between two fighters.

    Args:
        request: Match configuration

    Returns:
        Match result with telemetry
    """
    try:
        # Load fighters
        fighter_a_func = load_fighter_function(request.fighter_a_id)
        fighter_b_func = load_fighter_function(request.fighter_b_id)

        # Get fighter metadata for names and default masses
        metadata_a = registry.get_fighter(request.fighter_a_id)
        metadata_b = registry.get_fighter(request.fighter_b_id)

        # Use provided masses or defaults
        mass_a = request.mass_a if request.mass_a is not None else metadata_a.mass_default
        mass_b = request.mass_b if request.mass_b is not None else metadata_b.mass_default

        # Create world config
        config = WorldConfig()

        # Set up match orchestrator
        orchestrator = MatchOrchestrator(
            config=config,
            max_ticks=request.max_ticks,
            record_telemetry=True
        )

        # Create fighter specs
        fighter_a_spec = {
            "name": metadata_a.name,
            "mass": mass_a,
            "position": request.position_a
        }
        fighter_b_spec = {
            "name": metadata_b.name,
            "mass": mass_b,
            "position": request.position_b
        }

        # Run the match
        result = orchestrator.run_match(
            fighter_a_spec,
            fighter_b_spec,
            fighter_a_func,
            fighter_b_func,
            seed=request.seed
        )

        # Evaluate spectacle
        evaluator = SpectacleEvaluator()
        spectacle_score = evaluator.evaluate(result.telemetry, result)

        # Format response
        return MatchResponse(
            status="complete",
            telemetry=result.telemetry,
            result={
                "winner": result.winner,
                "total_ticks": result.total_ticks,
                "final_hp_a": result.final_hp_a,
                "final_hp_b": result.final_hp_b
            },
            spectacle_score={
                "duration": spectacle_score.duration,
                "close_finish": spectacle_score.close_finish,
                "stamina_drama": spectacle_score.stamina_drama,
                "comeback_potential": spectacle_score.comeback_potential,
                "positional_exchange": spectacle_score.positional_exchange,
                "pacing_variety": spectacle_score.pacing_variety,
                "collision_drama": spectacle_score.collision_drama,
                "overall": spectacle_score.overall
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other errors
        return MatchResponse(
            status="error",
            error=str(e)
        )


@app.post("/api/export-replay")
async def export_replay(request: ExportReplayRequest):
    """
    Export a replay as standalone HTML file.

    Args:
        request: Replay data and configuration

    Returns:
        File download of standalone HTML
    """
    try:
        # Create renderer
        renderer = HtmlRenderer()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        # Generate HTML
        from src.orchestrator import MatchResult
        match_result = MatchResult(
            winner=request.result["winner"],
            total_ticks=request.result["total_ticks"],
            final_hp_a=request.result["final_hp_a"],
            final_hp_b=request.result["final_hp_b"],
            telemetry=request.telemetry,
            events=request.telemetry.get("events", [])
        )

        # Create spectacle score if provided
        spectacle_score = None
        if request.spectacle_score:
            from src.evaluator import SpectacleScore
            spectacle_score = SpectacleScore(**request.spectacle_score)

        # Generate the HTML
        output_path = renderer.generate_replay_html(
            request.telemetry,
            match_result,
            str(tmp_path),
            spectacle_score=spectacle_score
        )

        # Return as file download
        return FileResponse(
            path=str(output_path),
            filename=request.filename,
            media_type="text/html"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating replay: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "fighters_loaded": len(registry.fighters),
        "registry_path": str(registry.registry_path)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
