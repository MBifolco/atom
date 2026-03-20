"""Pydantic models for FastAPI endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FighterResponse(BaseModel):
    """Fighter metadata response."""

    id: str
    name: str
    description: str
    creator: str
    type: str
    file_path: str
    mass_default: float
    strategy_tags: List[str] = []
    performance_stats: Optional[Dict[str, Any]] = None
    version: str
    created_date: str
    protocol_version: str
    world_spec_version: str
    code_hash: str


class MatchRequest(BaseModel):
    """Request to run a match between two fighters."""

    fighter_a_id: str = Field(..., description="Fighter A ID from registry")
    fighter_b_id: str = Field(..., description="Fighter B ID from registry")
    mass_a: Optional[float] = Field(None, description="Fighter A mass (uses default if not specified)")
    mass_b: Optional[float] = Field(None, description="Fighter B mass (uses default if not specified)")
    seed: int = Field(42, description="Random seed for reproducibility")
    max_ticks: int = Field(1000, description="Maximum ticks before timeout")
    position_a: float = Field(2.0, description="Fighter A starting position")
    position_b: float = Field(10.0, description="Fighter B starting position")


class MatchResponse(BaseModel):
    """Response from running a match."""

    status: str = Field(..., description="'complete', 'error', or 'in_progress'")
    telemetry: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    spectacle_score: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExportReplayRequest(BaseModel):
    """Request to export a replay as standalone HTML."""

    telemetry: Dict[str, Any]
    result: Dict[str, Any]
    spectacle_score: Optional[Dict[str, Any]] = None
    filename: str = Field("replay.html", description="Desired filename")
