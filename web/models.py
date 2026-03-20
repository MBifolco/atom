"""Compatibility wrapper for Atom Combat web API models."""

from apps.web.models import ExportReplayRequest, FighterResponse, MatchRequest, MatchResponse

__all__ = [
    "ExportReplayRequest",
    "FighterResponse",
    "MatchRequest",
    "MatchResponse",
]
