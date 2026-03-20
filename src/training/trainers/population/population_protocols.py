"""
Shared typing protocols for population trainer components.

These protocols centralize the lightweight interfaces exchanged between
parallel orchestration, evolution, and evaluation helper modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Protocol, Tuple


class FighterModelProtocol(Protocol):
    """Minimal protocol for trainable model objects."""

    def save(self, path: Path) -> None:
        ...


class PopulationFighterProtocol(Protocol):
    """Minimal protocol for population fighter objects."""

    name: str
    mass: float
    model: Any
    training_episodes: int
    last_checkpoint: str | None


class EloTrackerPopulationProtocol(Protocol):
    """ELO tracker operations required by population evolution logic."""

    def get_rankings(self) -> List[Any]:
        ...

    def remove_fighter(self, name: str) -> None:
        ...

    def add_fighter(self, name: str) -> None:
        ...


class EloTrackerEvaluationProtocol(Protocol):
    """ELO tracker operations required by evaluation match updates."""

    def update_ratings(
        self,
        fighter_a: str,
        fighter_b: str,
        result: str,
        damage_a: float,
        damage_b: float,
        metadata: dict,
    ) -> Tuple[float, float]:
        ...
