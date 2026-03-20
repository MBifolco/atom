"""Population-Based Training for Atom Combat."""

from .elo_tracker import EloTracker

__all__ = ["PopulationTrainer", "EloTracker"]


def __getattr__(name):
    if name == "PopulationTrainer":
        from .population_trainer import PopulationTrainer
        return PopulationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
