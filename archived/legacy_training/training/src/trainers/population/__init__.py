"""Population-Based Training for Atom Combat"""

# Don't import PopulationTrainer automatically to avoid import issues
# Import only EloTracker which has no external dependencies
from .elo_tracker import EloTracker

__all__ = ["PopulationTrainer", "EloTracker"]

def __getattr__(name):
    """Lazy import PopulationTrainer to avoid circular dependencies."""
    if name == "PopulationTrainer":
        from .population_trainer import PopulationTrainer
        return PopulationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")