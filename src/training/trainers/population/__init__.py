"""Population-Based Training for Atom Combat"""

from src.atom.training.trainers.population.elo_tracker import EloTracker

__all__ = ["PopulationTrainer", "EloTracker"]


def __getattr__(name):
    """Lazy import PopulationTrainer to avoid circular dependencies."""
    if name == "PopulationTrainer":
        from src.atom.training.trainers.population.population_trainer import PopulationTrainer

        return PopulationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
