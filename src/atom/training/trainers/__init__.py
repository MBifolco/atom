"""Training algorithms for Atom Combat."""

__all__ = [
    "CurriculumTrainer",
    "PopulationTrainer",
]


def __getattr__(name):
    if name == "CurriculumTrainer":
        from .curriculum_trainer import CurriculumTrainer
        return CurriculumTrainer
    if name == "PopulationTrainer":
        from .population import PopulationTrainer
        return PopulationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
