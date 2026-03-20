"""
Training algorithms for Atom Combat.

Available trainers:
- curriculum: Curriculum-based progressive training
- population: Population-based training for diverse strategies
"""

import sys
from pathlib import Path

# Setup paths for imports when this package is loaded
project_root = Path(__file__).parent.parent.parent.parent  # /home/biff/eng/atom
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Lazy imports to avoid circular dependencies
__all__ = [
    'CurriculumTrainer',
    'PopulationTrainer',
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'CurriculumTrainer':
        from .curriculum_trainer import CurriculumTrainer
        return CurriculumTrainer
    elif name == 'PopulationTrainer':
        from .population.population_trainer import PopulationTrainer
        return PopulationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
