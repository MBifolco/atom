"""Compatibility alias for population module population_protocols."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.trainers.population.population_protocols")
