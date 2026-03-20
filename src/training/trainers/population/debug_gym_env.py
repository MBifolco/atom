"""Compatibility alias for population module debug_gym_env."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.trainers.population.debug_gym_env")
