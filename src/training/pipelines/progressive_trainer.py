"""Compatibility alias for the progressive training pipeline."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.pipelines.progressive_trainer")
