"""Compatibility alias for the signal engine module."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.signal_engine")
