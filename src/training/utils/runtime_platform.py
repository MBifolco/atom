"""Compatibility alias for runtime GPU platform utilities."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.utils.runtime_platform")
