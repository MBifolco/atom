"""Compatibility alias for the vmap environment wrapper module."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("src.atom.training.vmap_env_wrapper")
