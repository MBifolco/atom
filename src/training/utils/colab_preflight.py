"""Compatibility alias for training utility module colab_preflight."""

from importlib import import_module as _import_module
import sys as _sys


if __name__ == "__main__":
    from src.atom.training.utils.colab_preflight import main as _main

    raise SystemExit(_main())


_sys.modules[__name__] = _import_module("src.atom.training.utils.colab_preflight")
