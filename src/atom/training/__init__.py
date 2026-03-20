"""Transitional training namespace for Atom Combat."""

from importlib import import_module

__all__ = ["AtomCombatEnv", "ProgressiveTrainer", "VmapEnvWrapper"]

_SYMBOL_TO_MODULE = {
    "AtomCombatEnv": "src.atom.training.gym_env",
    "ProgressiveTrainer": "src.atom.training.pipelines",
    "VmapEnvWrapper": "src.atom.training.vmap_env_wrapper",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SYMBOL_TO_MODULE[name])
    return getattr(module, name)
