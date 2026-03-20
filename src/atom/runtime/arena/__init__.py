"""Runtime arena package."""

from importlib import import_module

__all__ = ["Arena1DJAXJit", "FighterState", "StanceConfig", "WorldConfig"]

_SYMBOL_TO_MODULE = {
    "Arena1DJAXJit": "src.atom.runtime.arena.arena_1d_jax_jit",
    "FighterState": "src.atom.runtime.arena.fighter",
    "StanceConfig": "src.atom.runtime.arena.world_config",
    "WorldConfig": "src.atom.runtime.arena.world_config",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SYMBOL_TO_MODULE[name])
    return getattr(module, name)
