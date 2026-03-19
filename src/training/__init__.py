"""Training modules for Atom Combat AI fighters."""

__all__ = ["AtomCombatEnv"]


def __getattr__(name):
    # Lazy import to avoid pulling in JAX-heavy modules during package import.
    if name == "AtomCombatEnv":
        from .gym_env import AtomCombatEnv
        return AtomCombatEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
