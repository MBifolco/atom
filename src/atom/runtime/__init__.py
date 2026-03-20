"""Transitional runtime namespace for Atom Combat."""

from importlib import import_module

__all__ = [
    "Action",
    "Arena1DJAXJit",
    "AsciiRenderer",
    "FighterState",
    "HtmlRenderer",
    "MatchOrchestrator",
    "MatchResult",
    "ProtocolValidator",
    "ReplayStore",
    "Snapshot",
    "SpectacleEvaluator",
    "SpectacleScore",
    "StanceConfig",
    "WorldConfig",
    "generate_snapshot",
    "load_replay",
    "save_replay",
]

_SYMBOL_TO_MODULE = {
    "Action": "src.atom.runtime.protocol",
    "Arena1DJAXJit": "src.atom.runtime.arena",
    "AsciiRenderer": "src.atom.runtime.renderer",
    "FighterState": "src.atom.runtime.arena",
    "HtmlRenderer": "src.atom.runtime.renderer",
    "MatchOrchestrator": "src.atom.runtime.orchestrator",
    "MatchResult": "src.atom.runtime.orchestrator",
    "ProtocolValidator": "src.atom.runtime.protocol",
    "ReplayStore": "src.atom.runtime.telemetry",
    "Snapshot": "src.atom.runtime.protocol",
    "SpectacleEvaluator": "src.atom.runtime.evaluator",
    "SpectacleScore": "src.atom.runtime.evaluator",
    "StanceConfig": "src.atom.runtime.arena",
    "WorldConfig": "src.atom.runtime.arena",
    "generate_snapshot": "src.atom.runtime.protocol",
    "load_replay": "src.atom.runtime.telemetry",
    "save_replay": "src.atom.runtime.telemetry",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SYMBOL_TO_MODULE[name])
    return getattr(module, name)
