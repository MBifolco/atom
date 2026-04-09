"""
Noble_Heron_G17 — Trained AI fighter exported from population training.

Record: 11-3-0
ELO: 1563
Generation: 17
Lineage: Noble_Heron_G17

Trained using PPO with 4D logit action space.
"""

import numpy as np
from pathlib import Path

_model = None
_model_path = str(Path(__file__).parent / "Noble_Heron_G17.zip")


def _load_model():
    global _model
    if _model is not None:
        return _model
    from stable_baselines3 import PPO
    _model = PPO.load(_model_path, device="cpu")
    return _model


def decide(snapshot):
    """
    Make a decision based on the current game state.

    Args:
        snapshot: dict with you, opponent, and arena state

    Returns:
        dict with acceleration (float) and stance (str)
    """
    from src.atom.training.signal_engine import build_observation_from_snapshot

    model = _load_model()
    obs = build_observation_from_snapshot(snapshot)
    action, _ = model.predict(obs, deterministic=False)

    acceleration = float(np.clip(action[0], -1.0, 1.0)) * 4.375
    stance_idx = int(np.argmax(action[1:4]))
    stances = ["neutral", "extended", "defending"]

    return {
        "acceleration": acceleration,
        "stance": stances[stance_idx],
    }
