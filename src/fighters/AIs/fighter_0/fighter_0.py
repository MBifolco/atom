"""
fighter_0 - Trained AI Fighter

Generation: 0
Lineage: founder
Mass: 70.0kg
Training Episodes: 0

Auto-generated wrapper for trained ONNX model.
Compatible with atom_fight.py
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from src.atom.training.signal_engine import build_observation_from_snapshot
from src.atom.training.action_codec import scale_and_validate_action, stance_idx_to_str

# ONNX model path (relative to this file)
ONNX_MODEL = "fighter_0.onnx"

# Global session (loaded once)
_session = None


def _load_session():
    """Load ONNX session (lazy loading)."""
    global _session
    if _session is None:
        model_path = Path(__file__).parent / ONNX_MODEL
        _session = ort.InferenceSession(str(model_path))
    return _session


def decide(snapshot):
    """
    Decision function for trained fighter.

    Args:
        snapshot: Combat snapshot from the arena
            - you: dict with position, velocity, hp, max_hp, stamina, max_stamina
            - opponent: dict with distance, velocity, hp, max_hp, stamina, max_stamina
            - arena: dict with width

    Returns:
        dict with:
            - acceleration: float (-4.5 to +4.5)
            - stance: str ("neutral", "extended", "defending")
    """
    session = _load_session()

    obs = build_observation_from_snapshot(snapshot).reshape(1, -1)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: obs})

    # Parse action — 2D Box: [acceleration, stance_selector]
    action = outputs[0][0]
    acceleration, stance_idx = scale_and_validate_action(action, 4.5)
    stance = stance_idx_to_str(stance_idx)

    return {"acceleration": acceleration, "stance": stance}
