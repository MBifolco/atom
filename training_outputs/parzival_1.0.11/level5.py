"""
Trained Fighter - ONNX Wrapper

Auto-generated wrapper for trained ONNX model.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# ONNX model path (relative to this file)
ONNX_MODEL = "level5.onnx"

# Global session (loaded once)
_session = None
_stance_names = ["neutral", "extended", "retracted", "defending"]


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
        snapshot: Combat snapshot

    Returns:
        {"acceleration": float, "stance": str}
    """
    session = _load_session()

    # Convert snapshot to observation
    you = snapshot["you"]
    opponent = snapshot["opponent"]
    arena = snapshot["arena"]

    you_hp_norm = you["hp"] / you["max_hp"]
    you_stamina_norm = you["stamina"] / you["max_stamina"]
    opp_hp_norm = opponent["hp"] / opponent["max_hp"]
    opp_stamina_norm = opponent["stamina"] / opponent["max_stamina"]

    obs = np.array([
        you["position"],
        you["velocity"],
        you_hp_norm,
        you_stamina_norm,
        opponent["distance"],
        opponent["velocity"],
        opp_hp_norm,
        opp_stamina_norm,
        arena["width"]
    ], dtype=np.float32).reshape(1, -1)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: obs})

    # Parse action
    # Action space is Box: [acceleration_normalized, stance_selector]
    action = outputs[0][0]
    acceleration_normalized = np.clip(action[0], -1.0, 1.0)
    stance_idx = int(np.clip(action[1], 0, 3))

    # Scale acceleration (max_acceleration = 4.3751)
    acceleration = float(acceleration_normalized * 4.3751)
    stance = _stance_names[stance_idx]

    return {"acceleration": acceleration, "stance": stance}
