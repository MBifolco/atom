"""
ONNX Fighter Export and Inference

Export trained models to ONNX and load them for matches.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


class ONNXFighter:
    """
    Wrapper to use ONNX model as a fighter decision function.

    Usage:
        fighter = ONNXFighter("my_fighter.onnx")
        action = fighter.decide(snapshot)  # Compatible with atom_fight.py
    """

    def __init__(self, onnx_path: str):
        """
        Load ONNX model for inference.

        Args:
            onnx_path: Path to ONNX model file
        """
        self.onnx_path = Path(onnx_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Create ONNX runtime session
        self.session = ort.InferenceSession(str(self.onnx_path))

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Stance mapping (matches gym_env.py)
        self.stance_names = ["neutral", "extended", "retracted", "defending"]

    def decide(self, snapshot: dict) -> dict:
        """
        Make decision using ONNX model.

        Args:
            snapshot: Combat snapshot (same format as handcoded fighters)

        Returns:
            {"acceleration": float, "stance": str}
        """
        # Convert snapshot to observation array (matches gym_env.py)
        obs = self._snapshot_to_obs(snapshot)

        # Run inference
        obs_input = obs.reshape(1, -1).astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: obs_input})

        # Parse outputs
        # ONNX outputs raw network values that need clipping to action space bounds
        action = outputs[0][0]  # First output, first (and only) sample

        # Action space is Box: [acceleration_normalized (-1 to 1), stance_selector (0 to 3.99)]
        # Must clip to action space bounds (SB3 does this in predict())
        acceleration_normalized = float(np.clip(action[0], -1.0, 1.0))
        stance_idx = int(np.clip(action[1], 0, 3))

        # Scale acceleration to arena range
        from src.arena import WorldConfig
        config = WorldConfig()
        acceleration = float(acceleration_normalized * config.max_acceleration)

        # Get stance name
        stance = self.stance_names[stance_idx]

        return {
            "acceleration": acceleration,
            "stance": stance
        }

    def _snapshot_to_obs(self, snapshot: dict) -> np.ndarray:
        """
        Convert combat snapshot to observation array.

        Must match the observation format in gym_env.py:
        [position, velocity, hp_norm, stamina_norm, distance, rel_velocity, opp_hp_norm, opp_stamina_norm, arena_width]
        """
        you = snapshot["you"]
        opponent = snapshot["opponent"]
        arena = snapshot["arena"]

        # Normalize HP and stamina
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
            opponent["velocity"],  # Already relative in snapshot
            opp_hp_norm,
            opp_stamina_norm,
            arena["width"]
        ], dtype=np.float32)

        return obs


def export_to_onnx(sb3_model_path: str, output_path: str):
    """
    Export Stable-Baselines3 model to ONNX format.
    Supports both PPO and SAC models.

    Args:
        sb3_model_path: Path to saved SB3 model (.zip)
        output_path: Where to save ONNX model (.onnx)
    """
    import torch
    from stable_baselines3 import PPO, SAC
    import torch.nn as nn

    print(f"Loading SB3 model from: {sb3_model_path}")

    # Try to detect model type and load appropriately
    try:
        model = PPO.load(sb3_model_path)
        model_type = "PPO"
    except:
        try:
            model = SAC.load(sb3_model_path)
            model_type = "SAC"
        except Exception as e:
            raise ValueError(f"Could not load model as PPO or SAC: {e}")

    print(f"Detected model type: {model_type}")
    print("Extracting policy network with action processing...")

    # Create a wrapper that handles both PPO and SAC policies
    if model_type == "PPO":
        class PolicyWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.features_extractor = policy.features_extractor
                self.mlp_extractor = policy.mlp_extractor
                self.action_net = policy.action_net

            def forward(self, obs):
                features = self.features_extractor(obs)
                latent_pi, _ = self.mlp_extractor(features)
                mean_actions = self.action_net(latent_pi)
                return mean_actions
    else:  # SAC
        class PolicyWrapper(nn.Module):
            def __init__(self, policy, action_space):
                super().__init__()
                # SAC actor (includes squashing via tanh)
                self.actor = policy.actor
                # Action space bounds for rescaling
                self.action_space_low = torch.FloatTensor(action_space.low)
                self.action_space_high = torch.FloatTensor(action_space.high)

            def forward(self, obs):
                # Get squashed actions from actor (deterministic=True uses mean with tanh)
                mean_actions = self.actor(obs, deterministic=True)

                # Rescale from [-1, 1] (after tanh) to action space bounds
                # Formula: low + (action + 1) / 2 * (high - low)
                actions = self.action_space_low + (mean_actions + 1.0) / 2.0 * (
                    self.action_space_high - self.action_space_low
                )

                return actions

    # Pass action space for SAC rescaling
    if model_type == "SAC":
        wrapped_policy = PolicyWrapper(model.policy, model.action_space)
    else:
        wrapped_policy = PolicyWrapper(model.policy)
    wrapped_policy.eval()

    # Create dummy input (batch_size=1, obs_dim=9)
    dummy_obs = torch.randn(1, 9)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapped_policy,
        dummy_obs,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}},
        opset_version=11
    )

    print(f"✓ ONNX model saved: {output_path}")


def create_fighter_wrapper(onnx_path: str, output_path: str):
    """
    Create a standalone Python file that wraps the ONNX model.

    This creates a .py file compatible with atom_fight.py.

    Args:
        onnx_path: Path to ONNX model
        output_path: Where to save the wrapper .py file
    """
    wrapper_code = f'''"""
Trained Fighter - ONNX Wrapper

Auto-generated wrapper for trained ONNX model.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

# ONNX model path (relative to this file)
ONNX_MODEL = "{Path(onnx_path).name}"

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
        {{"acceleration": float, "stance": str}}
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
    outputs = session.run(output_names, {{input_name: obs}})

    # Parse action
    # Action space is Box: [acceleration_normalized, stance_selector]
    action = outputs[0][0]
    acceleration_normalized = np.clip(action[0], -1.0, 1.0)
    stance_idx = int(np.clip(action[1], 0, 3))

    # Scale acceleration (max_acceleration = 4.3751)
    acceleration = float(acceleration_normalized * 4.3751)
    stance = _stance_names[stance_idx]

    return {{"acceleration": acceleration, "stance": stance}}
'''

    with open(output_path, 'w') as f:
        f.write(wrapper_code)

    print(f"✓ Fighter wrapper saved: {output_path}")
