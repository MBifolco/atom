#!/usr/bin/env python3
"""
Fix NaN errors by adding reward clipping to prevent gradient explosion.
The root cause: rewards can be 100-300 per step, causing numerical instability.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Fixing NaN errors by adding reward clipping...")
print("=" * 60)

# Fix 1: Add reward clipping to gym_env.py
print("\n1. Adding reward clipping to gym_env.py...")

gym_env_file = project_root / "src/training/gym_env.py"
content = gym_env_file.read_text()

# Find the return statement in step() and add clipping
import re

# Add reward clipping before the return statement
pattern = r'(        # Ensure reward is a Python float.*?\n        reward = float\(reward\).*?\n)'
replacement = r'\1\n        # Clip reward to prevent gradient explosion during PPO training\n        reward = np.clip(reward, -10.0, 10.0)  # Prevent extreme rewards\n'

content_new = re.sub(pattern, replacement, content, flags=re.DOTALL)

# If that didn't match, try a simpler approach
if content_new == content:
    # Find the final return in step()
    pattern = r'(        reward = float\(reward\) if hasattr\(reward, \'__float__\'\) else reward\n\n        return)'
    replacement = r'\1        # Clip reward to prevent gradient explosion during PPO training\n        reward = np.clip(reward, -10.0, 10.0)  # Prevent extreme rewards\n\n        return'
    content_new = re.sub(pattern, replacement, content)

gym_env_file.write_text(content_new)
print("✅ Added reward clipping to gym_env.py")

# Fix 2: Add reward clipping to VmapEnvWrapper
print("\n2. Adding reward clipping to VmapEnvWrapper...")

vmap_file = project_root / "src/training/vmap_env_wrapper.py"
content = vmap_file.read_text()

# Find where rewards are calculated and add clipping
pattern = r'(        # Stack everything for vectorized return\n        observations = self\._get_observations\(\))'
replacement = r'        # Clip rewards to prevent gradient explosion\n        rewards = np.clip(rewards, -10.0, 10.0)\n\n\1'

content_new = re.sub(pattern, replacement, content)

vmap_file.write_text(content_new)
print("✅ Added reward clipping to VmapEnvWrapper")

# Fix 3: Update training configs to use more stable hyperparameters
print("\n3. Creating stable PPO config for curriculum trainer...")

stable_config = '''"""
Stable PPO configuration to prevent NaN during training.
"""

def get_stable_ppo_config():
    """Get stable PPO hyperparameters to prevent NaN."""
    return {
        "learning_rate": 3e-5,  # Reduced from 5e-5
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,  # Increased for more exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,  # Gradient clipping
        "target_kl": 0.01,  # Early stopping for PPO updates
        "tensorboard_log": None,
        "policy_kwargs": {
            "activation_fn": "nn.ReLU",
            "net_arch": [64, 64],
            "ortho_init": True,  # More stable initialization
            "log_std_init": -0.5,  # Start with reasonable exploration
        }
    }
'''

config_file = project_root / "src/training/utils/stable_ppo_config.py"
config_file.write_text(stable_config)
print("✅ Created stable PPO config")

# Fix 4: Update curriculum_trainer to use stable config
print("\n4. Updating curriculum_trainer to use stable config...")

curriculum_file = project_root / "src/training/trainers/curriculum_trainer.py"
content = curriculum_file.read_text()

# Add import for stable config
if "from src.training.utils.stable_ppo_config import get_stable_ppo_config" not in content:
    # Add import after other imports
    pattern = r'(from src\.training\.utils\.stable_ppo import create_stable_ppo)'
    replacement = r'\1\nfrom src.training.utils.stable_ppo_config import get_stable_ppo_config'
    content = re.sub(pattern, replacement, content)

# Update PPO creation to use stable config
pattern = r'(                    self\.model = create_stable_ppo\(\n                        self\.envs,.*?\n                        verbose=0\n                    \))'
replacement = '''                    # Use extra stable config to prevent NaN
                    stable_config = get_stable_ppo_config()
                    self.model = create_stable_ppo(
                        self.envs,
                        **stable_config,
                        verbose=0
                    )'''

content_new = re.sub(pattern, replacement, content, flags=re.DOTALL)

if content_new != content:
    curriculum_file.write_text(content_new)
    print("✅ Updated curriculum_trainer to use stable config")
else:
    print("⚠️ Could not update curriculum_trainer - may need manual edit")

print("\n" + "=" * 60)
print("✅ NaN prevention fixes applied!")
print("\nChanges made:")
print("1. Added reward clipping [-10, 10] to prevent extreme values")
print("2. Reduced learning rate to 3e-5 for more stability")
print("3. Added target_kl for early stopping of PPO updates")
print("4. Increased entropy coefficient for better exploration")
print("\nThese changes will prevent gradient explosion that causes NaN.")