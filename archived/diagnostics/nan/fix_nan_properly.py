#!/usr/bin/env python3
"""
Properly fix NaN errors without destroying the reward signal.
Instead of hard clipping, use proper reward normalization.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Fixing NaN errors properly (removing hard clipping)...")
print("=" * 60)

# Remove the hard clipping from gym_env.py
print("\n1. Removing hard reward clipping from gym_env.py...")

gym_env_file = project_root / "src/training/gym_env.py"
content = gym_env_file.read_text()

# Remove the clipping we added
import re
pattern = r'\n        # Clip reward to prevent gradient explosion during PPO training\n        reward = np\.clip\(reward, -10\.0, 10\.0\)  # Prevent extreme rewards\n'
content_new = re.sub(pattern, '', content)

if content_new != content:
    gym_env_file.write_text(content_new)
    print("✅ Removed hard clipping from gym_env.py")
else:
    print("⚠️ No hard clipping found in gym_env.py")

# Remove the hard clipping and restore original rewards in VmapEnvWrapper
print("\n2. Restoring original rewards in VmapEnvWrapper...")

vmap_file = project_root / "src/training/vmap_env_wrapper.py"
content = vmap_file.read_text()

# Restore original reward values (but keep the -10, 10 clipping as a safety net at 100x scale)
replacements = [
    # Restore win reward
    (r'                win_reward = 10\.0  # Simplified large win reward, will be clipped anyway',
     '                win_reward = 200.0 + time_bonus + hp_bonus + stamina_efficiency'),

    # Restore tie penalty
    (r'            rewards = np\.where\(tie_mask, -5\.0, rewards\)',
     '            rewards = np.where(tie_mask, -50.0, rewards)'),

    # Restore loss penalty
    (r'                loss_reward = -10\.0  # Simplified large loss penalty, will be clipped anyway',
     '                hp_diff = opponent_hp_pct - fighter_hp_pct\n' +
     '                hp_penalty = hp_diff * 100\n' +
     '                loss_reward = -200.0 - hp_penalty'),

    # Restore timeout rewards
    (r'            timeout_reward = 5\.0 \+ \(hp_pct_diff \* 2\.0\)  # Scaled down',
     '            timeout_reward = 100.0 + (hp_pct_diff * 50)'),

    (r'            timeout_reward = -5\.0 \+ \(hp_pct_diff \* 2\.0\)  # Scaled down',
     '            timeout_reward = -100.0 + (hp_pct_diff * 50)'),

    (r'            rewards = np\.where\(slight_loss_mask, -2\.0, rewards\)  # Scaled down',
     '            rewards = np.where(slight_loss_mask, -50.0, rewards)'),

    (r'            rewards = np\.where\(exact_tie_mask, -10\.0, rewards\)  # Scaled down',
     '            rewards = np.where(exact_tie_mask, -200.0, rewards)'),

    # Restore damage rewards
    (r'            damage_reward = \(damage_dealt - damage_taken\) \* 0\.5  # Was 10\.0, now 0\.5',
     '            damage_reward = (damage_dealt - damage_taken) * 10.0'),

    (r'            close_range_bonus = damage_dealt \* 0\.1  # Was 2\.0, now 0\.1 - small bonus for close hits',
     '            close_range_bonus = damage_dealt * 2.0  # Double damage reward for close hits'),

    # Change hard clipping to safety clipping at reasonable range
    (r'        rewards = np\.clip\(rewards, -10\.0, 10\.0\)',
     '        rewards = np.clip(rewards, -1000.0, 1000.0)  # Safety clip only for extreme outliers')
]

for old, new in replacements:
    content = re.sub(old, new, content)

vmap_file.write_text(content)
print("✅ Restored original reward values in VmapEnvWrapper")

# Now update curriculum_trainer to use VecNormalize wrapper
print("\n3. Updating curriculum_trainer to use VecNormalize for proper reward scaling...")

curriculum_file = project_root / "src/training/trainers/curriculum_trainer.py"
content = curriculum_file.read_text()

# Add VecNormalize import
if "from stable_baselines3.common.vec_env import VecNormalize" not in content:
    pattern = r'(from stable_baselines3\.common\.vec_env import DummyVecEnv, SubprocVecEnv)'
    replacement = r'\1, VecNormalize'
    content = re.sub(pattern, replacement, content)

# Wrap the environment with VecNormalize before creating PPO
pattern = r'(            self\.envs = DummyVecEnv\(\[make_env\(\) for _ in range\(self\.n_envs\)\]\))'
replacement = '''            self.envs = DummyVecEnv([make_env() for _ in range(self.n_envs)])

            # Wrap with VecNormalize to handle large reward scales
            # This normalizes rewards to have mean=0, std=1 while preserving relative differences
            self.envs = VecNormalize(
                self.envs,
                norm_obs=False,  # Don't normalize observations (they're already normalized)
                norm_reward=True,  # Normalize rewards to prevent gradient explosion
                clip_obs=10.0,  # Safety clip for observations
                clip_reward=10.0,  # Clip normalized rewards to [-10, 10] std deviations
                gamma=0.99  # Same as PPO gamma
            )'''

content = re.sub(pattern, replacement, content)

# Do the same for VmapEnvWrapper
pattern = r'(                self\.envs = vmap_env)'
replacement = '''                self.envs = vmap_env

                # Wrap with VecNormalize to handle large reward scales
                from stable_baselines3.common.vec_env import VecNormalize
                self.envs = VecNormalize(
                    self.envs,
                    norm_obs=False,  # Don't normalize observations
                    norm_reward=True,  # Normalize rewards
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=0.99
                )'''

content = re.sub(pattern, replacement, content)

curriculum_file.write_text(content)
print("✅ Added VecNormalize wrapper to curriculum_trainer")

print("\n" + "=" * 60)
print("✅ Proper fix applied!")
print("\nWhat this does:")
print("1. Removes hard reward clipping that destroys the training signal")
print("2. Restores original reward design (-200 to +200 range)")
print("3. Uses VecNormalize to properly scale rewards while preserving relative differences")
print("\nVecNormalize:")
print("- Maintains running mean/std of rewards")
print("- Normalizes rewards to mean=0, std=1")
print("- Preserves the relative differences between rewards")
print("- Is the standard RL approach for handling varying reward scales")