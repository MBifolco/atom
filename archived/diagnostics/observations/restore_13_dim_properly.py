#!/usr/bin/env python3
"""
Properly restore 13-dimensional observations to match the saved model.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Restoring 13-dimensional observations properly...")
print("=" * 60)

# First, update gym_env.py
gym_env_file = project_root / "src/training/gym_env.py"
content = gym_env_file.read_text()

# Restore 13-dim observation space
content_new = content.replace(
    """        # Define observation space (9 continuous values)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )""",
    """        # Define observation space (13 values for enhanced training)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width,
        #  wall_dist_left, wall_dist_right, opp_stance_int, recent_damage_dealt]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )""")

# Fix _get_observation to return 13 values
import re
obs_pattern = r'(    def _get_observation\(self\):.*?)(?=\n    def |\n\nclass |\Z)'
obs_replacement = '''    def _get_observation(self):
        """Get current observation as numpy array (13 dimensions)."""
        # Normalize HP and stamina to [0, 1] with division protection
        you_hp_norm = float(self.fighter.hp) / max(float(self.fighter.max_hp), 1.0)
        you_stamina_norm = float(self.fighter.stamina) / max(float(self.fighter.max_stamina), 1.0)
        opp_hp_norm = float(self.opponent.hp) / max(float(self.opponent.max_hp), 1.0)
        opp_stamina_norm = float(self.opponent.stamina) / max(float(self.opponent.max_stamina), 1.0)

        # Calculate opponent distance and relative velocity
        distance = float(abs(self.opponent.position - self.fighter.position))

        # Relative velocity (negative = approaching)
        if float(self.fighter.position) < float(self.opponent.position):
            rel_velocity = float(self.opponent.velocity) - float(self.fighter.velocity)
        else:
            rel_velocity = float(self.fighter.velocity) - float(self.opponent.velocity)

        # Wall distances
        wall_dist_left = float(self.fighter.position)
        wall_dist_right = float(self.config.arena_width) - float(self.fighter.position)

        # Opponent stance as integer
        opp_stance = self.opponent.stance
        if hasattr(opp_stance, '__array__'):  # JAX array
            opp_stance_int = float(opp_stance)
        else:  # String
            stance_map = {"neutral": 0, "extended": 1, "defending": 2}
            opp_stance_int = float(stance_map.get(opp_stance, 0))

        # Recent damage dealt (use episode tracking)
        recent_damage = float(self.episode_damage_dealt)

        obs = np.array([
            float(self.fighter.position),
            float(self.fighter.velocity),
            you_hp_norm,
            you_stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            float(self.config.arena_width),
            wall_dist_left,
            wall_dist_right,
            opp_stance_int,
            recent_damage
        ], dtype=np.float32)

        # Safety check: Replace any NaN or inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs'''

content_new = re.sub(obs_pattern, obs_replacement, content_new, flags=re.DOTALL)
gym_env_file.write_text(content_new)
print("✅ Fixed gym_env.py to use 13 dimensions")

# Now fix VmapEnvWrapper
vmap_file = project_root / "src/training/vmap_env_wrapper.py"
if vmap_file.exists():
    vmap_content = vmap_file.read_text()

    # Fix observation space
    vmap_content = vmap_content.replace(
        """        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )""",
        """        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )""")

    vmap_file.write_text(vmap_content)
    print("✅ Fixed VmapEnvWrapper observation space")

print("\n" + "=" * 60)
print("✅ Restored 13-dimensional observations!")
print("\nNow you need to also fix the _get_observations methods in VmapEnvWrapper")