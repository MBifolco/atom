#!/usr/bin/env python3
"""
Revert to the stable 9-dimensional observation space while keeping other improvements.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Reverting to stable 9-dimensional observation space...")
print("=" * 60)

# Read gym_env.py
gym_env_file = project_root / "src/training/gym_env.py"
content = gym_env_file.read_text()

# Replace 13-dim observation space with 9-dim
old_obs_definition = """        # Define observation space (enhanced with 13 values)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width,
        #  wall_dist_left, wall_dist_right, opp_stance_int, recent_damage_dealt]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )"""

new_obs_definition = """        # Define observation space (9 continuous values)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )"""

if old_obs_definition in content:
    content = content.replace(old_obs_definition, new_obs_definition)
    print("✅ Updated observation space definition in gym_env.py")
else:
    print("⚠️  Could not find observation space definition to update")

# Now update the _get_observation method
print("\n2. Updating _get_observation method...")

# Save the updated content
gym_env_file.write_text(content)

print("\n" + "=" * 60)
print("✅ Reverted to stable configuration!")
print("\nNext steps:")
print("1. Update _get_observation() in gym_env.py to return 9 values")
print("2. Update VmapEnvWrapper observations to match")
print("3. Test the training again")