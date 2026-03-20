#!/usr/bin/env python3
"""
Fix observation dimensions to use stable 9-dimensional format.
This should resolve the NaN issues by reverting to a known working configuration.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Fixing observation dimensions across all files...")
print("=" * 60)

# 1. Fix gym_env.py to use 9 dimensions
print("\n1. Fixing gym_env.py...")
gym_env_file = project_root / "src/training/gym_env.py"
gym_content = gym_env_file.read_text()

# First, fix the observation space definition
gym_content_new = gym_content.replace(
    """        # Define observation space (enhanced with 13 values)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width,
        #  wall_dist_left, wall_dist_right, opp_stance_int, recent_damage_dealt]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )""",
    """        # Define observation space (9 continuous values)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )""")

# Now fix the _get_observation method - find and replace the entire method
import re

# Pattern to match the _get_observation method
obs_pattern = r'(    def _get_observation\(self\):.*?)(?=\n    def |\n\nclass |\Z)'
obs_replacement = '''    def _get_observation(self):
        """Get current observation as numpy array (9 dimensions)."""
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

        obs = np.array([
            float(self.fighter.position),
            float(self.fighter.velocity),
            you_hp_norm,
            you_stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            float(self.config.arena_width)
        ], dtype=np.float32)

        # Safety check: Replace any NaN or inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs'''

gym_content_new = re.sub(obs_pattern, obs_replacement, gym_content_new, flags=re.DOTALL)

gym_env_file.write_text(gym_content_new)
print("   ✅ Fixed gym_env.py to use 9-dimensional observations")

# 2. Fix VmapEnvWrapper
print("\n2. Fixing VmapEnvWrapper...")
vmap_file = project_root / "src/training/vmap_env_wrapper.py"
if vmap_file.exists():
    vmap_content = vmap_file.read_text()

    # Fix observation space definition
    vmap_content = vmap_content.replace(
        """        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )""",
        """        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )""")

    vmap_file.write_text(vmap_content)
    print("   ✅ Fixed VmapEnvWrapper observation space")

# 3. Create a simplified VmapEnvWrapper _get_observations method
vmap_obs_fix = '''#!/usr/bin/env python3
"""
Patch for VmapEnvWrapper to use 9-dimensional observations.
"""

def create_9d_observations_patch():
    """
    Returns a method that creates 9-dimensional observations from JAX states.
    """
    def _get_observations(self):
        """Extract 9-dimensional observations from JAX states."""
        import numpy as np

        # Extract fighter and opponent states
        fighter_pos = np.array(self.jax_states.fighter_a.position)
        fighter_vel = np.array(self.jax_states.fighter_a.velocity)
        fighter_hp = np.array(self.jax_states.fighter_a.hp)
        fighter_stamina = np.array(self.jax_states.fighter_a.stamina)
        fighter_max_hp = np.array(self.jax_states.fighter_a.max_hp)
        fighter_max_stamina = np.array(self.jax_states.fighter_a.max_stamina)

        opponent_pos = np.array(self.jax_states.fighter_b.position)
        opponent_vel = np.array(self.jax_states.fighter_b.velocity)
        opponent_hp = np.array(self.jax_states.fighter_b.hp)
        opponent_stamina = np.array(self.jax_states.fighter_b.stamina)
        opponent_max_hp = np.array(self.jax_states.fighter_b.max_hp)
        opponent_max_stamina = np.array(self.jax_states.fighter_b.max_stamina)

        # Compute relative metrics
        distance = np.abs(opponent_pos - fighter_pos)

        # Relative velocity (negative = approaching)
        # Vectorized version for multiple environments
        rel_velocity = np.where(
            fighter_pos < opponent_pos,
            opponent_vel - fighter_vel,  # Fighter on left
            fighter_vel - opponent_vel   # Fighter on right
        )

        # Normalize with protection against division by zero
        hp_norm = fighter_hp / np.maximum(fighter_max_hp, 1.0)
        stamina_norm = fighter_stamina / np.maximum(fighter_max_stamina, 1.0)
        opp_hp_norm = opponent_hp / np.maximum(opponent_max_hp, 1.0)
        opp_stamina_norm = opponent_stamina / np.maximum(opponent_max_stamina, 1.0)

        # Stack observations (9 values)
        obs = np.stack([
            fighter_pos,
            fighter_vel,
            hp_norm,
            stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            np.full(self.n_envs, self.arena_width)
        ], axis=1).astype(np.float32)

        # Validate observations - replace any NaN or Inf with safe values
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️  WARNING: NaN or Inf detected in observations! Clipping...")
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs

    return _get_observations

# Similar for opponent observations
def create_9d_opponent_observations_patch():
    """
    Returns a method that creates 9-dimensional opponent observations.
    """
    def _get_opponent_observations(self):
        """Extract 9-dimensional observations from opponent's perspective."""
        import numpy as np

        # Extract states (opponent is fighter_b)
        opponent_pos = np.array(self.jax_states.fighter_b.position)
        opponent_vel = np.array(self.jax_states.fighter_b.velocity)
        opponent_hp = np.array(self.jax_states.fighter_b.hp)
        opponent_stamina = np.array(self.jax_states.fighter_b.stamina)
        opponent_max_hp = np.array(self.jax_states.fighter_b.max_hp)
        opponent_max_stamina = np.array(self.jax_states.fighter_b.max_stamina)

        fighter_pos = np.array(self.jax_states.fighter_a.position)
        fighter_vel = np.array(self.jax_states.fighter_a.velocity)
        fighter_hp = np.array(self.jax_states.fighter_a.hp)
        fighter_stamina = np.array(self.jax_states.fighter_a.stamina)
        fighter_max_hp = np.array(self.jax_states.fighter_a.max_hp)
        fighter_max_stamina = np.array(self.jax_states.fighter_a.max_stamina)

        # Compute relative metrics (from opponent's view)
        distance = np.abs(fighter_pos - opponent_pos)

        # Relative velocity from opponent's perspective
        rel_velocity = np.where(
            opponent_pos < fighter_pos,
            fighter_vel - opponent_vel,  # Opponent on left
            opponent_vel - fighter_vel   # Opponent on right
        )

        # Normalize with protection
        hp_norm = opponent_hp / np.maximum(opponent_max_hp, 1.0)
        stamina_norm = opponent_stamina / np.maximum(opponent_max_stamina, 1.0)
        opp_hp_norm = fighter_hp / np.maximum(fighter_max_hp, 1.0)
        opp_stamina_norm = fighter_stamina / np.maximum(fighter_max_stamina, 1.0)

        # Stack observations (9 values)
        obs = np.stack([
            opponent_pos,
            opponent_vel,
            hp_norm,
            stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            np.full(self.n_envs, self.arena_width)
        ], axis=1).astype(np.float32)

        # Validate
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️  WARNING: NaN or Inf in opponent observations! Clipping...")
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

        return obs

    return _get_opponent_observations
'''

patch_file = project_root / "archived/diagnostics/observations/vmap_obs_patch.py"
patch_file.write_text(vmap_obs_fix)
print("\n3. Created archived/diagnostics/observations/vmap_obs_patch.py with 9-dimensional observation methods")

print("\n" + "=" * 60)
print("✅ Observation dimensions fixed!")
print("\nIMPORTANT: You now need to:")
print("1. Apply the VmapEnvWrapper patch by updating the methods in vmap_env_wrapper.py")
print("2. Update any opponent decision functions to expect 9-dimensional observations")
print("3. Restart training with the fixed configuration")
print("\nThe 9-dimensional observation format is:")
print("  [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,")
print("   opp_hp_norm, opp_stamina_norm, arena_width]")
