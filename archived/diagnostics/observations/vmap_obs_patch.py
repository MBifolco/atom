#!/usr/bin/env python3
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
