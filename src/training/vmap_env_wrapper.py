"""
Vectorized Environment Wrapper for JAX (Level 2 Optimization)

Wraps multiple Atom Combat environments into a single vectorized JAX environment
that runs all episodes in parallel using vmap.

This is the bridge between SBX and JAX vmap parallelization.
"""

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Tuple, List

from ..arena import WorldConfig, FighterState
from ..arena.arena_1d_jax_jit import (
    Arena1DJAXJit,
    FighterStateJAX,
    ArenaStateJAX,
    create_stance_arrays,
    stance_to_int
)


class VmapEnvWrapper(gym.Env):
    """
    Vectorized environment wrapper that runs N episodes in parallel using JAX vmap.

    This wrapper presents a Gym VecEnv-like interface while internally using
    JAX vmap for maximum parallelization efficiency.

    Key Features:
    - Runs 100+ episodes in parallel
    - All physics in JAX (JIT-compiled)
    - Minimal Python overhead
    - Compatible with SBX/SB3

    Usage:
        env = VmapEnvWrapper(
            n_envs=100,
            opponent_decision_func=opponent_func,
            config=WorldConfig()
        )

        obs = env.reset()
        for step in range(1000):
            actions = model.predict(obs)
            obs, rewards, dones, infos = env.step(actions)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int,
        opponent_decision_func: Callable,
        config: WorldConfig = None,
        max_ticks: int = 250,
        fighter_mass: float = 70.0,
        opponent_mass: float = 75.0,
        seed: int = 42
    ):
        """
        Initialize vectorized environment.

        Args:
            n_envs: Number of parallel environments (recommend 100-500)
            opponent_decision_func: Opponent decision function
            config: WorldConfig (uses default if None)
            max_ticks: Max ticks per episode
            fighter_mass: Fighter mass
            opponent_mass: Opponent mass
            seed: Random seed base
        """
        super().__init__()

        self.n_envs = n_envs
        self.opponent_decide = opponent_decision_func
        self.config = config or WorldConfig()
        self.max_ticks = max_ticks
        self.fighter_mass = fighter_mass
        self.opponent_mass = opponent_mass
        self.seed_base = seed

        # Define observation/action spaces (same as AtomCombatEnv)
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.99], dtype=np.float32),
            dtype=np.float32
        )

        # Pre-compute config values for JAX
        self.dt = self.config.dt
        self.max_accel = self.config.max_acceleration
        self.max_vel = self.config.max_velocity
        self.friction = self.config.friction
        self.arena_width = self.config.arena_width
        self.stamina_accel_cost = self.config.stamina_accel_cost
        self.stamina_base_regen = self.config.stamina_base_regen
        self.stamina_neutral_bonus = self.config.stamina_neutral_bonus

        # Pre-compute stance arrays
        self.stance_reach, self.stance_defense, self.stance_drain = create_stance_arrays(self.config)

        # Stance mapping
        self.stance_names = ["neutral", "extended", "retracted", "defending"]

        # JAX states for all environments
        self.jax_states = None
        self.tick_counts = None
        self.episode_rewards = None  # Track cumulative rewards per episode
        self.prev_fighter_hp = None  # Track HP for damage calculation
        self.prev_opponent_hp = None

        # Initialize environments
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset all environments in parallel.

        Returns:
            obs: [n_envs, obs_dim] numpy array
            info: dict
        """
        if seed is not None:
            self.seed_base = seed

        # Create initial states for all envs
        fighter = FighterState.create("learner", self.fighter_mass, 2.0, self.config)
        opponent = FighterState.create("opponent", self.opponent_mass, 10.0, self.config)

        jax_fighter = FighterStateJAX.from_fighter_state(fighter)
        jax_opponent = FighterStateJAX.from_fighter_state(opponent)

        # Create batch of initial states
        initial_states = []
        for i in range(self.n_envs):
            state = ArenaStateJAX(jax_fighter, jax_opponent, 0)
            initial_states.append(state)

        # Stack into batch using tree.map (JAX 0.6+ API)
        self.jax_states = jax.tree.map(lambda *xs: jnp.stack(xs), *initial_states)

        # Reset tick counts and episode rewards
        self.tick_counts = np.zeros(self.n_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(self.n_envs, dtype=np.float32)

        # Initialize HP tracking for damage calculation
        self.prev_fighter_hp = np.array(self.jax_states.fighter_a.hp, dtype=np.float32)
        self.prev_opponent_hp = np.array(self.jax_states.fighter_b.hp, dtype=np.float32)

        # Get initial observations
        obs = self._get_observations()

        return obs, {}

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel using vmap.

        Args:
            actions: [n_envs, 2] numpy array (acceleration, stance_selector)

        Returns:
            obs: [n_envs, obs_dim]
            rewards: [n_envs]
            dones: [n_envs]
            truncated: [n_envs]
            infos: list of dicts
        """
        # Convert actions to JAX format
        # actions[:, 0] = acceleration (-1 to 1)
        # actions[:, 1] = stance selector (0 to 3.99) -> int(stance)

        accel = jnp.array(actions[:, 0])
        stance_int = jnp.array(actions[:, 1].astype(np.int32))

        # Get opponent actions (for now, random or fixed)
        # TODO: Call opponent_decide for each env
        opponent_accel = jnp.zeros(self.n_envs)
        opponent_stance = jnp.zeros(self.n_envs, dtype=jnp.int32)  # Neutral

        # Stack actions
        actions_a = jnp.stack([accel, stance_int], axis=1)
        actions_b = jnp.stack([opponent_accel, opponent_stance], axis=1)

        # Step all environments in parallel with vmap
        new_states = self._vmap_step(
            self.jax_states,
            actions_a,
            actions_b
        )

        self.jax_states = new_states

        # Increment tick counts
        self.tick_counts += 1

        # Get observations
        obs = self._get_observations()

        # Calculate rewards (simplified for now)
        rewards = self._calculate_rewards()

        # Accumulate episode rewards
        self.episode_rewards += rewards

        # Check dones
        dones = self._check_dones()
        truncated = self.tick_counts >= self.max_ticks

        # Create infos with both 'r' (reward) and 'l' (length) for SB3 compatibility
        infos = []
        for i, (r, d, t) in enumerate(zip(rewards, dones, truncated)):
            if d or t:
                # Determine who won
                fighter_hp = float(self.jax_states.fighter_a.hp[i])
                opponent_hp = float(self.jax_states.fighter_b.hp[i])
                won = fighter_hp > opponent_hp

                # Episode ended - include cumulative episode stats
                infos.append({
                    "episode": {
                        "r": float(self.episode_rewards[i]),
                        "l": int(self.tick_counts[i])
                    },
                    "won": won,  # Add win/loss flag for curriculum trainer
                    "fighter_hp": fighter_hp,
                    "opponent_hp": opponent_hp
                })
                # Reset episode reward for this env
                self.episode_rewards[i] = 0.0
                self.tick_counts[i] = 0
            else:
                infos.append({})

        # Auto-reset finished environments
        # Reset individual environments that are done
        reset_mask = dones | truncated
        if np.any(reset_mask):
            self._reset_envs(reset_mask)

        return obs, rewards, dones, truncated, infos

    def _vmap_step(self, states, actions_a, actions_b):
        """Vectorized step across all environments."""

        def single_step(state, action_a, action_b):
            """Step a single environment."""
            # Explicitly cast stance to int32 (stack promotes to float32)
            action_a_dict = {"acceleration": action_a[0], "stance": jnp.int32(action_a[1])}
            action_b_dict = {"acceleration": action_b[0], "stance": jnp.int32(action_b[1])}

            new_state, _ = Arena1DJAXJit._jax_step_jit(
                state,
                action_a_dict,
                action_b_dict,
                self.dt,
                self.max_accel,
                self.max_vel,
                self.friction,
                self.arena_width,
                self.stamina_accel_cost,
                self.stamina_base_regen,
                self.stamina_neutral_bonus,
                self.stance_reach,
                self.stance_defense,
                self.stance_drain
            )
            return new_state

        # vmap across batch dimension
        return vmap(single_step)(states, actions_a, actions_b)

    def _get_observations(self):
        """Extract observations from JAX states."""
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
        relative_velocity = opponent_vel - fighter_vel

        # Normalize
        hp_norm = fighter_hp / fighter_max_hp
        stamina_norm = fighter_stamina / fighter_max_stamina
        opp_hp_norm = opponent_hp / opponent_max_hp
        opp_stamina_norm = opponent_stamina / opponent_max_stamina

        # Stack observations
        obs = np.stack([
            fighter_pos,
            fighter_vel,
            hp_norm,
            stamina_norm,
            distance,
            relative_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            np.full(self.n_envs, self.arena_width)
        ], axis=1).astype(np.float32)

        return obs

    def _calculate_rewards(self):
        """Calculate rewards based on damage differential."""
        # Get current HP
        fighter_hp = np.array(self.jax_states.fighter_a.hp, dtype=np.float32)
        opponent_hp = np.array(self.jax_states.fighter_b.hp, dtype=np.float32)

        # Calculate damage dealt and taken this step
        damage_dealt = self.prev_opponent_hp - opponent_hp
        damage_taken = self.prev_fighter_hp - fighter_hp

        # Reward is damage differential * 10 (matches gym_env.py)
        rewards = (damage_dealt - damage_taken) * 10.0

        # Update previous HP for next step
        self.prev_fighter_hp = fighter_hp.copy()
        self.prev_opponent_hp = opponent_hp.copy()

        return rewards.astype(np.float32)

    def _check_dones(self):
        """Check which environments are done."""
        fighter_hp = np.array(self.jax_states.fighter_a.hp)
        opponent_hp = np.array(self.jax_states.fighter_b.hp)

        dones = (fighter_hp <= 0) | (opponent_hp <= 0)

        return dones

    def _reset_envs(self, reset_mask):
        """Reset specific environments that are done.

        Args:
            reset_mask: Boolean array [n_envs] indicating which envs to reset
        """
        # Create fresh fighters for reset
        from ..arena import FighterState

        fighter = FighterState(
            name="fighter",
            position=0.0,
            velocity=0.0,
            hp=100.0,
            stamina=100.0,
            mass=self.fighter_mass,
            max_hp=100.0,
            max_stamina=100.0,
            stance="neutral"
        )

        opponent = FighterState(
            name="opponent",
            position=self.arena_width,
            velocity=0.0,
            hp=100.0,
            stamina=100.0,
            mass=self.opponent_mass,
            max_hp=100.0,
            max_stamina=100.0,
            stance="neutral"
        )

        jax_fighter = FighterStateJAX.from_fighter_state(fighter)
        jax_opponent = FighterStateJAX.from_fighter_state(opponent)
        fresh_state = ArenaStateJAX(jax_fighter, jax_opponent, 0)

        # Reset only the environments indicated by reset_mask
        # Use JAX tree operations to selectively update states
        def reset_if_mask(old_val, fresh_val, mask):
            """Replace old_val with fresh_val where mask is True."""
            mask_expanded = jnp.broadcast_to(mask.reshape(-1, 1), old_val.shape) if old_val.ndim > 1 else mask
            return jnp.where(mask_expanded, fresh_val, old_val)

        reset_mask_jax = jnp.array(reset_mask)

        # Reset each field of the state tree
        self.jax_states = jax.tree.map(
            lambda old, fresh: reset_if_mask(old, jnp.broadcast_to(fresh, old.shape), reset_mask_jax),
            self.jax_states,
            fresh_state
        )

        # Reset HP tracking for reset environments
        fresh_fighter_hp = 100.0
        fresh_opponent_hp = 100.0
        for i, should_reset in enumerate(reset_mask):
            if should_reset:
                self.prev_fighter_hp[i] = fresh_fighter_hp
                self.prev_opponent_hp[i] = fresh_opponent_hp


# Standalone test
if __name__ == "__main__":
    print("Testing VmapEnvWrapper...")

    def dummy_opponent(state):
        return {"acceleration": 0.0, "stance": "neutral"}

    env = VmapEnvWrapper(
        n_envs=100,
        opponent_decision_func=dummy_opponent,
        max_ticks=250
    )

    print(f"✅ Created env with {env.n_envs} parallel environments")

    obs, _ = env.reset()
    print(f"✅ Reset complete - obs shape: {obs.shape}")

    # Run a few steps
    for i in range(10):
        actions = env.action_space.sample()
        actions = np.tile(actions, (env.n_envs, 1))  # Replicate for all envs

        obs, rewards, dones, truncated, infos = env.step(actions)

        if i == 0:
            print(f"✅ Step complete - obs shape: {obs.shape}, rewards shape: {rewards.shape}")

    print(f"✅ All tests passed!")
    print(f"\nThis wrapper can now be used with SBX for vmap-accelerated training!")
