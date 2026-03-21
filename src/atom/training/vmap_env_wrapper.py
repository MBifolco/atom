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
from types import SimpleNamespace

from src.atom.runtime.arena import WorldConfig, FighterState
from src.atom.runtime.arena.arena_1d_jax_jit import (
    Arena1DJAXJit,
    FighterStateJAX,
    ArenaStateJAX,
    create_stance_arrays,
)
from .signal_engine import (
    build_observation_batch,
    compute_step_rewards_batch,
)
from src.atom.runtime.protocol import generate_snapshot


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
        opponent_decision_func: Callable = None,
        opponent_paths: List[str] = None,
        opponent_models: List = None,
        config: WorldConfig = None,
        max_ticks: int = 250,
        fighter_mass: float = 70.0,
        opponent_mass: float = 70.0,
        seed: int = 42,
        debug: bool = False
    ):
        """
        Initialize vectorized environment.

        Args:
            n_envs: Number of parallel environments (recommend 100-500)
            opponent_decision_func: Single opponent decision function (legacy)
            opponent_paths: List of opponent file paths for curriculum training
            opponent_models: List of trained SB3 models for population training
            config: WorldConfig (uses default if None)
            max_ticks: Max ticks per episode
            fighter_mass: Fighter mass
            opponent_mass: Opponent mass
            seed: Random seed base
        """
        super().__init__()

        self.n_envs = n_envs
        self.config = config or WorldConfig()
        self.max_ticks = max_ticks
        self.fighter_mass = fighter_mass
        self.opponent_mass = opponent_mass
        self.seed_base = seed
        self.debug = debug

        # Setup opponent system

        if opponent_models is not None and len(opponent_models) > 0:
            # Population training: Use trained models
            self.opponent_models = opponent_models
            self.opponent_paths = None
            self.opponent_decide = None
            self.use_multi_opponent = False
            self.use_opponent_models = True
        elif opponent_paths is not None and len(opponent_paths) > 0:
            # Curriculum training: Use JAX test dummies
            from .opponents_jax import create_multi_opponent_func
            self.opponent_decide = create_multi_opponent_func(opponent_paths, self.config)
            self.opponent_paths = opponent_paths
            self.opponent_models = None
            self.use_multi_opponent = True
            self.use_opponent_models = False
        else:
            # Legacy: single opponent
            self.opponent_decide = opponent_decision_func
            self.opponent_paths = None
            self.opponent_models = None
            self.use_multi_opponent = False
            self.use_opponent_models = False

        # Define observation/action spaces (enhanced to match AtomCombatEnv)
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.99], dtype=np.float32),  # 3 stances (0-2.99)
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

        # Hit/damage parameters
        self.hit_cooldown_ticks = self.config.hit_cooldown_ticks
        self.hit_impact_threshold = self.config.hit_impact_threshold
        self.base_damage = self.config.base_collision_damage
        self.hit_stamina_cost = self.config.hit_stamina_cost
        self.block_stamina_cost = self.config.block_stamina_cost
        self.hit_recoil_multiplier = self.config.hit_recoil_multiplier

        # Stance mapping (3-stance system)
        self.stance_names = ["neutral", "extended", "defending"]

        # JAX states for all environments
        self.jax_states = None
        self.tick_counts = None
        self.episode_rewards = None  # Track cumulative rewards per episode
        self.prev_fighter_hp = None  # Track HP for damage calculation
        self.prev_opponent_hp = None

        # Additional tracking for complete reward system
        self.last_distance = None  # For proximity rewards
        self.prev_fighter_stamina = None
        self.prev_opponent_stamina = None

        # Episode-level statistics
        self.episode_damage_dealt = None
        self.episode_damage_taken = None
        self.episode_stamina_used = None  # Track accumulated stamina usage

        # Episode-level reward breakdowns (for debugging/analysis)
        self.episode_damage_reward = None
        self.episode_proximity_reward = None
        self.episode_stamina_reward = None
        self.episode_stance_reward = None
        self.episode_inaction_penalty = None
        self.episode_terminal_reward = None

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

        # Create initial states for all envs (match gym_env.py)
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

        # Initialize stamina tracking
        self.prev_fighter_stamina = np.array(self.jax_states.fighter_a.stamina, dtype=np.float32)
        self.prev_opponent_stamina = np.array(self.jax_states.fighter_b.stamina, dtype=np.float32)

        # Initialize distance tracking (None = first step)
        self.last_distance = None

        # Initialize episode statistics
        self.episode_damage_dealt = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_damage_taken = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_stamina_used = np.zeros(self.n_envs, dtype=np.float32)

        # Initialize reward breakdowns
        self.episode_damage_reward = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_proximity_reward = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_stamina_reward = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_stance_reward = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_inaction_penalty = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_terminal_reward = np.zeros(self.n_envs, dtype=np.float32)

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
        # actions[:, 0] = acceleration (-1 to 1) - MUST SCALE like gym_env.py does!
        # actions[:, 1] = stance selector (0 to 3.99) -> int(stance)

        # Scale acceleration to match gym_env.py (line 171: acceleration_normalized * max_acceleration)
        accel = jnp.array(actions[:, 0]) * self.max_accel
        stance_int = jnp.array(actions[:, 1].astype(np.int32))

        # Get opponent actions
        if self.use_opponent_models:
            # Population training: Use trained models to predict opponent actions
            opponent_observations = self._get_opponent_observations()  # [n_envs, obs_dim]
            opponent_actions_np = self._predict_opponent_actions(opponent_observations)  # [n_envs, 2]
            opponent_accel = jnp.array(opponent_actions_np[:, 0]) * self.max_accel
            opponent_stance = jnp.array(opponent_actions_np[:, 1].astype(np.int32))
        elif self.use_multi_opponent:
            # Curriculum training: Call batched JAX opponent functions
            env_indices = jnp.arange(self.n_envs)
            opponent_actions = self.opponent_decide(self.jax_states, env_indices)
            opponent_accel = opponent_actions[:, 0]
            opponent_stance = opponent_actions[:, 1].astype(jnp.int32)
        else:
            # Legacy single-opponent function path (used in unit/integration tests and some local runs).
            opponent_accel_np, opponent_stance_np = self._compute_legacy_opponent_actions()
            opponent_accel = jnp.array(opponent_accel_np, dtype=jnp.float32)
            opponent_stance = jnp.array(opponent_stance_np, dtype=jnp.int32)

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

        # Debug logging for first environment
        if self.debug and self.tick_counts[0] % 50 == 0:
            i = 0  # Log first env
            print(f"\n=== DEBUG: Env 0, Tick {self.tick_counts[i]} ===")
            print(f"Actions: accel={actions[i, 0]:.2f}, stance={int(actions[i, 1])}")
            print(f"Fighter: pos={self.jax_states.fighter_a.position[i]:.2f}, vel={self.jax_states.fighter_a.velocity[i]:.2f}, HP={self.jax_states.fighter_a.hp[i]:.1f}, stamina={self.jax_states.fighter_a.stamina[i]:.1f}")
            print(f"Opponent: pos={self.jax_states.fighter_b.position[i]:.2f}, vel={self.jax_states.fighter_b.velocity[i]:.2f}, HP={self.jax_states.fighter_b.hp[i]:.1f}, stamina={self.jax_states.fighter_b.stamina[i]:.1f}")
            print(f"Distance: {abs(self.jax_states.fighter_b.position[i] - self.jax_states.fighter_a.position[i]):.2f}")

        # Get observations
        obs = self._get_observations()

        # Check dones and truncated BEFORE calculating rewards
        dones = self._check_dones()
        truncated = self.tick_counts >= self.max_ticks

        # Calculate rewards with complete reward system
        rewards = self._calculate_rewards(dones, truncated)

        # Debug reward breakdown
        if self.debug and (dones[0] or truncated[0]):
            i = 0
            print(f"\n=== EPISODE END: Env 0 ===")
            print(f"Done: {dones[i]}, Truncated: {truncated[i]}, Ticks: {self.tick_counts[i]}")
            print(f"Final HP - Fighter: {self.jax_states.fighter_a.hp[i]:.1f}, Opponent: {self.jax_states.fighter_b.hp[i]:.1f}")
            print(f"Episode Damage - Dealt: {self.episode_damage_dealt[i]:.1f}, Taken: {self.episode_damage_taken[i]:.1f}")
            print(f"Reward Breakdown:")
            print(f"  Damage rewards: {self.episode_damage_reward[i]:.2f}")
            print(f"  Proximity rewards: {self.episode_proximity_reward[i]:.2f}")
            print(f"  Stamina rewards: {self.episode_stamina_reward[i]:.2f}")
            print(f"  Stance rewards: {self.episode_stance_reward[i]:.2f}")
            print(f"  Inaction penalties: {self.episode_inaction_penalty[i]:.2f}")
            print(f"  Terminal reward: {rewards[i]:.2f}")
            print(f"  Total episode reward: {self.episode_rewards[i] + rewards[i]:.2f}")

        # Accumulate episode rewards
        self.episode_rewards += rewards

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
                    "opponent_hp": opponent_hp,
                    "reward_breakdown": {
                        "proximity": float(self.episode_proximity_reward[i]),
                        "damage": float(self.episode_damage_reward[i]),
                        "stamina": float(self.episode_stamina_reward[i]),
                        "stance": float(self.episode_stance_reward[i]),
                        "inaction": float(self.episode_inaction_penalty[i]),
                        "terminal": float(self.episode_terminal_reward[i]),
                        "total": float(
                            self.episode_proximity_reward[i]
                            + self.episode_damage_reward[i]
                            + self.episode_stamina_reward[i]
                            + self.episode_stance_reward[i]
                            + self.episode_inaction_penalty[i]
                            + self.episode_terminal_reward[i]
                        ),
                    },
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

        # Validate observations - replace any NaN or Inf with zeros to prevent training crashes
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️  WARNING: NaN or Inf detected in observations! Clipping...")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0)

        # Clip rewards to prevent gradient explosion
        rewards = np.clip(rewards, -1000.0, 1000.0)  # Safety clip only for extreme outliers

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
                self.stance_drain,
                self.hit_cooldown_ticks,
                self.hit_impact_threshold,
                self.base_damage,
                self.hit_stamina_cost,
                self.block_stamina_cost,
                self.hit_recoil_multiplier
            )
            return new_state

        # vmap across batch dimension
        return vmap(single_step)(states, actions_a, actions_b)

    def _get_observations(self):
        """Extract enhanced observations from JAX states."""
        return build_observation_batch(
            you_position=np.array(self.jax_states.fighter_a.position),
            you_velocity=np.array(self.jax_states.fighter_a.velocity),
            you_hp=np.array(self.jax_states.fighter_a.hp),
            you_max_hp=np.array(self.jax_states.fighter_a.max_hp),
            you_stamina=np.array(self.jax_states.fighter_a.stamina),
            you_max_stamina=np.array(self.jax_states.fighter_a.max_stamina),
            opponent_position=np.array(self.jax_states.fighter_b.position),
            opponent_velocity=np.array(self.jax_states.fighter_b.velocity),
            opponent_hp=np.array(self.jax_states.fighter_b.hp),
            opponent_max_hp=np.array(self.jax_states.fighter_b.max_hp),
            opponent_stamina=np.array(self.jax_states.fighter_b.stamina),
            opponent_max_stamina=np.array(self.jax_states.fighter_b.max_stamina),
            opponent_stance=np.array(self.jax_states.fighter_b.stance),
            arena_width=self.arena_width,
            recent_damage=np.array(self.episode_damage_dealt),
        )

    def _get_opponent_observations(self):
        """Extract observations from opponent's perspective for model predictions."""
        return build_observation_batch(
            you_position=np.array(self.jax_states.fighter_b.position),
            you_velocity=np.array(self.jax_states.fighter_b.velocity),
            you_hp=np.array(self.jax_states.fighter_b.hp),
            you_max_hp=np.array(self.jax_states.fighter_b.max_hp),
            you_stamina=np.array(self.jax_states.fighter_b.stamina),
            you_max_stamina=np.array(self.jax_states.fighter_b.max_stamina),
            opponent_position=np.array(self.jax_states.fighter_a.position),
            opponent_velocity=np.array(self.jax_states.fighter_a.velocity),
            opponent_hp=np.array(self.jax_states.fighter_a.hp),
            opponent_max_hp=np.array(self.jax_states.fighter_a.max_hp),
            opponent_stamina=np.array(self.jax_states.fighter_a.stamina),
            opponent_max_stamina=np.array(self.jax_states.fighter_a.max_stamina),
            opponent_stance=np.array(self.jax_states.fighter_a.stance),
            arena_width=self.arena_width,
            recent_damage=np.array(self.episode_damage_taken),
        )

    def _predict_opponent_actions(self, opponent_observations):
        """
        Predict opponent actions using trained models.

        Distributes environments across population models and batches predictions on GPU.

        Args:
            opponent_observations: [n_envs, obs_dim] numpy array

        Returns:
            actions: [n_envs, 2] numpy array (acceleration, stance_selector)
        """
        n_models = len(self.opponent_models)
        envs_per_model = self.n_envs // n_models

        all_actions = np.zeros((self.n_envs, 2), dtype=np.float32)

        # Predict in batches for each model
        for i, model in enumerate(self.opponent_models):
            start_idx = i * envs_per_model
            # Last model handles remaining envs
            end_idx = (i + 1) * envs_per_model if i < n_models - 1 else self.n_envs

            if start_idx >= self.n_envs:
                break

            batch_obs = opponent_observations[start_idx:end_idx]

            # Predict actions (SB3 models can use GPU internally)
            # predict() returns (actions, _states) but we only need actions
            actions, _ = model.predict(batch_obs, deterministic=False)
            all_actions[start_idx:end_idx] = actions

        return all_actions

    def _compute_legacy_opponent_actions(self):
        """
        Compute opponent actions via legacy `opponent_decision_func`.

        This keeps behavior aligned with AtomCombatEnv when callers pass
        a Python decision function instead of opponent_paths/opponent_models.
        """
        if self.opponent_decide is None:
            return np.zeros(self.n_envs, dtype=np.float32), np.zeros(self.n_envs, dtype=np.int32)

        accel = np.zeros(self.n_envs, dtype=np.float32)
        stance = np.zeros(self.n_envs, dtype=np.int32)

        for i in range(self.n_envs):
            fighter_a = SimpleNamespace(
                position=float(self.jax_states.fighter_a.position[i]),
                velocity=float(self.jax_states.fighter_a.velocity[i]),
                hp=float(self.jax_states.fighter_a.hp[i]),
                max_hp=float(self.jax_states.fighter_a.max_hp[i]),
                stamina=float(self.jax_states.fighter_a.stamina[i]),
                max_stamina=float(self.jax_states.fighter_a.max_stamina[i]),
                stance=int(self.jax_states.fighter_a.stance[i]),
            )
            fighter_b = SimpleNamespace(
                position=float(self.jax_states.fighter_b.position[i]),
                velocity=float(self.jax_states.fighter_b.velocity[i]),
                hp=float(self.jax_states.fighter_b.hp[i]),
                max_hp=float(self.jax_states.fighter_b.max_hp[i]),
                stamina=float(self.jax_states.fighter_b.stamina[i]),
                max_stamina=float(self.jax_states.fighter_b.max_stamina[i]),
                stance=int(self.jax_states.fighter_b.stance[i]),
            )

            # Opponent controls fighter_b, observing fighter_a as "opponent".
            snapshot = generate_snapshot(
                fighter_b,
                fighter_a,
                int(self.tick_counts[i]),
                float(self.arena_width),
            )
            action = self.opponent_decide(snapshot) or {}
            accel[i] = float(np.clip(action.get("acceleration", 0.0), -self.max_accel, self.max_accel))

            stance_value = action.get("stance", "neutral")
            if isinstance(stance_value, str):
                if stance_value in self.stance_names:
                    stance[i] = self.stance_names.index(stance_value)
                else:
                    stance[i] = 0
            else:
                stance[i] = int(np.clip(int(stance_value), 0, 2))

        return accel, stance

    def _calculate_rewards(self, dones, truncated):
        """
        Calculate rewards using the complete reward system from gym_env.py.

        Handles terminal rewards, timeout rewards, and mid-episode shaped rewards.
        """
        # Get current state
        fighter_hp = np.array(self.jax_states.fighter_a.hp, dtype=np.float32)
        opponent_hp = np.array(self.jax_states.fighter_b.hp, dtype=np.float32)
        fighter_stamina = np.array(self.jax_states.fighter_a.stamina, dtype=np.float32)
        opponent_stamina = np.array(self.jax_states.fighter_b.stamina, dtype=np.float32)
        fighter_max_hp = np.array(self.jax_states.fighter_a.max_hp, dtype=np.float32)
        opponent_max_hp = np.array(self.jax_states.fighter_b.max_hp, dtype=np.float32)
        fighter_max_stamina = np.array(self.jax_states.fighter_a.max_stamina, dtype=np.float32)
        opponent_max_stamina = np.array(self.jax_states.fighter_b.max_stamina, dtype=np.float32)

        # Calculate damage dealt and taken this step
        damage_dealt = self.prev_opponent_hp - opponent_hp
        damage_taken = self.prev_fighter_hp - fighter_hp

        # Track episode damage (accumulate before calculating rewards)
        self.episode_damage_dealt += damage_dealt
        self.episode_damage_taken += damage_taken

        # Track stamina usage (like gym_env.py line 196)
        stamina_spent = self.prev_fighter_stamina - fighter_stamina
        self.episode_stamina_used += np.maximum(0, stamina_spent)

        # Calculate HP percentages with division protection
        fighter_hp_pct = fighter_hp / np.maximum(fighter_max_hp, 1.0)
        opponent_hp_pct = opponent_hp / np.maximum(opponent_max_hp, 1.0)

        # Calculate stamina percentages with division protection
        stamina_pct = fighter_stamina / np.maximum(fighter_max_stamina, 1.0)
        opp_stamina_pct = opponent_stamina / np.maximum(opponent_max_stamina, 1.0)

        # Get positions and calculate distance
        fighter_pos = np.array(self.jax_states.fighter_a.position, dtype=np.float32)
        opponent_pos = np.array(self.jax_states.fighter_b.position, dtype=np.float32)
        distance = np.abs(opponent_pos - fighter_pos)

        reward_result = compute_step_rewards_batch(
            dones=dones,
            truncated=truncated,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            fighter_hp_pct=fighter_hp_pct,
            opponent_hp_pct=opponent_hp_pct,
            stamina_pct=stamina_pct,
            opp_stamina_pct=opp_stamina_pct,
            fighter_stance=np.array(self.jax_states.fighter_a.stance, dtype=np.int32),
            distance=distance,
            last_distance=self.last_distance,
            tick_counts=self.tick_counts,
            max_ticks=self.max_ticks,
            arena_width=self.arena_width,
            episode_damage_dealt=self.episode_damage_dealt,
            episode_stamina_used=self.episode_stamina_used,
        )

        rewards = reward_result.rewards
        self.episode_damage_reward += reward_result.damage_component
        self.episode_proximity_reward += reward_result.proximity_component
        self.episode_stamina_reward += reward_result.stamina_component
        self.episode_stance_reward += reward_result.stance_component
        self.episode_inaction_penalty += reward_result.inaction_component
        self.episode_terminal_reward += reward_result.terminal_component

        # Update tracking
        self.prev_fighter_hp = fighter_hp.copy()
        self.prev_opponent_hp = opponent_hp.copy()
        self.prev_fighter_stamina = fighter_stamina.copy()
        self.prev_opponent_stamina = opponent_stamina.copy()
        self.last_distance = reward_result.next_last_distance.copy()

        # Validate rewards - replace any NaN or Inf with zeros
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print(f"⚠️  WARNING: NaN or Inf detected in rewards! Clipping...")
            print(f"   NaN locations: {np.where(np.isnan(rewards))}")
            print(f"   Inf locations: {np.where(np.isinf(rewards))}")
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=1000.0, neginf=-1000.0)

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
        # Create fresh fighters for reset (match gym_env.py)
        from src.atom.runtime.arena import FighterState

        fighter = FighterState.create("fighter", self.fighter_mass, 2.0, self.config)
        opponent = FighterState.create("opponent", self.opponent_mass, 10.0, self.config)

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

        # Reset HP and stamina tracking for reset environments
        fresh_fighter_hp = fighter.hp
        fresh_opponent_hp = opponent.hp
        fresh_fighter_stamina = fighter.stamina
        fresh_opponent_stamina = opponent.stamina
        for i, should_reset in enumerate(reset_mask):
            if should_reset:
                self.prev_fighter_hp[i] = fresh_fighter_hp
                self.prev_opponent_hp[i] = fresh_opponent_hp
                self.prev_fighter_stamina[i] = fresh_fighter_stamina
                self.prev_opponent_stamina[i] = fresh_opponent_stamina

                # Reset episode statistics
                self.episode_damage_dealt[i] = 0.0
                self.episode_damage_taken[i] = 0.0
                self.episode_stamina_used[i] = 0.0

                # Reset reward breakdowns
                self.episode_damage_reward[i] = 0.0
                self.episode_proximity_reward[i] = 0.0
                self.episode_stamina_reward[i] = 0.0
                self.episode_stance_reward[i] = 0.0
                self.episode_inaction_penalty[i] = 0.0
                self.episode_terminal_reward[i] = 0.0

        # Note: last_distance doesn't need per-env reset, it's updated each step

    def close(self):
        """Clean up JAX resources and free memory."""
        # Clear JAX states
        if hasattr(self, 'jax_states') and self.jax_states is not None:
            del self.jax_states
            self.jax_states = None

        # Clear tracking arrays
        if hasattr(self, 'tick_counts') and self.tick_counts is not None:
            del self.tick_counts
            self.tick_counts = None

        if hasattr(self, 'episode_rewards') and self.episode_rewards is not None:
            del self.episode_rewards
            self.episode_rewards = None

        # Clear HP/stamina tracking
        if hasattr(self, 'prev_fighter_hp'):
            del self.prev_fighter_hp
            self.prev_fighter_hp = None

        if hasattr(self, 'prev_opponent_hp'):
            del self.prev_opponent_hp
            self.prev_opponent_hp = None

        if hasattr(self, 'prev_fighter_stamina'):
            del self.prev_fighter_stamina
            self.prev_fighter_stamina = None

        if hasattr(self, 'prev_opponent_stamina'):
            del self.prev_opponent_stamina
            self.prev_opponent_stamina = None

        # Clear opponent decision function
        if hasattr(self, 'opponent_decide') and self.opponent_decide is not None:
            del self.opponent_decide
            self.opponent_decide = None

        # Clear stance arrays
        if hasattr(self, 'stance_reach'):
            del self.stance_reach
            self.stance_reach = None

        if hasattr(self, 'stance_defense'):
            del self.stance_defense
            self.stance_defense = None

        if hasattr(self, 'stance_drain'):
            del self.stance_drain
            self.stance_drain = None

        # Clear all episode tracking arrays
        for attr in ['episode_damage_dealt', 'episode_damage_taken', 'episode_stamina_used',
                     'episode_damage_reward', 'episode_proximity_reward',
                     'episode_stamina_reward', 'episode_stance_reward',
                     'episode_inaction_penalty', 'episode_terminal_reward', 'last_distance',
                     'hits_landed', 'hits_taken']:
            if hasattr(self, attr):
                delattr(self, attr)

        # Clear JAX PRNG key
        if hasattr(self, 'rng'):
            del self.rng
            self.rng = None

        # Force garbage collection of JAX arrays
        import gc
        gc.collect()

        # Clear JAX compilation cache if possible
        try:
            import jax
            jax.clear_caches()
        except:
            pass


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
