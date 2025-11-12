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
        opponent_mass: float = 70.0,
        seed: int = 42,
        debug: bool = False
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
        self.debug = debug

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

        # Calculate HP percentages
        fighter_hp_pct = fighter_hp / fighter_max_hp
        opponent_hp_pct = opponent_hp / opponent_max_hp

        # Calculate stamina percentages
        stamina_pct = fighter_stamina / fighter_max_stamina
        opp_stamina_pct = opponent_stamina / opponent_max_stamina

        # Get positions and calculate distance
        fighter_pos = np.array(self.jax_states.fighter_a.position, dtype=np.float32)
        opponent_pos = np.array(self.jax_states.fighter_b.position, dtype=np.float32)
        distance = np.abs(opponent_pos - fighter_pos)

        # Get current stance (convert from int to string for logic)
        # stance_int = np.array(self.jax_states.fighter_a.stance, dtype=np.int32)
        # For now, we'll work with stance indices: 0=neutral, 1=extended, 2=retracted, 3=defending

        # Initialize rewards array
        rewards = np.zeros(self.n_envs, dtype=np.float32)

        # === TERMINAL REWARDS (one fighter died) ===
        terminal_mask = dones & ~truncated
        if np.any(terminal_mask):
            # Win condition
            win_mask = terminal_mask & (fighter_hp_pct > opponent_hp_pct)
            if np.any(win_mask):
                # Base win reward + bonuses
                time_bonus = np.clip((self.max_ticks - self.tick_counts) * 0.5, 0, 100)
                hp_bonus = fighter_hp_pct * 100  # 0-100 based on remaining HP

                # Stamina efficiency bonus (using accumulated stamina usage like gym_env.py line 228)
                damage_per_stamina = self.episode_damage_dealt / np.maximum(self.episode_stamina_used, 1.0)
                stamina_efficiency = np.minimum(25, damage_per_stamina * 5)

                win_reward = 200.0 + time_bonus + hp_bonus + stamina_efficiency
                rewards = np.where(win_mask, win_reward, rewards)

            # Tie condition (both died simultaneously)
            tie_mask = terminal_mask & (fighter_hp_pct == opponent_hp_pct)
            rewards = np.where(tie_mask, -50.0, rewards)

            # Loss condition
            loss_mask = terminal_mask & (fighter_hp_pct < opponent_hp_pct)
            if np.any(loss_mask):
                hp_diff = opponent_hp_pct - fighter_hp_pct
                hp_penalty = hp_diff * 100
                loss_reward = -200.0 - hp_penalty
                rewards = np.where(loss_mask, loss_reward, rewards)

        # === TIMEOUT REWARDS (max ticks reached) ===
        timeout_mask = truncated & ~dones
        if np.any(timeout_mask):
            hp_pct_diff = fighter_hp_pct - opponent_hp_pct

            # Clear win (>10% HP margin)
            clear_win_mask = timeout_mask & (hp_pct_diff > 0.1)
            timeout_reward = 100.0 + (hp_pct_diff * 50)
            rewards = np.where(clear_win_mask, timeout_reward, rewards)

            # Slight win
            slight_win_mask = timeout_mask & (hp_pct_diff > 0) & (hp_pct_diff <= 0.1)
            rewards = np.where(slight_win_mask, 0.0, rewards)

            # Clear loss (<-10% HP margin)
            clear_loss_mask = timeout_mask & (hp_pct_diff < -0.1)
            timeout_reward = -100.0 + (hp_pct_diff * 50)
            rewards = np.where(clear_loss_mask, timeout_reward, rewards)

            # Slight loss
            slight_loss_mask = timeout_mask & (hp_pct_diff < 0) & (hp_pct_diff >= -0.1)
            rewards = np.where(slight_loss_mask, -50.0, rewards)

            # Exact tie
            exact_tie_mask = timeout_mask & (hp_pct_diff == 0)
            rewards = np.where(exact_tie_mask, -200.0, rewards)

        # === MID-EPISODE REWARDS (shaped rewards for learning) ===
        mid_episode_mask = ~dones & ~truncated
        if np.any(mid_episode_mask):
            mid_reward = np.zeros(self.n_envs, dtype=np.float32)

            # 1. Damage differential (core reward)
            damage_reward = (damage_dealt - damage_taken) * 10.0
            mid_reward += damage_reward
            self.episode_damage_reward += np.where(mid_episode_mask, damage_reward, 0.0)

            # 1b. Close-range engagement bonus
            close_range_mask = mid_episode_mask & (damage_dealt > 0) & (distance < self.arena_width * 0.3)
            close_range_bonus = damage_dealt * 2.0
            mid_reward += np.where(close_range_mask, close_range_bonus, 0.0)
            self.episode_damage_reward += np.where(close_range_mask, close_range_bonus, 0.0)

            # 2. Stamina-aware rewards
            # Bonus for stamina advantage
            stamina_adv_mask = mid_episode_mask & (stamina_pct > opp_stamina_pct + 0.2)
            stamina_bonus = 0.02
            mid_reward += np.where(stamina_adv_mask, stamina_bonus, 0.0)
            self.episode_stamina_reward += np.where(stamina_adv_mask, stamina_bonus, 0.0)

            # Penalty for fighting at very low stamina
            # Note: stance checking requires stance data - skip for now or use stance index
            low_stamina_mask = mid_episode_mask & (stamina_pct < 0.2)
            stamina_penalty = -0.05
            mid_reward += np.where(low_stamina_mask, stamina_penalty, 0.0)
            self.episode_stamina_reward += np.where(low_stamina_mask, stamina_penalty, 0.0)

            # 3. Smart proximity rewards (distance-aware)
            if self.last_distance is not None:
                distance_delta = self.last_distance - distance  # Positive = closing

                # Reward closing when opponent low HP or stamina
                closing_mask = mid_episode_mask & ((opponent_hp_pct < 0.3) | (opp_stamina_pct < 0.2)) & (distance_delta > 0.1)
                proximity_bonus = 0.2
                mid_reward += np.where(closing_mask, proximity_bonus, 0.0)
                self.episode_proximity_reward += np.where(closing_mask, proximity_bonus, 0.0)

                # Reward backing off when low stamina
                backing_mask = mid_episode_mask & (stamina_pct < 0.2) & (distance_delta < -0.1)
                proximity_bonus = 0.1
                mid_reward += np.where(backing_mask, proximity_bonus, 0.0)
                self.episode_proximity_reward += np.where(backing_mask, proximity_bonus, 0.0)

                # Normal engagement distance reward
                engagement_mask = mid_episode_mask & (distance < self.arena_width * 0.25)
                proximity_bonus = 0.1 * (1.0 - distance / (self.arena_width * 0.25))
                mid_reward += np.where(engagement_mask, proximity_bonus, 0.0)
                self.episode_proximity_reward += np.where(engagement_mask, proximity_bonus, 0.0)

            # 4. Stance-appropriate rewards (simplified - would need stance data)
            # Skipping for now as stance requires mapping from JAX state

            # 5. Inaction penalty (distance-aware)
            no_action_mask = mid_episode_mask & (damage_dealt == 0) & (damage_taken == 0)

            # Very close range
            close_inaction_mask = no_action_mask & (distance < self.arena_width * 0.2)
            mid_reward += np.where(close_inaction_mask, -0.5, 0.0)
            self.episode_inaction_penalty += np.where(close_inaction_mask, -0.5, 0.0)

            # Medium range
            medium_inaction_mask = no_action_mask & (distance >= self.arena_width * 0.2) & (distance < self.arena_width * 0.4)
            mid_reward += np.where(medium_inaction_mask, -0.3, 0.0)
            self.episode_inaction_penalty += np.where(medium_inaction_mask, -0.3, 0.0)

            # Far range
            far_inaction_mask = no_action_mask & (distance >= self.arena_width * 0.4)
            mid_reward += np.where(far_inaction_mask, -0.1, 0.0)
            self.episode_inaction_penalty += np.where(far_inaction_mask, -0.1, 0.0)

            # Apply mid-episode rewards only where not done/truncated
            rewards = np.where(mid_episode_mask, mid_reward, rewards)

        # Update tracking
        self.prev_fighter_hp = fighter_hp.copy()
        self.prev_opponent_hp = opponent_hp.copy()
        self.prev_fighter_stamina = fighter_stamina.copy()
        self.prev_opponent_stamina = opponent_stamina.copy()
        self.last_distance = distance.copy()

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
        from ..arena import FighterState

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

        # Note: last_distance doesn't need per-env reset, it's updated each step


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
