"""
Gymnasium Environment Wrapper for Atom Combat

Wraps the Atom Combat arena as a Gym environment for RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Use relative imports within the src package
from src.atom.runtime.arena import WorldConfig, FighterState, Arena1DJAXJit
from src.atom.runtime.protocol import generate_snapshot
from .signal_engine import build_observation, compute_step_reward_scalar, hp_pct, ObservationBuilder
from .action_codec import (
    ACTION_SPACE_LOW, ACTION_SPACE_HIGH,
    extract_stance, scale_and_validate_action,
    STANCE_NAMES, stance_idx_to_str, stance_str_to_idx,
)


class AtomCombatEnv(gym.Env):
    """
    Gym environment for training Atom Combat fighters.

    Observation Space:
        - you_position: float
        - you_velocity: float
        - you_hp: float (normalized 0-1)
        - you_stamina: float (normalized 0-1)
        - opponent_distance: float
        - opponent_velocity: float (relative)
        - opponent_hp: float (normalized 0-1)
        - opponent_stamina: float (normalized 0-1)
        - arena_width: float

    Action Space:
        - acceleration: continuous [-1, 1] (scaled to max_acceleration)
        - stance: discrete [0-2] (neutral, extended, defending)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_decision_func,
        config: WorldConfig = None,
        max_ticks: int = 250,
        fighter_mass: float = 70.0,
        opponent_mass: float = 70.0,
        seed: int = None,
        reward_weights: dict = None,
        opponent_name: str = None,
        use_history: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            opponent_decision_func: Decision function for opponent
            config: WorldConfig instance (uses default if None)
            max_ticks: Maximum ticks before timeout
            fighter_mass: Mass of the learning fighter
            opponent_mass: Mass of the opponent
            seed: Random seed
        """
        super().__init__()

        self.config = config or WorldConfig()
        self.opponent_decide = opponent_decision_func
        self.max_ticks = max_ticks
        self.fighter_mass = fighter_mass
        self.opponent_mass = opponent_mass
        self._seed = seed
        self.reward_weights = reward_weights
        self.opponent_name = opponent_name

        # Observation space: 26D default (18D base + 8D EMA), 666D with history
        self.use_history = use_history
        base_low = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
        base_high = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ema_low = [0, -1, -1, 0, -1, -1, 0, 0]
        ema_high = [1, 1, 1, 1, 1, 1, 1, 1]
        obs_low = base_low + ema_low
        obs_high = base_high + ema_high
        if use_history:
            from .signal_engine import HISTORY_OBS_DIM
            obs_low += [-1.0] * HISTORY_OBS_DIM
            obs_high += [1.0] * HISTORY_OBS_DIM
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            dtype=np.float32
        )

        # Action space: [acceleration, stance_selector]
        # Stance selected via bin lookup on selector — see action_codec.py
        self.action_space = spaces.Box(
            low=ACTION_SPACE_LOW,
            high=ACTION_SPACE_HIGH,
            dtype=np.float32
        )

        # Stance mapping (3-stance system) — uses STANCE_NAMES from action_codec
        self.stance_names = STANCE_NAMES

        # State
        self.arena = None
        self.tick = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0
        self.last_distance = None  # initialized to real distance in reset()
        self.stamina_used = 0
        self.obs_builder = ObservationBuilder(n_envs=1, use_history=use_history)
        self._temporal_features = None
        self._history_flat = None
        self.hits_landed = 0
        self.hits_taken = 0

        # Stance usage tracking (per-episode tick counts)
        self.stance_ticks = [0, 0, 0]  # neutral, extended, defending

        # Reward component tracking
        self.episode_proximity_reward = 0
        self.episode_damage_reward = 0
        self.episode_inaction_penalty = 0
        self.episode_terminal_reward = 0
        self.episode_stamina_reward = 0
        self.episode_stance_reward = 0

    @property
    def fighter(self):
        """Get current fighter state from arena."""
        if self.arena is None:
            return None
        return self.arena.state.fighter_a

    @property
    def opponent(self):
        """Get current opponent state from arena."""
        if self.arena is None:
            return None
        return self.arena.state.fighter_b

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Create initial fighters
        fighter_init = FighterState.create("learner", self.fighter_mass, 2.0, self.config)
        opponent_init = FighterState.create("opponent", self.opponent_mass, 10.0, self.config)

        # Create arena (JAX JIT > JAX > Python)
        # Always use JAX JIT implementation
        self.arena = Arena1DJAXJit(fighter_init, opponent_init, self.config, seed=self._seed or 0)

        self.tick = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0
        self.last_distance = float(abs(self.fighter.position - self.opponent.position))
        self.stamina_used = 0
        self.hits_landed = 0
        self.hits_taken = 0
        self.stance_ticks = [0, 0, 0]

        # Reset reward component tracking
        self.episode_proximity_reward = 0
        self.episode_damage_reward = 0
        self.episode_inaction_penalty = 0
        self.episode_terminal_reward = 0
        self.episode_stamina_reward = 0
        self.episode_stance_reward = 0

        # Reset temporal observation builder
        self.obs_builder.reset(
            mask=np.array([True]),
            distance=np.array([self.last_distance / self.config.arena_width], dtype=np.float32),
            opp_hp=np.array([self.opponent.hp / self.opponent.max_hp], dtype=np.float32),
            self_hp=np.array([self.fighter.hp / self.fighter.max_hp], dtype=np.float32),
            stamina=np.array([self.fighter.stamina / self.fighter.max_stamina], dtype=np.float32),
            opp_stance_int=np.array([int(self.opponent.stance)], dtype=np.int32),
        )
        self._temporal_features = np.zeros(8, dtype=np.float32)
        if self.use_history:
            from .signal_engine import HISTORY_OBS_DIM
            self._history_flat = np.zeros(HISTORY_OBS_DIM, dtype=np.float32)

        # Return initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: numpy array [accel_toward_opponent, stance_selector]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to arena format via canonical codec
        acceleration, stance_idx = scale_and_validate_action(
            action, self.config.max_acceleration
        )

        # Egocentric action: policy outputs "toward opponent" (positive) / "away" (negative).
        # Multiply by opponent direction to get absolute acceleration.
        opponent_dir = float(np.sign(float(self.opponent.position) - float(self.fighter.position)))
        if opponent_dir == 0.0:
            opponent_dir = 1.0  # arbitrary when overlapping
        acceleration *= opponent_dir
        self.stance_ticks[stance_idx] += 1

        # Use integer stance for JAX arena, string stance for Python arena
        from src.atom.runtime.arena.arena_1d_jax_jit import Arena1DJAXJit

        if isinstance(self.arena, Arena1DJAXJit):
            fighter_action = {"acceleration": acceleration, "stance": stance_idx}
        else:
            stance = stance_idx_to_str(stance_idx)
            fighter_action = {"acceleration": acceleration, "stance": stance}

        # Get opponent action
        snapshot_opp = generate_snapshot(self.opponent, self.fighter, self.tick, self.config.arena_width)
        opponent_action_dict = self.opponent_decide(snapshot_opp)

        # Convert opponent stance to int if using JAX arena
        if isinstance(self.arena, Arena1DJAXJit) and isinstance(opponent_action_dict.get("stance"), str):
            opponent_action_dict = opponent_action_dict.copy()
            opponent_action_dict["stance"] = stance_str_to_idx(opponent_action_dict["stance"])

        # Execute tick in arena
        prev_fighter_hp = self.fighter.hp
        prev_opponent_hp = self.opponent.hp
        prev_fighter_stamina = self.fighter.stamina

        events = self.arena.step(fighter_action, opponent_action_dict)

        # Calculate damage dealt/taken this step
        # Convert to floats to handle JAX Arrays
        damage_dealt = float(prev_opponent_hp - self.opponent.hp)
        damage_taken = float(prev_fighter_hp - self.fighter.hp)
        stamina_spent = float(prev_fighter_stamina - self.fighter.stamina)

        self.episode_damage_dealt += damage_dealt
        self.episode_damage_taken += damage_taken
        self.stamina_used += max(0, stamina_spent)  # Only count stamina spent, not regen

        if damage_dealt > 0:
            self.hits_landed += 1
        if damage_taken > 0:
            self.hits_taken += 1

        self.tick += 1

        # Update temporal features before building observation
        distance = float(abs(self.fighter.position - self.opponent.position))
        opp_dir = float(np.sign(float(self.opponent.position) - float(self.fighter.position)))
        if opp_dir == 0.0:
            opp_dir = 1.0
        closing_vel = float(self.fighter.velocity) * opp_dir

        self._temporal_features = self.obs_builder.update_and_get_temporal(
            distance=np.array([distance / self.config.arena_width], dtype=np.float32),
            opp_hp=np.array([float(self.opponent.hp) / float(self.opponent.max_hp)], dtype=np.float32),
            self_hp=np.array([float(self.fighter.hp) / float(self.fighter.max_hp)], dtype=np.float32),
            stamina=np.array([float(self.fighter.stamina) / float(self.fighter.max_stamina)], dtype=np.float32),
            opp_stance_int=np.array([int(self.opponent.stance)], dtype=np.int32),
            closing_vel=np.array([closing_vel / 5.0], dtype=np.float32),
            damage_dealt=np.array([max(0, damage_dealt)], dtype=np.float32),
            damage_taken=np.array([max(0, damage_taken)], dtype=np.float32),
            opp_stamina=np.array([float(self.opponent.stamina) / float(self.opponent.max_stamina)], dtype=np.float32),
        )[0]  # [0] to get scalar (8,) from (1, 8)

        # Get history if enabled
        if self.use_history:
            self._history_flat = self.obs_builder.get_flat_history()[0]  # (640,)

        # Get new observation (includes temporal + optional history)
        obs = self._get_observation()

        # Check termination
        terminated = bool(self.fighter.hp <= 0 or self.opponent.hp <= 0)
        truncated = bool(self.tick >= self.max_ticks)

        # Calculate normalized state needed by canonical reward engine.
        fighter_hp_pct = hp_pct(self.fighter.hp, self.fighter.max_hp)
        opponent_hp_pct = hp_pct(self.opponent.hp, self.opponent.max_hp)
        stamina_pct = float(self.fighter.stamina) / float(self.fighter.max_stamina)
        opp_stamina_pct = float(self.opponent.stamina) / float(self.opponent.max_stamina)
        distance = float(abs(self.fighter.position - self.opponent.position))

        reward_result = compute_step_reward_scalar(
            done=terminated,
            truncated=truncated,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            fighter_hp_pct=fighter_hp_pct,
            opponent_hp_pct=opponent_hp_pct,
            stamina_pct=stamina_pct,
            opp_stamina_pct=opp_stamina_pct,
            fighter_stance=self.fighter.stance,
            distance=distance,
            last_distance=self.last_distance,
            tick_count=self.tick,
            max_ticks=self.max_ticks,
            arena_width=self.config.arena_width,
            episode_damage_dealt=self.episode_damage_dealt,
            episode_stamina_used=self.stamina_used,
            reward_weights=self.reward_weights,
        )

        reward = reward_result.reward
        self.last_distance = reward_result.next_last_distance
        self.episode_damage_reward += reward_result.damage_component
        self.episode_proximity_reward += reward_result.proximity_component
        self.episode_stamina_reward += reward_result.stamina_component
        self.episode_stance_reward += reward_result.stance_component
        self.episode_inaction_penalty += reward_result.inaction_component
        self.episode_terminal_reward += reward_result.terminal_component

        # Info dict
        info = {
            "tick": self.tick,
            "opponent_name": self.opponent_name,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "episode_damage_dealt": self.episode_damage_dealt,
            "episode_damage_taken": self.episode_damage_taken,
            "fighter_hp": float(self.fighter.hp),
            "opponent_hp": float(self.opponent.hp),
            "fighter_stamina": float(self.fighter.stamina),
            "opponent_stamina": float(self.opponent.stamina),
            "hits_landed": self.hits_landed,
            "hits_taken": self.hits_taken,
            "stamina_used": self.stamina_used,
            "won": fighter_hp_pct > opponent_hp_pct if (terminated or truncated) else None,
            # Stance usage for this episode
            "stance_distribution": {
                "neutral": self.stance_ticks[0],
                "extended": self.stance_ticks[1],
                "defending": self.stance_ticks[2],
            } if (terminated or truncated) else None,
            # Reward breakdown (only available at episode end)
            "reward_breakdown": {
                "proximity": self.episode_proximity_reward,
                "damage": self.episode_damage_reward,
                "stamina": self.episode_stamina_reward,
                "stance": self.episode_stance_reward,
                "inaction": self.episode_inaction_penalty,
                "terminal": self.episode_terminal_reward,
                "total": self.episode_proximity_reward + self.episode_damage_reward +
                        self.episode_stamina_reward + self.episode_stance_reward +
                        self.episode_inaction_penalty + self.episode_terminal_reward
            } if (terminated or truncated) else None
        }

        # Ensure reward is a Python float (not a JAX Array)
        reward = float(reward) if hasattr(reward, '__float__') else reward

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation as numpy array (26D: 18D base + 8D temporal)."""
        opponent_direction = float(np.sign(float(self.opponent.position) - float(self.fighter.position)))
        ticks_since_hit = max(0, self.tick - int(self.fighter.last_hit_tick))
        hit_cooldown_fraction = min(ticks_since_hit / self.config.hit_cooldown_ticks, 1.0)
        obs = build_observation(
            you_position=float(self.fighter.position),
            you_velocity=float(self.fighter.velocity),
            you_hp=float(self.fighter.hp),
            you_max_hp=float(self.fighter.max_hp),
            you_stamina=float(self.fighter.stamina),
            you_max_stamina=float(self.fighter.max_stamina),
            opponent_position=float(self.opponent.position),
            opponent_velocity=float(self.opponent.velocity),
            opponent_hp=float(self.opponent.hp),
            opponent_max_hp=float(self.opponent.max_hp),
            opponent_stamina=float(self.opponent.stamina),
            opponent_max_stamina=float(self.opponent.max_stamina),
            opponent_stance=self.opponent.stance,
            arena_width=float(self.config.arena_width),
            you_stance=self.fighter.stance,
            tick_fraction=self.tick / self.max_ticks,
            opponent_direction=opponent_direction,
            hit_cooldown_fraction=hit_cooldown_fraction,
            temporal_features=self._temporal_features,
            history_features=self._history_flat if self.use_history else None,
        )
        # Sanitize NaN/Inf to match vmap_env_wrapper behaviour
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1000.0, neginf=-1000.0)
        return obs

    def render(self):
        """Rendering not implemented for training."""
        pass

    def set_opponent(self, opponent_decision_func):
        """
        Change the opponent decision function mid-training.

        This allows curriculum learning without recreating the environment,
        avoiding Monitor file handle issues during level transitions.

        Args:
            opponent_decision_func: New decision function for opponent
        """
        self.opponent_decide = opponent_decision_func

    def close(self):
        """Clean up resources."""
        pass
