"""
Gymnasium Environment Wrapper for Atom Combat

Wraps the Atom Combat arena as a Gym environment for RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Use relative imports within the src package
from ..arena import WorldConfig, FighterState, Arena1DJAXJit
from ..protocol.combat_protocol import generate_snapshot
from .signal_engine import build_observation, compute_step_reward_scalar


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
        seed: int = None
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

        # Define observation space (13 values for enhanced training)
        # [position, velocity, hp_norm, stamina_norm, distance, rel_velocity,
        #  opp_hp_norm, opp_stamina_norm, arena_width,
        #  wall_dist_left, wall_dist_right, opp_stance_int, recent_damage_dealt]
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15, 15, 15, 2, 100], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: Box (continuous) for finer control
        # [acceleration_normalized (-1 to 1), stance_selector (0 to 3.99)]
        # PPO works well with continuous action spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 2.99], dtype=np.float32),  # Changed from 3.99 to 2.99 for 3 stances
            dtype=np.float32
        )

        # Stance mapping (3-stance system)
        self.stance_names = ["neutral", "extended", "defending"]  # Removed retracted

        # State
        self.arena = None
        self.tick = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0
        self.last_distance = None
        self.stamina_used = 0
        self.hits_landed = 0
        self.hits_taken = 0

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
        self.last_distance = None
        self.stamina_used = 0
        self.hits_landed = 0
        self.hits_taken = 0

        # Reset reward component tracking
        self.episode_proximity_reward = 0
        self.episode_damage_reward = 0
        self.episode_inaction_penalty = 0
        self.episode_terminal_reward = 0
        self.episode_stamina_reward = 0
        self.episode_stance_reward = 0

        # Return initial observation
        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: numpy array [acceleration_normalized, stance_selector] from Box space

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action to arena format
        # action[0] is acceleration normalized (-1 to 1)
        # action[1] is stance selector (0.0-3.99, int cast to 0-3)
        acceleration_normalized = float(np.clip(action[0], -1.0, 1.0))
        acceleration = acceleration_normalized * self.config.max_acceleration

        stance_idx = int(np.clip(action[1], 0, 2))  # Clip to 2 for 3 stances (0,1,2)

        # Use integer stance for JAX arena, string stance for Python arena
        from ..arena.arena_1d_jax_jit import Arena1DJAXJit

        if isinstance(self.arena, Arena1DJAXJit):
            fighter_action = {"acceleration": acceleration, "stance": stance_idx}
        else:
            stance = self.stance_names[stance_idx]
            fighter_action = {"acceleration": acceleration, "stance": stance}

        # Get opponent action
        snapshot_opp = generate_snapshot(self.opponent, self.fighter, self.tick, self.config.arena_width)
        opponent_action_dict = self.opponent_decide(snapshot_opp)

        # Convert opponent stance to int if using JAX arena
        if isinstance(self.arena, Arena1DJAXJit) and isinstance(opponent_action_dict.get("stance"), str):
            opponent_action_dict = opponent_action_dict.copy()
            opponent_action_dict["stance"] = self.stance_names.index(opponent_action_dict["stance"])

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

        # Get new observation
        obs = self._get_observation()

        # Check termination
        terminated = bool(self.fighter.hp <= 0 or self.opponent.hp <= 0)
        truncated = bool(self.tick >= self.max_ticks)

        # Calculate normalized state needed by canonical reward engine.
        fighter_hp_pct = float(self.fighter.hp) / float(self.fighter.max_hp)
        opponent_hp_pct = float(self.opponent.hp) / float(self.opponent.max_hp)
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
        """Get current observation as numpy array (13 dimensions)."""
        return build_observation(
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
            recent_damage=float(self.episode_damage_dealt),
        )

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
