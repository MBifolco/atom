"""
Gymnasium Environment Wrapper for Atom Combat

Wraps the Atom Combat arena as a Gym environment for RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.arena import WorldConfig, FighterState, Arena1D
from src.protocol.combat_protocol import generate_snapshot


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
        - stance: discrete [0-3] (neutral, extended, retracted, defending)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_decision_func,
        config: WorldConfig = None,
        max_ticks: int = 1000,
        fighter_mass: float = 70.0,
        opponent_mass: float = 75.0,
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

        # Define observation space (9 continuous values)
        # Normalized to roughly [-1, 1] or [0, 1] range
        self.observation_space = spaces.Box(
            low=np.array([0, -3, 0, 0, 0, -5, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 3, 1, 1, 15, 5, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: Box (continuous) for finer control
        # [acceleration_normalized (-1 to 1), stance_selector (0 to 3.99)]
        # PPO works well with continuous action spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.99], dtype=np.float32),
            dtype=np.float32
        )

        # Stance mapping
        self.stance_names = ["neutral", "extended", "retracted", "defending"]

        # State
        self.arena = None
        self.fighter = None
        self.opponent = None
        self.tick = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Create fighters
        self.fighter = FighterState.create("learner", self.fighter_mass, 2.0, self.config)
        self.opponent = FighterState.create("opponent", self.opponent_mass, 10.0, self.config)

        # Create arena
        self.arena = Arena1D(self.fighter, self.opponent, self.config, seed=self._seed or 0)

        self.tick = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0

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

        stance_idx = int(np.clip(action[1], 0, 3))  # Clip and convert to int
        stance = self.stance_names[stance_idx]

        fighter_action = {"acceleration": acceleration, "stance": stance}

        # Get opponent action
        snapshot_opp = generate_snapshot(self.opponent, self.fighter, self.tick, self.config.arena_width)
        opponent_action_dict = self.opponent_decide(snapshot_opp)

        # Execute tick in arena
        prev_fighter_hp = self.fighter.hp
        prev_opponent_hp = self.opponent.hp

        events = self.arena.step(fighter_action, opponent_action_dict)

        # Calculate damage dealt/taken this step
        damage_dealt = prev_opponent_hp - self.opponent.hp
        damage_taken = prev_fighter_hp - self.fighter.hp

        self.episode_damage_dealt += damage_dealt
        self.episode_damage_taken += damage_taken

        self.tick += 1

        # Get new observation
        obs = self._get_observation()

        # Check termination
        terminated = self.fighter.hp <= 0 or self.opponent.hp <= 0
        truncated = self.tick >= self.max_ticks

        # Calculate reward based on outcome
        # Use HP percentage (not absolute HP) since fighters can have different max HP
        fighter_hp_pct = self.fighter.hp / self.fighter.max_hp
        opponent_hp_pct = self.opponent.hp / self.opponent.max_hp

        if terminated:
            # Someone died - determine winner
            if fighter_hp_pct > opponent_hp_pct:
                # WIN: Base bonus + time bonus + HP differential bonus
                time_bonus = max(0, (self.max_ticks - self.tick) / 20)  # Up to +50 for quick wins
                hp_diff = fighter_hp_pct - opponent_hp_pct  # 0.0 to 1.0 (100% margin)
                hp_bonus = hp_diff * 100  # Up to +100 for perfect win (100% HP vs 0%)
                reward = 200.0 + time_bonus + hp_bonus
            elif fighter_hp_pct == opponent_hp_pct:
                # TIE: Both died simultaneously - PENALTY
                # Strongly discourages mutual destruction, encourages survival
                reward = -50.0
            else:
                # LOSS: Penalty scales with how badly you lost
                hp_diff = opponent_hp_pct - fighter_hp_pct  # 0.0 to 1.0
                hp_penalty = hp_diff * 100  # Up to -100 for getting dominated
                reward = -200.0 - hp_penalty
        elif truncated:
            # Timeout - compare HP percentage
            hp_pct_diff = fighter_hp_pct - opponent_hp_pct
            if hp_pct_diff > 0:
                # Winning on HP but timed out - small reward
                reward = 50.0
            elif hp_pct_diff < 0:
                # Losing on HP - penalty
                reward = -100.0
            else:
                # Exact tie - small penalty (encourage decisive action)
                reward = -25.0
        else:
            # Mid-episode: Pure damage differential
            # Positive reward for dealing more damage than taking
            reward = (damage_dealt - damage_taken) * 2.0

            # Small penalty for being too passive (no damage dealt or taken)
            if damage_dealt == 0 and damage_taken == 0:
                reward -= 0.2  # Penalty for inaction

        # Info dict
        info = {
            "tick": self.tick,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "episode_damage_dealt": self.episode_damage_dealt,
            "episode_damage_taken": self.episode_damage_taken,
            "fighter_hp": self.fighter.hp,
            "opponent_hp": self.opponent.hp,
            "won": fighter_hp_pct > opponent_hp_pct if terminated else None  # Only actual wins count
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation as numpy array."""
        # Normalize HP and stamina to [0, 1]
        you_hp_norm = self.fighter.hp / self.fighter.max_hp
        you_stamina_norm = self.fighter.stamina / self.fighter.max_stamina
        opp_hp_norm = self.opponent.hp / self.opponent.max_hp
        opp_stamina_norm = self.opponent.stamina / self.opponent.max_stamina

        # Calculate opponent distance and relative velocity
        distance = abs(self.opponent.position - self.fighter.position)

        # Relative velocity (negative = approaching)
        if self.fighter.position < self.opponent.position:
            rel_velocity = self.opponent.velocity - self.fighter.velocity
        else:
            rel_velocity = self.fighter.velocity - self.opponent.velocity

        obs = np.array([
            self.fighter.position,
            self.fighter.velocity,
            you_hp_norm,
            you_stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            self.config.arena_width
        ], dtype=np.float32)

        return obs

    def _calculate_reward(self, damage_dealt, damage_taken):
        """
        Calculate reward for this step.

        NOTE: This is now only used for mid-episode rewards.
        Episode-end rewards are calculated in step() based on win/loss/timeout.
        """
        return damage_dealt - damage_taken

    def render(self):
        """Rendering not implemented for training."""
        pass

    def close(self):
        """Clean up resources."""
        pass
