"""
Gymnasium Environment Wrapper for Atom Combat

Wraps the Atom Combat arena as a Gym environment for RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Use relative imports within the src package
from ..arena import WorldConfig, FighterState, Arena1D
from ..protocol.combat_protocol import generate_snapshot


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

        stance_idx = int(np.clip(action[1], 0, 3))  # Clip and convert to int
        stance = self.stance_names[stance_idx]

        fighter_action = {"acceleration": acceleration, "stance": stance}

        # Get opponent action
        snapshot_opp = generate_snapshot(self.opponent, self.fighter, self.tick, self.config.arena_width)
        opponent_action_dict = self.opponent_decide(snapshot_opp)

        # Execute tick in arena
        prev_fighter_hp = self.fighter.hp
        prev_opponent_hp = self.opponent.hp
        prev_fighter_stamina = self.fighter.stamina

        events = self.arena.step(fighter_action, opponent_action_dict)

        # Calculate damage dealt/taken this step
        damage_dealt = prev_opponent_hp - self.opponent.hp
        damage_taken = prev_fighter_hp - self.fighter.hp
        stamina_spent = prev_fighter_stamina - self.fighter.stamina

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
        terminated = self.fighter.hp <= 0 or self.opponent.hp <= 0
        truncated = self.tick >= self.max_ticks

        # Calculate reward based on outcome
        # Use HP percentage (not absolute HP) since fighters can have different max HP
        fighter_hp_pct = self.fighter.hp / self.fighter.max_hp
        opponent_hp_pct = self.opponent.hp / self.opponent.max_hp

        if terminated:
            # Someone died - determine winner
            if fighter_hp_pct > opponent_hp_pct:
                # WIN: Base bonus + time bonus + HP differential bonus + efficiency bonus
                time_bonus = max(0, (self.max_ticks - self.tick) / 20)  # Up to +50 for quick wins
                hp_diff = fighter_hp_pct - opponent_hp_pct  # 0.0 to 1.0 (100% margin)
                hp_bonus = hp_diff * 100  # Up to +100 for perfect win (100% HP vs 0%)

                # Stamina efficiency bonus (up to +25 for stamina-efficient wins)
                stamina_efficiency = 0
                if self.stamina_used > 0:
                    damage_per_stamina = self.episode_damage_dealt / max(self.stamina_used, 1)
                    stamina_efficiency = min(25, damage_per_stamina * 5)

                reward = 200.0 + time_bonus + hp_bonus + stamina_efficiency
            elif fighter_hp_pct == opponent_hp_pct:
                # TIE: Both died simultaneously - PENALTY
                # Strongly discourages mutual destruction, encourages survival
                reward = -50.0
            else:
                # LOSS: Penalty scales with how badly you lost
                hp_diff = opponent_hp_pct - fighter_hp_pct  # 0.0 to 1.0
                hp_penalty = hp_diff * 100  # Up to -100 for getting dominated
                reward = -200.0 - hp_penalty
            self.episode_terminal_reward = reward
        elif truncated:
            # Timeout - compare HP percentage
            hp_pct_diff = fighter_hp_pct - opponent_hp_pct
            if hp_pct_diff > 0.1:
                # Clear win on HP (>10% margin) - significant reward
                # Increased from 25 to 100 to properly reward dominance
                reward = 100.0 + (hp_pct_diff * 50)  # Up to 150 for large HP margin
            elif hp_pct_diff > 0:
                # Slight win on HP - moderate reward
                reward = 50.0
            elif hp_pct_diff < -0.1:
                # Clear loss on HP
                reward = -100.0 + (hp_pct_diff * 50)  # Down to -150 for bad loss
            elif hp_pct_diff < 0:
                # Slight loss on HP
                reward = -50.0
            else:
                # Exact tie - small penalty (encourage decisive action)
                reward = -25.0
            self.episode_terminal_reward = reward
        else:
            # Mid-episode rewards: Shaped to encourage good fighting behavior
            reward = 0

            # 1. Damage differential (core reward) - INCREASED
            damage_reward = (damage_dealt - damage_taken) * 10.0  # Was 2.0, now 10.0
            reward += damage_reward
            self.episode_damage_reward += damage_reward

            # 2. Stamina-aware rewards - REDUCED (not core to basic combat)
            stamina_pct = self.fighter.stamina / self.fighter.max_stamina
            opp_stamina_pct = self.opponent.stamina / self.opponent.max_stamina

            # Small reward for maintaining stamina advantage
            if stamina_pct > opp_stamina_pct + 0.2:
                stamina_bonus = 0.02  # Was 0.1, now 0.02 (5x reduction)
                reward += stamina_bonus
                self.episode_stamina_reward += stamina_bonus

            # Small penalty for fighting at very low stamina (< 20%)
            if stamina_pct < 0.2 and stance != "defending":
                stamina_penalty = -0.05  # Was -0.2, now -0.05 (4x reduction)
                reward += stamina_penalty
                self.episode_stamina_reward += stamina_penalty

            # 3. Smart proximity rewards (distance-aware)
            distance = abs(self.fighter.position - self.opponent.position)
            arena_width = self.config.arena_width  # ~12.5 meters

            # Track closing/opening distance
            if self.last_distance is not None:
                distance_delta = self.last_distance - distance  # Positive = closing

                # Reward closing distance when opponent is low HP or stamina
                if opponent_hp_pct < 0.3 or opp_stamina_pct < 0.2:
                    if distance_delta > 0.1:  # Closing in
                        proximity_bonus = 0.2
                        reward += proximity_bonus
                        self.episode_proximity_reward += proximity_bonus

                # Reward backing off when low on stamina
                elif stamina_pct < 0.2:
                    if distance_delta < -0.1:  # Backing away
                        proximity_bonus = 0.1
                        reward += proximity_bonus
                        self.episode_proximity_reward += proximity_bonus

                # Normal engagement distance reward (moderate, not overwhelming)
                elif distance < arena_width * 0.25:  # Within ~3 meters
                    proximity_bonus = 0.1 * (1.0 - distance / (arena_width * 0.25))
                    reward += proximity_bonus
                    self.episode_proximity_reward += proximity_bonus

            self.last_distance = distance

            # 4. Stance-appropriate rewards
            stance_bonus = 0

            # Reward aggressive stance when opponent is hurt
            if stance == "extended" and opponent_hp_pct < 0.5:
                stance_bonus = 0.05
            # Reward defensive stance when recovering stamina
            elif stance == "defending" and stamina_pct < 0.3:
                stance_bonus = 0.05
            # Reward retracted stance when countering
            elif stance == "retracted" and damage_dealt > 0:
                stance_bonus = 0.15

            reward += stance_bonus
            self.episode_stance_reward += stance_bonus

            # 5. Inaction penalty (distance-aware) - DRASTICALLY REDUCED
            # Small penalty for complete inaction to encourage engagement
            if damage_dealt == 0 and damage_taken == 0:
                if distance < arena_width * 0.2:  # Very close
                    inaction_penalty = -0.05  # Was -0.8, now -0.05 (16x reduction!)
                elif distance < arena_width * 0.4:  # Medium range
                    inaction_penalty = -0.02  # Was -0.4, now -0.02
                else:  # Far away
                    inaction_penalty = -0.01  # Was -0.2, now -0.01
                reward += inaction_penalty
                self.episode_inaction_penalty += inaction_penalty

        # Info dict
        info = {
            "tick": self.tick,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "episode_damage_dealt": self.episode_damage_dealt,
            "episode_damage_taken": self.episode_damage_taken,
            "fighter_hp": self.fighter.hp,
            "opponent_hp": self.opponent.hp,
            "fighter_stamina": self.fighter.stamina,
            "opponent_stamina": self.opponent.stamina,
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
