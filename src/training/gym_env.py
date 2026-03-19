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

        # Calculate reward based on outcome
        # Use HP percentage (not absolute HP) since fighters can have different max HP
        fighter_hp_pct = float(self.fighter.hp) / float(self.fighter.max_hp)
        opponent_hp_pct = float(self.opponent.hp) / float(self.opponent.max_hp)

        if terminated:
            # Someone died - determine winner
            if fighter_hp_pct > opponent_hp_pct:
                # WIN: Reduced magnitude rewards
                time_bonus = max(0, (self.max_ticks - self.tick) / 40)  # Up to +25 for quick wins
                hp_diff = fighter_hp_pct - opponent_hp_pct  # 0.0 to 1.0
                hp_bonus = hp_diff * 50  # Up to +50 for perfect win

                # Stamina efficiency bonus (up to +25 for stamina-efficient wins)
                stamina_efficiency = 0
                if self.stamina_used > 0:
                    damage_per_stamina = self.episode_damage_dealt / max(self.stamina_used, 1)
                    stamina_efficiency = min(25, damage_per_stamina * 5)

                reward = 100.0 + time_bonus + hp_bonus + stamina_efficiency  # Max ~200, typical ~125
            elif fighter_hp_pct == opponent_hp_pct:
                # TIE: Both died simultaneously - moderate penalty
                reward = -25.0
            else:
                # LOSS: Reduced penalty magnitude
                hp_diff = opponent_hp_pct - fighter_hp_pct  # 0.0 to 1.0
                hp_penalty = hp_diff * 50  # Up to -50 for getting dominated
                reward = -100.0 - hp_penalty  # -100 to -150
            self.episode_terminal_reward = reward
        elif truncated:
            # Timeout - compare HP percentage
            hp_pct_diff = fighter_hp_pct - opponent_hp_pct
            if hp_pct_diff > 0.1:
                # Clear win on HP (>10% margin) - significant reward
                reward = 100.0 + (hp_pct_diff * 50)  # Up to 150 for large HP margin
            elif hp_pct_diff > 0:
                # Slight win on HP - neutral (didn't finish fight decisively)
                reward = 0.0
            elif hp_pct_diff < -0.1:
                # Clear loss on HP
                reward = -100.0 + (hp_pct_diff * 50)  # Down to -150 for bad loss
            elif hp_pct_diff < 0:
                # Slight loss on HP
                reward = -50.0
            else:
                # Exact tie - very strong penalty (strongly discourage indecisive fights)
                reward = -200.0
            self.episode_terminal_reward = reward
        else:
            # Mid-episode rewards: Shaped to encourage good fighting behavior
            reward = 0

            # 1. Damage differential (core reward) - INCREASED
            damage_reward = (damage_dealt - damage_taken) * 10.0  # Was 2.0, now 10.0
            reward += damage_reward
            self.episode_damage_reward += damage_reward

            # 1b. Close-range engagement bonus - reward aggressive fighting
            # Distance is calculated below, so we need to get it here
            distance = float(abs(self.fighter.position - self.opponent.position))
            arena_width = self.config.arena_width
            if damage_dealt > 0 and distance < arena_width * 0.3:
                close_range_bonus = damage_dealt * 2.0  # Double damage reward for close hits
                reward += close_range_bonus
                self.episode_damage_reward += close_range_bonus

            # 2. Stamina-aware rewards - REDUCED (not core to basic combat)
            stamina_pct = float(self.fighter.stamina) / float(self.fighter.max_stamina)
            opp_stamina_pct = float(self.opponent.stamina) / float(self.opponent.max_stamina)

            # Small reward for maintaining stamina advantage
            if stamina_pct > opp_stamina_pct + 0.2:
                stamina_bonus = 0.02  # Was 0.1, now 0.02 (5x reduction)
                reward += stamina_bonus
                self.episode_stamina_reward += stamina_bonus

            # Small penalty for fighting at very low stamina (< 20%)
            # Get fighter's current stance as a string if needed
            fighter_stance_check = self.fighter.stance
            if isinstance(fighter_stance_check, int):
                fighter_stance_check = self.stance_names[fighter_stance_check]

            if stamina_pct < 0.2 and fighter_stance_check != "defending":
                stamina_penalty = -0.05  # Was -0.2, now -0.05 (4x reduction)
                reward += stamina_penalty
                self.episode_stamina_reward += stamina_penalty

            # 3. Smart proximity rewards (distance-aware)
            # (distance and arena_width already calculated above for close-range bonus)

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

            # Get fighter's current stance as a string for reward calculations
            fighter_stance = self.fighter.stance
            if isinstance(fighter_stance, int):
                fighter_stance = self.stance_names[fighter_stance]

            # Reward aggressive stance when opponent is hurt
            if fighter_stance == "extended" and opponent_hp_pct < 0.5:
                stance_bonus = 0.05
            # Reward defensive stance when recovering stamina
            elif fighter_stance == "defending" and stamina_pct < 0.3:
                stance_bonus = 0.10  # Increased since defending now regens stamina

            reward += stance_bonus
            self.episode_stance_reward += stance_bonus

            # 5. Inaction penalty (distance-aware) - reduced to prevent overwhelming signal
            # Light penalty for complete inaction to discourage passive strategies
            if damage_dealt == 0 and damage_taken == 0:
                if distance < arena_width * 0.2:  # Very close
                    inaction_penalty = -0.1  # Light penalty for not engaging at close range
                elif distance < arena_width * 0.4:  # Medium range
                    inaction_penalty = -0.05  # Very light penalty
                else:  # Far away
                    inaction_penalty = -0.02  # Minimal penalty for keeping distance
                reward += inaction_penalty
                self.episode_inaction_penalty += inaction_penalty

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

        # Wall distances
        wall_dist_left = float(self.fighter.position)
        wall_dist_right = float(self.config.arena_width) - float(self.fighter.position)

        # Opponent stance as integer
        opp_stance = self.opponent.stance
        if hasattr(opp_stance, '__array__'):  # JAX array
            opp_stance_int = float(opp_stance)
        else:  # String
            stance_map = {"neutral": 0, "extended": 1, "defending": 2}
            opp_stance_int = float(stance_map.get(opp_stance, 0))

        # Recent damage dealt (use episode tracking)
        recent_damage = float(self.episode_damage_dealt)

        obs = np.array([
            float(self.fighter.position),
            float(self.fighter.velocity),
            you_hp_norm,
            you_stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            float(self.config.arena_width),
            wall_dist_left,
            wall_dist_right,
            opp_stance_int,
            recent_damage
        ], dtype=np.float32)

        # Safety check: Replace any NaN or inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

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
