"""
Simple combat environment for rapid population training.
Simplified physics but maintains core combat mechanics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Callable
import random


class SimpleCombatEnv(gym.Env):
    """Minimal combat environment for quick training iterations."""

    def __init__(self, opponent_func: Optional[Callable] = None):
        """
        Initialize the simple combat environment.

        Args:
            opponent_func: Optional function that takes a snapshot and returns actions
        """
        super().__init__()
        self.opponent_func = opponent_func

        # Observation space: [my_pos, my_vel, my_hp, my_stamina,
        #                     distance, rel_vel, opp_hp, opp_stamina, arena_width]
        self.observation_space = spaces.Box(
            low=np.array([0, -5, 0, 0, 0, -10, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 5, 1, 1, 15, 10, 1, 1, 15], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: [acceleration, stance_index]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.99], dtype=np.float32),
            dtype=np.float32
        )

        # Arena configuration
        self.arena_width = 12.48
        self.dt = 0.0842  # Time step

        # Stance definitions
        self.stances = ["neutral", "extended", "retracted", "defending"]

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Fighter state
        self.pos = 2.0
        self.vel = 0.0
        self.hp = 100.0
        self.stamina = 8.0
        self.max_hp = 100.0
        self.max_stamina = 8.0

        # Opponent state
        self.opp_pos = self.arena_width - 2.0  # Start on opposite side
        self.opp_vel = 0.0
        self.opp_hp = 100.0
        self.opp_stamina = 8.0

        # Episode tracking
        self.tick = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.last_distance = abs(self.opp_pos - self.pos)

        return self._get_obs(), {}

    def _get_obs(self):
        """Get current observation."""
        dist = abs(self.opp_pos - self.pos)
        return np.array([
            self.pos,
            self.vel,
            self.hp / self.max_hp,
            self.stamina / self.max_stamina,
            dist,
            self.opp_vel - self.vel,  # Relative velocity
            self.opp_hp / self.max_hp,
            self.opp_stamina / self.max_stamina,
            self.arena_width
        ], dtype=np.float32)

    def _make_snapshot(self, for_opponent=False):
        """
        Create a snapshot dict for fighter decision functions.

        Args:
            for_opponent: If True, swap perspective for opponent
        """
        if for_opponent:
            p, v, h, s = self.opp_pos, self.opp_vel, self.opp_hp, self.opp_stamina
            op, ov, oh, os = self.pos, self.vel, self.hp, self.stamina
        else:
            p, v, h, s = self.pos, self.vel, self.hp, self.stamina
            op, ov, oh, os = self.opp_pos, self.opp_vel, self.opp_hp, self.opp_stamina

        return {
            "tick": self.tick,
            "you": {
                "position": p,
                "velocity": v,
                "hp": h,
                "stamina": s,
                "max_hp": self.max_hp,
                "max_stamina": self.max_stamina,
                "stance": "neutral"  # Default for compatibility
            },
            "opponent": {
                "distance": abs(op - p),
                "velocity": ov - v,
                "hp": oh,
                "stamina": os,
                "max_hp": self.max_hp,
                "max_stamina": self.max_stamina
            },
            "arena": {
                "width": self.arena_width
            }
        }

    def step(self, action):
        """Execute one time step within the environment."""

        # Parse fighter action
        accel = float(action[0]) * 4.5  # Scale to actual acceleration
        stance_idx = int(np.clip(action[1], 0, 3))
        stance = self.stances[stance_idx]

        # Update fighter physics
        self.vel = (self.vel + accel * self.dt) * 0.95  # Damping
        self.pos = np.clip(self.pos + self.vel * self.dt, 0, self.arena_width)

        # Stamina cost for acceleration
        accel_cost = abs(accel) * 0.02
        self.stamina = max(0, self.stamina - accel_cost)

        # Stance effects on stamina
        if stance == "neutral":
            self.stamina = min(self.max_stamina, self.stamina + 0.05)  # Recovery
        elif stance == "extended":
            self.stamina = max(0, self.stamina - 0.08)  # Aggressive drain
        elif stance == "defending":
            self.stamina = max(0, self.stamina - 0.12)  # Defensive drain
        elif stance == "retracted":
            self.stamina = max(0, self.stamina - 0.03)  # Minimal drain

        # Get opponent action
        if self.opponent_func:
            opp_snapshot = self._make_snapshot(for_opponent=True)
            opp_decision = self.opponent_func(opp_snapshot)
            opp_accel = np.clip(opp_decision.get("acceleration", 0), -4.5, 4.5)
            opp_stance = opp_decision.get("stance", "neutral")
        else:
            # Random opponent
            opp_accel = np.random.uniform(-2, 2)
            opp_stance = random.choice(self.stances)

        # Update opponent physics
        self.opp_vel = (self.opp_vel + opp_accel * self.dt) * 0.95
        self.opp_pos = np.clip(self.opp_pos + self.opp_vel * self.dt, 0, self.arena_width)

        # Opponent stamina
        opp_accel_cost = abs(opp_accel) * 0.02
        self.opp_stamina = max(0, self.opp_stamina - opp_accel_cost)

        if opp_stance == "neutral":
            self.opp_stamina = min(self.max_stamina, self.opp_stamina + 0.05)

        # Combat resolution
        dist = abs(self.opp_pos - self.pos)

        # Track damage this tick
        tick_damage_dealt = 0
        tick_damage_taken = 0

        if dist < 1.5:  # Combat range
            # Fighter attacks opponent
            if self.stamina > 0:
                # Base damage based on stance
                base_dmg = 3.0 if stance == "extended" else 2.0

                # Stamina multiplier (25% to 100% based on stamina)
                stamina_mult = 0.25 + (0.75 * (self.stamina / self.max_stamina))
                dmg = base_dmg * stamina_mult

                # Reduce if opponent defending
                if opp_stance == "defending":
                    dmg *= 0.5

                self.opp_hp -= dmg
                tick_damage_dealt = dmg
                self.damage_dealt += dmg

            # Opponent attacks fighter
            if self.opp_stamina > 0:
                base_dmg = 3.0 if opp_stance == "extended" else 2.0
                stamina_mult = 0.25 + (0.75 * (self.opp_stamina / self.max_stamina))
                dmg = base_dmg * stamina_mult

                if stance == "defending":
                    dmg *= 0.5

                self.hp -= dmg
                tick_damage_taken = dmg
                self.damage_taken += dmg

        # Calculate reward
        reward = 0

        # Immediate rewards/penalties
        if self.opp_hp <= 0:
            reward = 100  # Win bonus
        elif self.hp <= 0:
            reward = -100  # Loss penalty
        else:
            # Damage differential reward
            reward += (tick_damage_dealt - tick_damage_taken) * 2

            # Distance-based reward (encourage engagement)
            if dist < 3:
                reward += 0.2
            elif dist > 8:
                reward -= 0.1  # Penalize running away

            # Stamina management reward
            if self.stamina > self.max_stamina * 0.3:
                reward += 0.1  # Maintain stamina

            # HP differential bonus (small, to avoid camping)
            hp_diff = (self.hp - self.opp_hp) / self.max_hp
            reward += hp_diff * 0.5

        # Update tracking
        self.tick += 1
        self.last_distance = dist

        # Check termination
        done = self.hp <= 0 or self.opp_hp <= 0 or self.tick >= 500

        # Episode info
        info = {
            "won": self.opp_hp <= 0 and self.hp > 0,
            "hp": self.hp,
            "opp_hp": self.opp_hp,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "tick": self.tick
        }

        return self._get_obs(), reward, done, False, info