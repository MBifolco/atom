#!/usr/bin/env python3
"""
Coaching wrapper for Atom Combat fighters.

This module provides a wrapper that adds real-time coaching capabilities
to existing AI fighters without requiring retraining.
"""

from typing import Dict, Optional
import numpy as np


class CoachingWrapper:
    """Wraps existing fighters to add coaching modifications."""

    def __init__(self, base_fighter):
        self.base_fighter = base_fighter
        self.coaching_mode = "balanced"
        self.aggression_bias = 0.0  # -1 to +1
        self.last_coach_command = None
        self.command_cooldown = 0
        self.command_history = []

        # Coaching statistics
        self.decisions_made = 0
        self.coaching_overrides = 0

    def decide(self, snapshot: Dict) -> Dict:
        """Modified decide that applies coaching."""

        # Get base AI decision
        base_decision = self.base_fighter.decide(snapshot)

        # Apply coaching modifications
        modified = self._apply_coaching(base_decision, snapshot)

        # Track statistics
        self.decisions_made += 1
        if modified != base_decision:
            self.coaching_overrides += 1

        # Update cooldowns
        if self.command_cooldown > 0:
            self.command_cooldown -= 1

        return modified

    def _apply_coaching(self, decision: Dict, snapshot: Dict) -> Dict:
        """Apply coaching overlay to AI decision."""

        modified = decision.copy()

        # Simple aggression/defense bias
        if self.aggression_bias > 0:
            # Aggressive coaching
            modified["acceleration"] *= (1 + self.aggression_bias * 0.5)
            if self.aggression_bias > 0.5 and snapshot["distance"] < 3:
                modified["stance"] = "extended"  # Force aggressive stance

        elif self.aggression_bias < 0:
            # Defensive coaching
            modified["acceleration"] *= (1 + self.aggression_bias * 0.3)
            if snapshot["stamina"] < 30:
                modified["stance"] = "defending"  # Force defensive when tired
                modified["acceleration"] = min(modified["acceleration"], -2.0)

        # Apply special commands if active
        if self.last_coach_command == "RUSH":
            if self.command_cooldown > 0:
                modified["acceleration"] = 4.5
                modified["stance"] = "extended"

        elif self.last_coach_command == "RETREAT":
            if self.command_cooldown > 0:
                modified["acceleration"] = -4.5
                modified["stance"] = "defending"

        elif self.last_coach_command == "COUNTER":
            if self.command_cooldown > 0:
                # Counter mode: defensive stance but ready to strike
                modified["stance"] = "retracted"
                if snapshot["distance"] < 2:
                    # Switch to attack when close
                    modified["stance"] = "extended"
                    modified["acceleration"] = 3.0
                else:
                    # Wait for opponent
                    modified["acceleration"] = -1.0

        return modified

    def receive_coaching(self, command: str) -> bool:
        """
        Receive coaching command from user.

        Returns True if command was accepted, False if on cooldown.
        """

        # Record command in history
        self.command_history.append((command, self.decisions_made))

        if command == "AGGRESSIVE":
            self.aggression_bias = min(1.0, self.aggression_bias + 0.3)
            self.coaching_mode = "aggressive"
            return True

        elif command == "DEFENSIVE":
            self.aggression_bias = max(-1.0, self.aggression_bias - 0.3)
            self.coaching_mode = "defensive"
            return True

        elif command == "BALANCED":
            self.aggression_bias = 0.0
            self.coaching_mode = "balanced"
            return True

        elif command == "RUSH" and self.command_cooldown == 0:
            self.last_coach_command = "RUSH"
            self.command_cooldown = 20  # 20 tick rush
            return True

        elif command == "RETREAT" and self.command_cooldown == 0:
            self.last_coach_command = "RETREAT"
            self.command_cooldown = 15  # 15 tick retreat
            return True

        elif command == "COUNTER" and self.command_cooldown == 0:
            self.last_coach_command = "COUNTER"
            self.command_cooldown = 25  # 25 tick counter mode
            return True

        return False  # Command rejected (on cooldown)

    def get_coaching_stats(self) -> Dict:
        """Get statistics about coaching effectiveness."""

        override_rate = 0
        if self.decisions_made > 0:
            override_rate = self.coaching_overrides / self.decisions_made * 100

        return {
            "mode": self.coaching_mode,
            "aggression_bias": self.aggression_bias,
            "cooldown": self.command_cooldown,
            "decisions_made": self.decisions_made,
            "coaching_overrides": self.coaching_overrides,
            "override_rate": override_rate,
            "command_history": self.command_history[-10:]  # Last 10 commands
        }

    def reset(self):
        """Reset coaching state between matches."""

        self.coaching_mode = "balanced"
        self.aggression_bias = 0.0
        self.last_coach_command = None
        self.command_cooldown = 0
        self.command_history = []
        self.decisions_made = 0
        self.coaching_overrides = 0


class SemanticCoachingIntent:
    """High-level coaching intents that work across all physics versions."""

    # Spatial Intents (abstract positioning)
    MAINTAIN_OPTIMAL_RANGE = "maintain_optimal_range"
    CONTROL_CENTER = "control_center"
    CREATE_DISTANCE = "create_distance"
    CLOSE_DISTANCE = "close_distance"
    LATERAL_MOVEMENT = "lateral_movement"  # No-op in 1D
    VERTICAL_CONTROL = "vertical_control"  # No-op in 1D

    # Tactical Intents (abstract strategy)
    AGGRESSIVE_PRESSURE = "aggressive_pressure"
    DEFENSIVE_POSTURE = "defensive_posture"
    COUNTER_FIGHTING = "counter_fighting"
    ENERGY_CONSERVATION = "energy_conservation"
    RISK_ASSESSMENT = "risk_assessment"

    # Temporal Intents (timing)
    IMMEDIATE_ACTION = "immediate_action"
    WAIT_FOR_OPENING = "wait_for_opening"
    SUSTAINED_PRESSURE = "sustained_pressure"
    BURST_OFFENSE = "burst_offense"
    RHYTHM_DISRUPTION = "rhythm_disruption"


class Physics1DTranslator:
    """Translator for current 1D physics."""

    def translate_intent(self, intent: str, context: Dict) -> Dict:
        """Translate semantic intent to 1D physics actions."""

        if intent == SemanticCoachingIntent.MAINTAIN_OPTIMAL_RANGE:
            current_distance = context.get("distance", 3.0)
            target_distance = context.get("optimal_range", 2.5)

            return {
                "acceleration": np.clip((target_distance - current_distance) * 2, -4.5, 4.5)
            }

        elif intent == SemanticCoachingIntent.CREATE_DISTANCE:
            return {"acceleration": -4.5, "stance": "defending"}

        elif intent == SemanticCoachingIntent.CLOSE_DISTANCE:
            return {"acceleration": 4.5, "stance": "neutral"}

        elif intent == SemanticCoachingIntent.AGGRESSIVE_PRESSURE:
            return {"acceleration": 3.0, "stance": "extended"}

        elif intent == SemanticCoachingIntent.DEFENSIVE_POSTURE:
            return {"acceleration": -2.0, "stance": "defending"}

        elif intent == SemanticCoachingIntent.COUNTER_FIGHTING:
            return {"acceleration": 0.0, "stance": "retracted"}

        elif intent == SemanticCoachingIntent.ENERGY_CONSERVATION:
            return {"acceleration": 0.0, "stance": "neutral"}

        elif intent == SemanticCoachingIntent.BURST_OFFENSE:
            return {"acceleration": 4.5, "stance": "extended"}

        elif intent == SemanticCoachingIntent.LATERAL_MOVEMENT:
            # No lateral movement in 1D
            return {}

        elif intent == SemanticCoachingIntent.VERTICAL_CONTROL:
            # No vertical control in 1D
            return {}

        return {}


class AdaptiveCoachingWrapper(CoachingWrapper):
    """Extended wrapper with semantic coaching support."""

    def __init__(self, base_fighter):
        super().__init__(base_fighter)
        self.translator = Physics1DTranslator()
        self.current_strategy = None
        self.strategy_timer = 0

    def apply_semantic_intent(self, intent: str, context: Dict) -> Dict:
        """Apply a semantic coaching intent."""

        action_modifier = self.translator.translate_intent(intent, context)
        return action_modifier

    def apply_boxing_strategy(self, strategy: str, snapshot: Dict):
        """Apply classic boxing strategies."""

        if strategy == "ROPE_A_DOPE":
            # Muhammad Ali's strategy
            if snapshot["stamina"] > 50:
                # Conserve energy early
                self.receive_coaching("DEFENSIVE")
            elif snapshot["opponent_stamina"] < 30:
                # Strike when opponent is tired
                self.receive_coaching("RUSH")

        elif strategy == "PEEK_A_BOO":
            # Mike Tyson's strategy
            if snapshot["distance"] > 3:
                # Close distance aggressively
                intent = self.apply_semantic_intent(
                    SemanticCoachingIntent.CLOSE_DISTANCE,
                    snapshot
                )
            else:
                # Unleash at close range
                self.receive_coaching("AGGRESSIVE")

        elif strategy == "OUTBOXER":
            # Floyd Mayweather's strategy
            if snapshot["distance"] < 2:
                # Too close, create distance
                intent = self.apply_semantic_intent(
                    SemanticCoachingIntent.CREATE_DISTANCE,
                    snapshot
                )
            else:
                # Counter from distance
                self.receive_coaching("COUNTER")