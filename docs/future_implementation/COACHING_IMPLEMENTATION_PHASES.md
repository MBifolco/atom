# Coaching System Implementation Phases
## Detailed Breakdown of Each Development Phase

---

## Overview

The coaching system will be implemented in progressive phases, each building upon the previous. This document provides detailed specifications for each phase to ensure consistent implementation.

---

## Phase 1: Simple Bias System (Current Implementation)

### Status: ✅ IMPLEMENTED

### Components Built
- `CoachingWrapper` class with aggression bias (-1 to +1)
- Basic coaching commands (AGGRESSIVE, DEFENSIVE, BALANCED)
- Simple decision modification logic
- Statistics tracking

### How It Works
```python
# Core modification logic
if self.aggression_bias > 0:
    # Aggressive: Increase forward movement by up to 50%
    modified["acceleration"] *= (1 + self.aggression_bias * 0.5)
    # Force aggressive stance when close
    if self.aggression_bias > 0.5 and snapshot["distance"] < 3:
        modified["stance"] = "extended"

elif self.aggression_bias < 0:
    # Defensive: Reduce movement by up to 30%
    modified["acceleration"] *= (1 + self.aggression_bias * 0.3)
    # Force defensive stance when tired
    if snapshot["stamina"] < 30:
        modified["stance"] = "defending"
        modified["acceleration"] = min(modified["acceleration"], -2.0)
```

### Testing Results Expected
- 10-15% win rate improvement with optimal coaching
- Higher impact against predictable opponents
- Reduced effectiveness against adaptive opponents

---

## Phase 1.5: Tactical Commands (Current Bonus)

### Status: ✅ IMPLEMENTED

### Components Built
- RUSH command (20 tick full aggression)
- RETREAT command (15 tick full defense)
- COUNTER command (25 tick counter-fighting mode)
- Cooldown system

### Implementation
```python
# Complete override during special commands
if self.last_coach_command == "RUSH" and self.command_cooldown > 0:
    modified["acceleration"] = 4.5     # Maximum forward
    modified["stance"] = "extended"    # Maximum aggression
```

---

## Phase 2: Corner Advice System

### Status: 📋 PLANNED

### Concept
Between-round coaching windows where strategic adjustments are made based on round analysis.

### Technical Specification

```python
class CornerAdviceSystem:
    def __init__(self):
        self.round_breaks = [100, 200, 300, 400]  # Every 100 ticks
        self.advice_window_duration = 30  # seconds
        self.round_stats = RoundAnalyzer()

    def analyze_round(self, round_data):
        """Analyze what happened in the round."""
        return {
            "damage_ratio": round_data.damage_dealt / max(round_data.damage_taken, 1),
            "stamina_efficiency": round_data.stamina_used / round_data.damage_dealt,
            "dominant_range": self.calculate_dominant_range(round_data),
            "opponent_patterns": self.detect_patterns(round_data),
            "vulnerable_moments": self.find_vulnerabilities(round_data)
        }

    def generate_advice_options(self, analysis):
        """Generate contextual advice based on analysis."""
        options = []

        if analysis["damage_ratio"] < 0.7:
            options.append({
                "text": "You're taking too much damage - increase defense",
                "adjustment": {"aggression_bias": -0.3, "priority": "survival"}
            })

        if analysis["opponent_patterns"]["predictable_rhythm"]:
            options.append({
                "text": "Opponent has predictable timing - disrupt their rhythm",
                "adjustment": {"strategy": "rhythm_breaker", "timing_offset": 0.5}
            })

        if analysis["stamina_efficiency"] < 0.5:
            options.append({
                "text": "You're wasting energy - be more selective with attacks",
                "adjustment": {"stamina_threshold": 40, "conservation_mode": True}
            })

        return options

    def apply_corner_advice(self, fighter, selected_advice):
        """Apply the selected advice to fighter configuration."""
        for key, value in selected_advice["adjustment"].items():
            setattr(fighter.coaching_config, key, value)
```

### UI Mockup
```
┌─────────────────────────────────────────┐
│         CORNER ADVICE - 25s             │
├─────────────────────────────────────────┤
│ Round 2 Summary:                        │
│ • Damage: 35 dealt / 42 taken          │
│ • Stamina efficiency: 68%              │
│ • Opponent favors: Close range         │
│                                        │
│ Select Strategy:                       │
│ [1] "Keep your distance"              │
│ [2] "Go for the knockout"             │
│ [3] "Tire them out"                   │
│ [4] "Counter their patterns"          │
└─────────────────────────────────────────┘
```

---

## Phase 3: Live Tactical Commands

### Status: 📋 PLANNED (Partially Implemented)

### Concept
Real-time special moves with strategic importance and resource management.

### Technical Specification

```python
class TacticalCommandSystem:
    def __init__(self):
        self.commands = {
            "SURGE": TacticalCommand(
                cooldown=50,
                duration=10,
                stamina_cost=20,
                effect=lambda f: setattr(f, 'damage_multiplier', 2.0),
                description="Double damage for 10 ticks"
            ),
            "FORTRESS": TacticalCommand(
                cooldown=30,
                duration=15,
                stamina_cost=0,
                effect=lambda f: setattr(f, 'damage_reduction', 0.5),
                stamina_regen=2.0,
                description="Half damage taken, regenerate stamina"
            ),
            "BERSERKER": TacticalCommand(
                cooldown=100,
                duration=20,
                stamina_cost=40,
                effect=lambda f: [
                    setattr(f, 'damage_multiplier', 3.0),
                    setattr(f, 'damage_reduction', 1.5)
                ],
                description="Triple damage but take 50% more damage"
            ),
            "FEINT": TacticalCommand(
                cooldown=20,
                duration=5,
                stamina_cost=10,
                effect=lambda f: setattr(f, 'evasion_chance', 0.5),
                description="50% chance to dodge attacks"
            )
        }

        self.active_effects = []
        self.cooldown_tracker = {}

    def execute_command(self, command_name, fighter, current_tick):
        """Execute a tactical command if available."""
        command = self.commands[command_name]

        # Check cooldown
        if not self.is_available(command_name, current_tick):
            return False

        # Check stamina
        if fighter.stamina < command.stamina_cost:
            return False

        # Apply effect
        command.effect(fighter)
        fighter.stamina -= command.stamina_cost

        # Track active effect
        self.active_effects.append({
            "command": command_name,
            "fighter": fighter,
            "start_tick": current_tick,
            "end_tick": current_tick + command.duration
        })

        # Set cooldown
        self.cooldown_tracker[command_name] = current_tick + command.cooldown

        return True

    def update_effects(self, current_tick):
        """Update and remove expired effects."""
        self.active_effects = [
            effect for effect in self.active_effects
            if effect["end_tick"] > current_tick
        ]
```

---

## Phase 4: Behavioral Modes System

### Status: 📋 PLANNED

### Concept
Pre-configured behavioral patterns that can be switched mid-fight.

### Technical Specification

```python
class BehavioralMode:
    def __init__(self, name, config):
        self.name = name
        self.stance_preference = config.get("stance_preference")
        self.distance_target = config.get("distance_target")
        self.aggression_level = config.get("aggression_level")
        self.stamina_threshold = config.get("stamina_threshold")
        self.reaction_style = config.get("reaction_style")

class BehavioralModeSystem:
    def __init__(self):
        self.modes = {
            "berserker": BehavioralMode("berserker", {
                "stance_preference": "extended",
                "distance_target": 1.0,
                "aggression_level": 1.0,
                "stamina_threshold": 0,  # Ignore stamina
                "reaction_style": "immediate"
            }),
            "turtle": BehavioralMode("turtle", {
                "stance_preference": "defending",
                "distance_target": 4.0,
                "aggression_level": -0.8,
                "stamina_threshold": 50,
                "reaction_style": "patient"
            }),
            "dancer": BehavioralMode("dancer", {
                "stance_preference": "neutral",
                "distance_target": "variable",  # Changes dynamically
                "aggression_level": 0,
                "stamina_threshold": 30,
                "reaction_style": "rhythmic"
            }),
            "sniper": BehavioralMode("sniper", {
                "stance_preference": "retracted",
                "distance_target": 6.0,
                "aggression_level": 0.3,
                "stamina_threshold": 40,
                "reaction_style": "opportunistic"
            })
        }

        self.current_mode = self.modes["balanced"]
        self.mode_history = []

    def switch_mode(self, mode_name, tick):
        """Switch to a new behavioral mode."""
        if mode_name not in self.modes:
            return False

        self.mode_history.append({
            "mode": self.current_mode.name,
            "switched_at": tick
        })

        self.current_mode = self.modes[mode_name]
        return True

    def apply_mode(self, base_decision, game_state):
        """Apply current mode to fighter decision."""
        mode = self.current_mode
        modified = base_decision.copy()

        # Apply stance preference
        if mode.stance_preference:
            modified["stance"] = mode.stance_preference

        # Apply distance targeting
        if isinstance(mode.distance_target, (int, float)):
            distance_error = game_state["distance"] - mode.distance_target
            modified["acceleration"] = -distance_error * 2.0

        # Apply aggression level
        modified["acceleration"] *= (1 + mode.aggression_level * 0.5)

        # Apply stamina conservation
        if game_state["stamina"] < mode.stamina_threshold:
            modified["acceleration"] *= 0.5
            modified["stance"] = "defending"

        return modified
```

---

## Phase 5: Ensemble Coaching

### Status: 📋 FUTURE

### Concept
Multiple specialized AIs work together, with coaching determining their influence.

### Technical Specification

```python
class EnsembleCoachingSystem:
    def __init__(self, specialized_models):
        self.models = specialized_models  # Dict of specialized AIs
        self.weights = self.initialize_equal_weights()
        self.performance_tracker = PerformanceTracker()

    def initialize_equal_weights(self):
        """Start with equal influence for all models."""
        n = len(self.models)
        return {name: 1.0/n for name in self.models.keys()}

    def get_ensemble_decision(self, game_state):
        """Combine decisions from all models based on weights."""
        decisions = {}
        for name, model in self.models.items():
            decisions[name] = model.decide(game_state)

        # Weighted average for continuous values
        combined = {
            "acceleration": sum(
                decisions[name]["acceleration"] * self.weights[name]
                for name in self.models.keys()
            ),
            "stance": self.weighted_stance_vote(decisions)
        }

        return combined

    def adjust_weights(self, coaching_command):
        """Adjust model weights based on coaching."""
        adjustments = {
            "AGGRESSIVE": {
                "aggressive_model": 0.6,
                "defensive_model": 0.1,
                "balanced_model": 0.3
            },
            "DEFENSIVE": {
                "aggressive_model": 0.1,
                "defensive_model": 0.6,
                "balanced_model": 0.3
            },
            "ADAPTIVE": self.calculate_adaptive_weights()
        }

        if coaching_command in adjustments:
            self.weights = adjustments[coaching_command]

    def calculate_adaptive_weights(self):
        """Dynamically calculate weights based on performance."""
        recent_performance = self.performance_tracker.get_recent(50)

        weights = {}
        for name in self.models.keys():
            model_success = recent_performance[name]["success_rate"]
            weights[name] = model_success / sum(recent_performance.values())

        return weights
```

---

## Phase 6: Natural Language Coaching

### Status: 🔮 FUTURE

### Concept
Convert natural language commands into coaching actions.

### Technical Specification

```python
class NaturalLanguageCoach:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.command_mapper = CommandMapper()

    def process_voice_command(self, audio_input):
        """Process voice coaching command."""
        # Speech to text
        text = self.speech_to_text(audio_input)

        # Extract intent and entities
        intent = self.intent_classifier.classify(text)
        entities = self.entity_extractor.extract(text)

        # Map to coaching command
        command = self.command_mapper.map(intent, entities)

        return command

    def example_mappings(self):
        return {
            "Get in there!": "AGGRESSIVE",
            "Back off!": "DEFENSIVE",
            "Watch your stamina": "ENERGY_CONSERVATION",
            "He's tired, go now!": "RUSH",
            "Keep your distance": "MAINTAIN_RANGE",
            "Break his rhythm": "RHYTHM_DISRUPTION"
        }
```

---

## Implementation Priority

### Must Have (MVP)
1. ✅ Phase 1: Simple Bias System
2. ✅ Phase 1.5: Basic Tactical Commands
3. ⏳ Testing Framework
4. ⏳ Basic Web UI Integration

### Should Have (V1)
1. Phase 2: Corner Advice System
2. Phase 4: Behavioral Modes
3. Coaching Analytics Dashboard
4. Tournament Mode

### Nice to Have (V2)
1. Phase 3: Advanced Tactical Commands
2. Phase 5: Ensemble Coaching
3. Coaching Replay System
4. AI Coach Assistant

### Future Vision
1. Phase 6: Natural Language Coaching
2. VR Coaching Interface
3. Multi-agent Team Coaching
4. Coaching Skill Progression System

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] 10%+ win rate improvement with optimal coaching
- [ ] <1ms coaching overhead per decision
- [ ] 80%+ user satisfaction in testing

### Overall Success Criteria
- [ ] 50%+ of matches use coaching
- [ ] 30%+ win rate swing possible with expert coaching
- [ ] Coaching becomes required for competitive play
- [ ] Community creates custom coaching strategies