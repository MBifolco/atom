# AI Fighter Coaching System - Design Document

## Executive Summary

Transform the Atom Combat platform from passive AI-vs-AI watching to active participation through real-time coaching. This system allows users to influence their AI fighters during battles, similar to how boxing coaches advise fighters from the corner. This creates a new layer of skill expression where coaching ability becomes as important as AI training quality.

## Core Concept

While AI fighters are pre-trained neural networks that can't be modified mid-fight, we can add a "coaching layer" that influences their decisions without retraining. This creates a hybrid system where:
- **AI provides the base fighting skills** (learned through training)
- **Coaches provide tactical adjustments** (real-time strategy modifications)
- **Success requires both** good AI training AND good coaching

---

## Coaching Mechanisms

### 1. Strategy Overlay System (Most Feasible)

Add a coaching layer that modifies AI outputs without changing the model:

```python
class CoachingOverlay:
    """Real-time coaching that modifies AI outputs without retraining."""

    def __init__(self):
        self.aggression_bias = 0.0      # -1 (defensive) to +1 (aggressive)
        self.stance_preference = None    # Force certain stances
        self.position_target = None      # Suggest positioning
        self.stamina_threshold = 0.3     # When to be conservative

    def apply_coaching(self, ai_decision, game_state):
        """Modify AI decision based on coaching."""
        decision = ai_decision.copy()

        # Coach says "be more aggressive!"
        if self.aggression_bias > 0:
            decision["acceleration"] *= (1 + self.aggression_bias)
            if game_state["distance"] < 3:
                decision["stance"] = "extended"  # Override to aggressive

        # Coach says "back off and recover!"
        if self.aggression_bias < 0:
            if game_state["stamina"] < self.stamina_threshold:
                decision["stance"] = "defending"
                decision["acceleration"] = -4.5  # Full retreat

        return decision
```

**Pros:**
- Easy to implement
- Doesn't require retraining
- Intuitive for users

**Cons:**
- Can conflict with AI's learned behavior
- May reduce effectiveness if used poorly

### 2. Behavioral Modes System

Coaches can switch between pre-defined behavioral patterns mid-fight:

```python
class FighterWithModes:
    def __init__(self):
        self.current_mode = "balanced"
        self.modes = {
            "aggressive": {
                "stance_bias": "extended",
                "distance_target": 1.5,
                "stamina_conservative": False
            },
            "defensive": {
                "stance_bias": "defending",
                "distance_target": 5.0,
                "stamina_conservative": True
            },
            "counter": {
                "stance_bias": "retracted",
                "wait_for_opening": True,
                "reaction_focused": True
            },
            "endurance": {
                "stamina_conserve": True,
                "stance_bias": "neutral",
                "minimize_movement": True
            }
        }

    def receive_coaching(self, command):
        """Coach shouts: 'Switch to counter-punch mode!'"""
        if command in self.modes:
            self.current_mode = command
            self.mode_switch_time = current_tick
```

**Implementation:**
- Each mode adjusts decision weights
- Smooth transitions between modes
- Visual feedback showing current mode

### 3. Weighted Ensemble Approach

Train multiple specialized AIs and let coaches adjust their influence:

```python
class EnsembleFighter:
    def __init__(self):
        self.aggressive_ai = load_model("aggressive.zip")
        self.defensive_ai = load_model("defensive.zip")
        self.technical_ai = load_model("technical.zip")

        # Coach adjusts these weights in real-time
        self.weights = {
            "aggressive": 0.33,
            "defensive": 0.33,
            "technical": 0.34
        }

    def decide(self, state):
        decisions = {
            "aggressive": self.aggressive_ai.predict(state),
            "defensive": self.defensive_ai.predict(state),
            "technical": self.technical_ai.predict(state)
        }

        # Weighted average of decisions
        return self.blend_decisions(decisions, self.weights)

    def coach_command(self, command):
        if command == "GET_AGGRESSIVE":
            self.weights = {"aggressive": 0.7, "defensive": 0.1, "technical": 0.2}
        elif command == "PLAY_IT_SAFE":
            self.weights = {"aggressive": 0.1, "defensive": 0.7, "technical": 0.2}
```

**Advantages:**
- Natural blending of strategies
- Coaches can create hybrid approaches
- Each AI can be highly specialized

### 4. Reward Shaping Interface

Coaches modify what the AI "cares about" in real-time:

```python
class CoachableRewardSystem:
    def __init__(self):
        self.reward_weights = {
            "damage_dealt": 1.0,
            "damage_avoided": 1.0,
            "stamina_conservation": 0.5,
            "positioning": 0.5,
            "opponent_stamina_drain": 0.3
        }

    def coach_adjust(self, command):
        if command == "GO_FOR_KNOCKOUT":
            self.reward_weights["damage_dealt"] = 3.0
            self.reward_weights["stamina_conservation"] = 0.1
        elif command == "SURVIVE_THIS_ROUND":
            self.reward_weights["damage_avoided"] = 3.0
            self.reward_weights["damage_dealt"] = 0.2
        elif command == "EXHAUST_OPPONENT":
            self.reward_weights["opponent_stamina_drain"] = 2.0
```

**Note:** This would require AIs trained with multi-objective optimization

### 5. Macro-Action System

Coaches can trigger pre-programmed combat sequences:

```python
class MacroActions:
    def __init__(self):
        self.macros = {
            "RUSH_COMBO": [
                {"acceleration": 4.5, "stance": "neutral", "duration": 5},
                {"acceleration": 0, "stance": "extended", "duration": 3},
                {"acceleration": 0, "stance": "extended", "duration": 3},
                {"acceleration": -4.5, "stance": "defending", "duration": 5}
            ],
            "BAIT_AND_COUNTER": [
                {"acceleration": 2.0, "stance": "retracted", "duration": 8},
                {"acceleration": -3.0, "stance": "defending", "duration": 4},
                {"acceleration": 4.5, "stance": "extended", "duration": 5}
            ],
            "CIRCLE_AND_STRIKE": [
                {"acceleration": 3.0, "stance": "neutral", "duration": 10},
                {"acceleration": -2.0, "stance": "extended", "duration": 3},
                {"acceleration": 3.0, "stance": "neutral", "duration": 10}
            ]
        }

    def execute_macro(self, macro_name):
        """Override AI for next N ticks with macro sequence."""
        self.current_macro = self.macros[macro_name]
        self.macro_position = 0
        self.macro_active = True
```

---

## Implementation Phases

### Phase 1: Simple Bias System (MVP)

Start with basic aggression/defensive adjustments:

```python
class SimpleCoaching:
    def __init__(self):
        self.bias_slider = 0.0  # -1 (full defense) to +1 (full attack)

    def modify_decision(self, ai_decision, bias):
        modified = ai_decision.copy()

        # Aggressive bias
        if bias > 0:
            modified["acceleration"] *= (1 + bias * 0.5)
            if bias > 0.5:
                modified["stance"] = "extended"

        # Defensive bias
        elif bias < 0:
            modified["acceleration"] *= (1 + bias * 0.3)
            if bias < -0.5:
                modified["stance"] = "defending"

        return modified
```

**UI:** Simple slider from Defense to Attack

### Phase 2: Corner Advice System

Between-round coaching (every 100 ticks):

```python
class CornerAdvice:
    def __init__(self):
        self.round_breaks = [100, 200, 300, 400]
        self.advice_window = 30  # seconds

    def give_advice(self, fighter_stats, opponent_patterns):
        """30-second window to adjust strategy."""

        # Analyze what happened
        analysis = {
            "stamina_usage": fighter_stats.stamina_burn_rate,
            "damage_ratio": fighter_stats.damage_dealt / max(fighter_stats.damage_taken, 1),
            "dominant_range": opponent_patterns.preferred_distance,
            "opponent_weakness": opponent_patterns.vulnerable_stance
        }

        # Generate advice options
        advice_options = self.generate_advice(analysis)

        # Coach selects (with timeout)
        selected = coach_ui.get_selection(advice_options, timeout=30)

        # Apply strategic changes
        return self.parse_advice_to_params(selected)
```

**Advice Examples:**
- "He's dropping guard after combos - counter-strike!"
- "You're burning stamina too fast - pace yourself"
- "He's weak at long range - keep distance"
- "Save energy for final push - go defensive"

### Phase 3: Live Tactical Commands

Real-time special moves with cooldowns:

```python
class TacticalCoaching:
    def __init__(self):
        self.commands = {
            "SURGE": {
                "cooldown": 50,
                "duration": 10,
                "effect": "double_aggression",
                "stamina_cost": 2.0,
                "description": "All-out offensive burst"
            },
            "FORTRESS": {
                "cooldown": 30,
                "duration": 15,
                "effect": "maximum_defense",
                "stamina_regen": True,
                "description": "Defensive stance with recovery"
            },
            "FEINT": {
                "cooldown": 20,
                "duration": 5,
                "effect": "deceptive_movement",
                "description": "Fake attack to create opening"
            },
            "BERSERKER": {
                "cooldown": 100,
                "duration": 20,
                "effect": "ignore_damage_trade_aggression",
                "description": "Trade damage for maximum offense"
            }
        }

        self.cooldown_tracker = {}

    def can_use(self, command):
        last_used = self.cooldown_tracker.get(command, -999)
        cooldown = self.commands[command]["cooldown"]
        return current_tick - last_used >= cooldown

    def execute_command(self, command):
        if self.can_use(command):
            self.cooldown_tracker[command] = current_tick
            return self.commands[command]["effect"]
        return None
```

---

## User Interface Concept

### In-Fight Coaching Interface

```
┌─────────────────────────────────────────────────────┐
│                 ATOM COMBAT ARENA                     │
├─────────────────────────────────────────────────────┤
│                                                       │
│     [FIGHTER A]  ════════════════  [FIGHTER B]       │
│         HP: ████████░░                HP: ██████░░   │
│     Stamina: ███░░░░░░            Stamina: ████████  │
│                                                       │
│     Distance: 3.2m    Round: 2/5    Time: 0:45      │
│                                                       │
├─────────────────────────────────────────────────────┤
│ COACHING PANEL:                                      │
│                                                       │
│ Mode: [BALANCED] ←→ Aggressive ○━━━●━━━○ Defensive   │
│                                                       │
│ Quick Commands:                                      │
│ [Q] Rush In      [W] Back Off     [E] Counter Mode  │
│ [A] Left Circle  [S] Center       [D] Right Circle  │
│                                                       │
│ Special Moves: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ [1] SURGE (Ready)    [2] FORTRESS (12s)             │
│ [3] FEINT (Ready)    [4] BERSERKER (45s)            │
│                                                       │
│ Coach Notes: "Opponent low stamina - press now!"     │
└─────────────────────────────────────────────────────┘
```

### Between-Round Corner Advice

```
┌─────────────────────────────────────────────────────┐
│                   CORNER ADVICE                       │
│                 Time Remaining: 24s                   │
├─────────────────────────────────────────────────────┤
│                                                       │
│ Round 2 Analysis:                                    │
│ • You dealt 35 damage (received 42)                 │
│ • Stamina efficiency: 68%                           │
│ • Hit rate: 45%                                     │
│ • Opponent favoring left side attacks               │
│                                                       │
│ Select Strategy for Round 3:                        │
│                                                       │
│ ┌─────────────────────────────────────────┐        │
│ │ [1] "He's tired - go for the knockout!"  │        │
│ │     → Aggressive stance, high pressure    │        │
│ └─────────────────────────────────────────┘        │
│                                                       │
│ ┌─────────────────────────────────────────┐        │
│ │ [2] "Circle right and counter-punch"     │        │
│ │     → Defensive, wait for openings        │        │
│ └─────────────────────────────────────────┘        │
│                                                       │
│ ┌─────────────────────────────────────────┐        │
│ │ [3] "Maintain pressure, watch stamina"   │        │
│ │     → Balanced approach, sustainable      │        │
│ └─────────────────────────────────────────┘        │
│                                                       │
└─────────────────────────────────────────────────────┤
```

---

## Why This System Is Transformative

### 1. **Active Engagement**
- Transforms passive watching into active participation
- Every fight becomes interactive, not just pre-determined
- Coaches feel invested in the outcome

### 2. **Skill Expression**
- Good coaching becomes a skill separate from AI training
- Bad coaching can lose fights with good AIs
- Great coaching can win fights with weaker AIs

### 3. **Comeback Mechanics**
- Smart tactical adjustments can turn fights around
- Recognizing opponent patterns rewards observation
- Critical moments become more exciting

### 4. **Spectator Experience**
- Viewers can see coaching decisions in real-time
- Adds narrative to fights ("The coach called for aggression!")
- Creates memorable moments

### 5. **Learning Tool**
- Helps users understand combat strategies
- Shows impact of different tactical approaches
- Teaches timing and resource management

### 6. **Competitive Depth**
- Tournaments could have "Coach of the Year" awards
- Teams could have specialized coaches for different scenarios
- Coaching meta-game develops alongside AI training meta

---

## Technical Architecture

### Integration Points

```python
class CoachedFighterWrapper:
    def __init__(self, base_ai_model, coaching_system):
        self.ai = base_ai_model
        self.coach = coaching_system
        self.decision_history = deque(maxlen=50)

    def decide(self, snapshot):
        # 1. Get base AI decision
        ai_decision = self.ai.predict(snapshot)

        # 2. Apply coaching modifications
        coach_input = self.coach.get_current_adjustment()
        modified_decision = self.coach.apply_coaching(
            ai_decision,
            snapshot,
            coach_input
        )

        # 3. Apply any active macros or special moves
        if self.coach.has_active_macro():
            modified_decision = self.coach.get_macro_action()

        # 4. Record for analysis
        self.decision_history.append({
            "tick": snapshot["tick"],
            "ai_decision": ai_decision,
            "coach_modification": coach_input,
            "final_decision": modified_decision
        })

        return modified_decision
```

### Performance Considerations

- Coaching logic must be fast (<1ms per decision)
- UI updates separate from game logic thread
- Network latency handling for online coaching
- Recording coaching inputs for replay system

---

## Implementation Roadmap

### Milestone 1: Proof of Concept (1-2 weeks)
- [ ] Simple aggression/defense slider
- [ ] Basic UI overlay
- [ ] Test with existing trained fighters
- [ ] Measure impact on fight outcomes

### Milestone 2: Core Features (3-4 weeks)
- [ ] 4 behavioral modes (Aggressive, Defensive, Counter, Endurance)
- [ ] Mode switching UI with keyboard shortcuts
- [ ] Visual feedback for current mode
- [ ] Basic cooldown system

### Milestone 3: Advanced Coaching (4-6 weeks)
- [ ] Corner advice system between rounds
- [ ] Special moves with cooldowns
- [ ] Coaching analytics dashboard
- [ ] Replay system showing coaching decisions

### Milestone 4: Competitive Features (6-8 weeks)
- [ ] Tournament coaching mode
- [ ] Coach profiles and statistics
- [ ] Coaching challenges/puzzles
- [ ] AI coach suggestions ("Your coach recommends...")

### Milestone 5: Polish & Extension
- [ ] Natural language processing for voice commands
- [ ] Advanced pattern recognition assists
- [ ] Multi-fighter coaching (team battles)
- [ ] Coach skill progression system

---

## Open Questions & Considerations

### Balance Questions
- How much influence should coaching have? (10%? 30%? 50%?)
- Should there be coaching-disabled tournaments?
- How to prevent coaching from making AI training irrelevant?

### Technical Questions
- Real-time networking for online coached matches?
- How to handle coaching in AI vs AI matches?
- Recording and replaying coached fights?

### Design Questions
- Visual feedback for coaching effects?
- Audio cues for coaching commands?
- Mobile interface for coaching?

### Competitive Integrity
- Preventing macro/scripted coaching?
- Fairness in latency-sensitive situations?
- Coaching assistance for beginners?

---

## Physics-Agnostic Semantic Abstraction Layer

### The Missing 15%: Scaling Across Physics Evolution

The coaching system must gracefully scale from 1D to 2D to 3D physics without requiring complete redesign. This requires a **semantic abstraction layer** that translates high-level coaching intents into physics-specific actions.

### Core Semantic Concepts (Physics-Independent)

These coaching concepts remain constant regardless of physics complexity:

```python
class SemanticCoachingIntent:
    """High-level coaching intents that work across all physics versions."""

    # Spatial Intents (abstract positioning)
    MAINTAIN_OPTIMAL_RANGE = "maintain_optimal_range"      # Close/mid/far in any dimension
    CONTROL_CENTER = "control_center"                      # Center control in 1D/2D/3D
    CREATE_DISTANCE = "create_distance"                    # Separation in any physics
    CLOSE_DISTANCE = "close_distance"                      # Approach in any physics
    LATERAL_MOVEMENT = "lateral_movement"                  # Side movement (null in 1D)
    VERTICAL_CONTROL = "vertical_control"                  # Height advantage (null in 1D/2D)

    # Tactical Intents (abstract strategy)
    AGGRESSIVE_PRESSURE = "aggressive_pressure"            # Attack focus
    DEFENSIVE_POSTURE = "defensive_posture"                # Defense focus
    COUNTER_FIGHTING = "counter_fighting"                  # Reactive style
    ENERGY_CONSERVATION = "energy_conservation"            # Stamina management
    RISK_ASSESSMENT = "risk_assessment"                    # Damage trade evaluation

    # Temporal Intents (timing)
    IMMEDIATE_ACTION = "immediate_action"                  # Act now
    WAIT_FOR_OPENING = "wait_for_opening"                  # Patience
    SUSTAINED_PRESSURE = "sustained_pressure"              # Continuous action
    BURST_OFFENSE = "burst_offense"                        # Short intense periods
    RHYTHM_DISRUPTION = "rhythm_disruption"                # Break patterns
```

### Physics Translation Layer

Each physics version implements translations from semantic intents to concrete actions:

```python
class PhysicsTranslator(ABC):
    """Translates semantic coaching to physics-specific actions."""

    @abstractmethod
    def translate_spatial_intent(self, intent: str, context: dict) -> dict:
        """Convert spatial intent to physics-specific movement."""
        pass

    @abstractmethod
    def translate_tactical_intent(self, intent: str, context: dict) -> dict:
        """Convert tactical intent to stance/action modifications."""
        pass


class Physics1DTranslator(PhysicsTranslator):
    """Translator for current 1D physics."""

    def translate_spatial_intent(self, intent: str, context: dict) -> dict:
        if intent == SemanticCoachingIntent.MAINTAIN_OPTIMAL_RANGE:
            current_distance = context["distance"]
            if context["fighter_style"] == "grappler":
                target_distance = 1.5  # Close range
            elif context["fighter_style"] == "outboxer":
                target_distance = 4.0  # Long range
            else:
                target_distance = 2.5  # Mid range

            return {
                "acceleration": np.clip((target_distance - current_distance) * 2, -4.5, 4.5)
            }

        elif intent == SemanticCoachingIntent.CREATE_DISTANCE:
            return {"acceleration": -4.5}  # Max retreat in 1D

        elif intent == SemanticCoachingIntent.LATERAL_MOVEMENT:
            return {}  # No-op in 1D physics

        # ... other translations


class Physics2DTranslator(PhysicsTranslator):
    """Future translator for 2D physics."""

    def translate_spatial_intent(self, intent: str, context: dict) -> dict:
        if intent == SemanticCoachingIntent.LATERAL_MOVEMENT:
            # In 2D, we can actually move laterally
            return {
                "movement_vector": [0, 3.0],  # Sidestep
                "facing_adjustment": context["opponent_angle"]
            }

        elif intent == SemanticCoachingIntent.CONTROL_CENTER:
            # Move toward arena center in 2D
            center = [0, 0]
            current = context["position"]
            return {
                "movement_vector": normalize(center - current) * 3.0
            }
        # ... other 2D translations


class Physics3DTranslator(PhysicsTranslator):
    """Future translator for 3D physics."""

    def translate_spatial_intent(self, intent: str, context: dict) -> dict:
        if intent == SemanticCoachingIntent.VERTICAL_CONTROL:
            # In 3D, height matters (jumping, ducking, high ground)
            return {
                "movement_vector": [0, 0, 2.0],  # Jump/climb
                "stance": "aerial_ready"
            }
        # ... other 3D translations
```

### Coaching Strategy Persistence

Coaching strategies learned in simpler physics transfer to complex physics:

```python
class UniversalCoachingStrategy:
    """Coaching patterns that work across physics versions."""

    def __init__(self):
        self.strategies = {
            "ROPE_A_DOPE": [
                # Muhammad Ali's strategy - works in any physics
                (SemanticCoachingIntent.DEFENSIVE_POSTURE, {"duration": 60}),
                (SemanticCoachingIntent.ENERGY_CONSERVATION, {"duration": 60}),
                (SemanticCoachingIntent.WAIT_FOR_OPENING, {"duration": 30}),
                (SemanticCoachingIntent.BURST_OFFENSE, {"duration": 20})
            ],

            "PRESSURE_FIGHTER": [
                # Mike Tyson style - aggressive constant pressure
                (SemanticCoachingIntent.CLOSE_DISTANCE, {"until": "in_range"}),
                (SemanticCoachingIntent.AGGRESSIVE_PRESSURE, {"duration": 100}),
                (SemanticCoachingIntent.SUSTAINED_PRESSURE, {"stamina_threshold": 0.3})
            ],

            "OUTBOXER": [
                # Floyd Mayweather style - distance and counters
                (SemanticCoachingIntent.MAINTAIN_OPTIMAL_RANGE, {"range": "long"}),
                (SemanticCoachingIntent.COUNTER_FIGHTING, {"duration": 50}),
                (SemanticCoachingIntent.LATERAL_MOVEMENT, {"when_available": True}),
                (SemanticCoachingIntent.RHYTHM_DISRUPTION, {"frequency": 30})
            ]
        }

    def apply_strategy(self, strategy_name: str, physics_translator: PhysicsTranslator):
        """Apply strategy using current physics translator."""
        strategy = self.strategies[strategy_name]

        for intent, params in strategy:
            # Each intent translates differently based on physics
            concrete_action = physics_translator.translate_intent(intent, params)
            yield concrete_action
```

### Automatic Physics Detection and Adaptation

The system automatically detects physics version and loads appropriate translator:

```python
class AdaptiveCoachingSystem:
    """Coaching that automatically adapts to physics version."""

    def __init__(self):
        self.translator = None
        self.physics_version = None

    def detect_physics_version(self, observation_space):
        """Auto-detect physics from observation space."""

        obs_dim = observation_space.shape[0]

        if obs_dim == 9:  # Current 1D: x, y, vx, vy, hp, stamina, opp_x, opp_y, distance
            self.physics_version = "1D"
            self.translator = Physics1DTranslator()

        elif obs_dim == 15:  # Future 2D: adds angles, rotation, etc
            self.physics_version = "2D"
            self.translator = Physics2DTranslator()

        elif obs_dim == 24:  # Future 3D: adds z-axis, pitch, roll
            self.physics_version = "3D"
            self.translator = Physics3DTranslator()

        print(f"Detected {self.physics_version} physics, loading translator...")

    def coach_with_intent(self, semantic_intent: str, game_state: dict):
        """Apply coaching using semantic intent."""

        # Same coaching intent works across all physics versions!
        concrete_action = self.translator.translate_intent(
            semantic_intent,
            game_state
        )

        return concrete_action
```

## Concrete Implementation Plan for Current 1D World

### Phase 1: Minimal Viable Coaching (This Week)

#### Step 1: Create the Coaching Wrapper

Create `src/coaching/coaching_wrapper.py`:

```python
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

    def decide(self, snapshot: Dict) -> Dict:
        """Modified decide that applies coaching."""

        # Get base AI decision
        base_decision = self.base_fighter.decide(snapshot)

        # Apply coaching modifications
        modified = self._apply_coaching(base_decision, snapshot)

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

        return modified

    def receive_coaching(self, command: str):
        """Receive coaching command from user."""

        if command == "AGGRESSIVE":
            self.aggression_bias = min(1.0, self.aggression_bias + 0.3)

        elif command == "DEFENSIVE":
            self.aggression_bias = max(-1.0, self.aggression_bias - 0.3)

        elif command == "BALANCED":
            self.aggression_bias = 0.0

        elif command == "RUSH" and self.command_cooldown == 0:
            self.last_coach_command = "RUSH"
            self.command_cooldown = 20  # 20 tick rush

        elif command == "RETREAT" and self.command_cooldown == 0:
            self.last_coach_command = "RETREAT"
            self.command_cooldown = 15  # 15 tick retreat
```

#### Step 2: Integrate with MatchOrchestrator

Modify `src/arena/match_orchestrator.py`:

```python
# Add to imports
from src.coaching.coaching_wrapper import CoachingWrapper

class MatchOrchestrator:
    def __init__(self, ...):
        # ... existing init ...
        self.coaching_enabled = False
        self.coached_fighter = None

    def enable_coaching(self, fighter_index: int = 0):
        """Enable coaching for specified fighter."""
        self.coaching_enabled = True

        # Wrap the fighter with coaching
        if fighter_index == 0:
            self.fighter_a = CoachingWrapper(self.fighter_a)
            self.coached_fighter = self.fighter_a
        else:
            self.fighter_b = CoachingWrapper(self.fighter_b)
            self.coached_fighter = self.fighter_b

    def apply_coaching_command(self, command: str):
        """Apply coaching command if enabled."""
        if self.coaching_enabled and self.coached_fighter:
            self.coached_fighter.receive_coaching(command)

    def run_match(self):
        """Modified match loop with coaching checks."""

        while self.tick < self.max_ticks:
            # ... existing physics ...

            # Check for coaching input (in real implementation, from UI)
            if self.coaching_enabled and self.tick % 10 == 0:
                # Placeholder for UI input
                coaching_input = self._get_coaching_input()
                if coaching_input:
                    self.apply_coaching_command(coaching_input)

            # ... rest of match loop ...
```

#### Step 3: Create Simple CLI Interface

Create `coach_fight.py`:

```python
#!/usr/bin/env python3
"""
Interactive coaching interface for Atom Combat.

Usage:
    python coach_fight.py fighter_a.py fighter_b.py --coach 0

    During fight:
    - Press 'a' for aggressive
    - Press 'd' for defensive
    - Press 'b' for balanced
    - Press 'r' for rush (cooldown: 50 ticks)
    - Press 'e' for retreat (cooldown: 30 ticks)
"""

import sys
import threading
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.arena.match_orchestrator import MatchOrchestrator
from src.coaching.coaching_wrapper import CoachingWrapper
import select
import termios
import tty

class InteractiveCoach:
    def __init__(self, orchestrator, fighter_index=0):
        self.orchestrator = orchestrator
        self.fighter_index = fighter_index
        self.running = True

    def get_key_press(self):
        """Non-blocking key press detection."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def run(self):
        """Main coaching loop."""

        # Enable coaching
        self.orchestrator.enable_coaching(self.fighter_index)

        # Set terminal to raw mode for instant key detection
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            # Start match in thread
            match_thread = threading.Thread(target=self.orchestrator.run_match)
            match_thread.start()

            print("\nCOACHING ACTIVE - Commands:")
            print("  [A]ggressive  [D]efensive  [B]alanced")
            print("  [R]ush        [E]scape\n")

            # Coaching input loop
            while self.orchestrator.tick < self.orchestrator.max_ticks:
                key = self.get_key_press()

                if key:
                    command = None

                    if key.lower() == 'a':
                        command = "AGGRESSIVE"
                        print("\r>> AGGRESSIVE MODE", end="")

                    elif key.lower() == 'd':
                        command = "DEFENSIVE"
                        print("\r>> DEFENSIVE MODE", end="")

                    elif key.lower() == 'b':
                        command = "BALANCED"
                        print("\r>> BALANCED MODE", end="")

                    elif key.lower() == 'r':
                        command = "RUSH"
                        print("\r>> RUSH COMMAND!", end="")

                    elif key.lower() == 'e':
                        command = "RETREAT"
                        print("\r>> RETREAT!", end="")

                    if command:
                        self.orchestrator.apply_coaching_command(command)

                # Display status
                if self.orchestrator.tick % 20 == 0:
                    self._display_status()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _display_status(self):
        """Show current match status."""
        snapshot = self.orchestrator.get_current_state()

        print(f"\rTick: {snapshot['tick']:4d} | "
              f"HP: {snapshot['fighter_a_hp']:3.0f} vs {snapshot['fighter_b_hp']:3.0f} | "
              f"Distance: {snapshot['distance']:4.1f}m | "
              f"Stamina: {snapshot['fighter_a_stamina']:3.0f}",
              end="")


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Coach an AI fighter")
    parser.add_argument("fighter_a", help="Path to fighter A")
    parser.add_argument("fighter_b", help="Path to fighter B")
    parser.add_argument("--coach", type=int, default=0,
                       help="Which fighter to coach (0 or 1)")
    args = parser.parse_args()

    # Load fighters
    fighter_a = load_fighter(args.fighter_a)
    fighter_b = load_fighter(args.fighter_b)

    # Create orchestrator
    orchestrator = MatchOrchestrator(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        max_ticks=500
    )

    # Run with coaching
    coach = InteractiveCoach(orchestrator, args.coach)
    coach.run()

    # Show results
    print("\n\nMATCH COMPLETE!")
    print(f"Winner: {orchestrator.get_winner()}")


if __name__ == "__main__":
    main()
```

### Phase 2: Web UI Integration (Next Week)

Add coaching to the web app's WebSocket interface:

```javascript
// coaching.js - Client side
class CoachingInterface {
    constructor(websocket) {
        this.ws = websocket;
        this.setupUI();
    }

    setupUI() {
        // Aggression slider
        this.slider = document.getElementById('aggression-slider');
        this.slider.addEventListener('input', (e) => {
            this.ws.send(JSON.stringify({
                type: 'coaching',
                command: 'SET_BIAS',
                value: parseFloat(e.target.value)
            }));
        });

        // Special commands
        document.getElementById('rush-btn').addEventListener('click', () => {
            this.ws.send(JSON.stringify({
                type: 'coaching',
                command: 'RUSH'
            }));
        });
    }

    updateStatus(data) {
        // Update UI with cooldowns, current mode, etc
        document.getElementById('mode-display').textContent = data.mode;
        document.getElementById('cooldown').textContent = data.cooldown;
    }
}
```

### Testing the Coaching System

Create `test_coaching.py`:

```python
"""Test coaching impact on fight outcomes."""

def test_coaching_impact():
    """Measure how coaching affects win rates."""

    results = {
        "no_coaching": {"wins": 0, "losses": 0},
        "aggressive_coaching": {"wins": 0, "losses": 0},
        "defensive_coaching": {"wins": 0, "losses": 0},
        "smart_coaching": {"wins": 0, "losses": 0}
    }

    for i in range(100):
        # Test each coaching style
        for style in results.keys():
            orchestrator = create_test_match()

            if style == "aggressive_coaching":
                orchestrator.enable_coaching(0)
                orchestrator.apply_coaching_command("AGGRESSIVE")

            elif style == "defensive_coaching":
                orchestrator.enable_coaching(0)
                orchestrator.apply_coaching_command("DEFENSIVE")

            elif style == "smart_coaching":
                orchestrator.enable_coaching(0)
                # Apply smart coaching based on game state
                apply_smart_coaching(orchestrator)

            orchestrator.run_match()

            if orchestrator.get_winner() == "fighter_a":
                results[style]["wins"] += 1
            else:
                results[style]["losses"] += 1

    # Print results
    print("\nCOACHING IMPACT ANALYSIS:")
    print("-" * 40)

    for style, record in results.items():
        win_rate = record["wins"] / (record["wins"] + record["losses"]) * 100
        print(f"{style:20s}: {win_rate:5.1f}% win rate")

    # Calculate coaching effectiveness
    base_rate = results["no_coaching"]["wins"] / 100
    best_coached = max(results.items(), key=lambda x: x[1]["wins"])

    print(f"\nBest coaching style: {best_coached[0]}")
    print(f"Improvement: {(best_coached[1]['wins']/100 - base_rate)*100:+.1f}%")
```

## Conclusion

The coaching system transforms Atom Combat from an AI training platform into an interactive combat sport. By allowing real-time tactical adjustments without modifying the underlying AI, we create a new layer of gameplay that:

1. Maintains the importance of good AI training
2. Adds real-time strategic depth
3. Creates engaging spectator experiences
4. Provides natural learning progression
5. Opens new competitive dimensions

**The key innovation is the semantic abstraction layer** that ensures coaching strategies remain valid as physics complexity increases. A "rush" command works whether fighters move in 1D, 2D, or 3D - the physics translator handles the implementation details.

This system would be unique in the AI gaming space, combining the depth of AI training with the excitement of real-time strategy games. The key is starting simple (bias adjustments) and gradually adding complexity based on user feedback and engagement metrics.

The coaching layer doesn't replace good AI training - it enhances it, creating a symbiotic relationship where both elements are necessary for success. This creates a richer, more engaging platform that appeals to both AI enthusiasts and competitive gaming fans.