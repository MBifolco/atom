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

## Conclusion

The coaching system transforms Atom Combat from an AI training platform into an interactive combat sport. By allowing real-time tactical adjustments without modifying the underlying AI, we create a new layer of gameplay that:

1. Maintains the importance of good AI training
2. Adds real-time strategic depth
3. Creates engaging spectator experiences
4. Provides natural learning progression
5. Opens new competitive dimensions

This system would be unique in the AI gaming space, combining the depth of AI training with the excitement of real-time strategy games. The key is starting simple (bias adjustments) and gradually adding complexity based on user feedback and engagement metrics.

The coaching layer doesn't replace good AI training - it enhances it, creating a symbiotic relationship where both elements are necessary for success. This creates a richer, more engaging platform that appeals to both AI enthusiasts and competitive gaming fans.