# New Hardcoded Fighters - Complete Guide

## Overview

Created 7 new specialized hardcoded fighters to teach population training diverse combat strategies. These complement the existing 3 fighters (Tank, Rusher, Balanced) and provide coverage for all major combat archetypes.

### Fighter Summary

| Fighter | Style | Primary Teaching | Difficulty | Introduction |
|---------|-------|------------------|------------|--------------|
| **Dodger** | Evasion | Pursuit & Prediction | Medium-High | Phase 3 |
| **Stamina Manager** | Resource Aware | Pacing & Efficiency | Medium | Phase 3 |
| **Counter Puncher** | Timing | Patience & Risk-Reward | High | Phase 3 |
| **Berserker** | All-Out Attack | Defense & Survival | Medium | Phase 2 |
| **Zoner** | Range Control | Distance Management | Medium-High | Phase 3 |
| **Grappler** | Close Combat | In-Fighting | Medium | Phase 2 |
| **Hit-and-Run** | Mobility | Prediction & Pursuit | High | Phase 3 |

---

## Individual Fighter Descriptions

### 1. Dodger - Evasion Specialist

**File:** `fighters/examples/dodger.py`

**Strategy:**
- Constantly moves away from opponent
- Kites and maintains distance through superior mobility
- Counter-attacks only when opponent is vulnerable
- Efficient stamina usage for movement

**What It Teaches:**
- How to pursue a moving target
- Prediction of movement patterns
- When to commit to attacks vs maintain distance
- Dealing with hit-and-run opponents

**Combat Style:**
```
Enemy Close (< 2.5m)
  - Backs away: -2.5 to -3.5 m/s acceleration
  - Retracted stance (minimal damage exposure)
  
Enemy Medium (2.5-4m)
  - Light retreat: -0.5 to -1.5 m/s
  - Neutral stance
  
Enemy Far (> 4m)
  - Stops or continues away
  - Recovers stamina
```

**Key Decision Points:**
- Charges in when opponent is low HP (< 50%) and exhausted
- Never engages at close range unless opponent is vulnerable
- Uses full stamina budget for mobility, not combat

**Training Implications:**
- Forces learner to develop chase mechanics
- Teaches baiting aggressive fighters into overextending
- Rewards patience and prediction
- Punishes reckless charging

---

### 2. Stamina Manager - Resource Efficiency Specialist

**File:** `fighters/examples/stamina_manager.py`

**Strategy:**
- Hyperaware of stamina levels (5 distinct phases)
- Alternates aggressive attack windows with recovery periods
- Demonstrates optimal stamina economy

**Five Stamina Phases:**
```
Phase 1: Exhausted (< 15%)
  - Neutral stance, zero acceleration
  - Complete recovery focus

Phase 2: Low (15-35%)
  - Back away: -1.5 m/s
  - Neutral stance
  
Phase 3: Moderate (35-55%)
  - Strategic attacks: only when opportunity is clear
  - Extended stance only vs stationary/slow opponent
  
Phase 4: Good (55-75%)
  - Balanced combat: attack + move
  - Can charge from distance
  
Phase 5: High (75%+)
  - Maximum offense
  - Aggressive acceleration 3.5+ m/s
  - Relentless pressure
```

**What It Teaches:**
- Stamina is a critical resource
- Attack windows are strategic, not constant
- Recovery periods are necessary and valuable
- Pacing and rhythm of combat

**Training Implications:**
- Clear demonstration of optimal stamina timing
- Learner discovers stamina-aware decision making
- Teaches phase-based strategy (attack-recover cycles)
- Shows how exhaustion changes combat options

---

### 3. Counter Puncher - Timing Specialist

**File:** `fighters/examples/counter_puncher.py`

**Strategy:**
- Patient, defensive fighter
- Waits for opponent to overextend
- Punishes mistakes with timed counter-attacks
- Rarely initiates, always capitalizes on weakness

**Attack Conditions (ONLY attacks when):**
1. Opponent is charging in (distance < 1.5m AND velocity > 1.0)
2. Opponent is exhausted (distance < 2.0m AND HP < 40%)
3. Opponent just recovered from charge (distance < 2.5m AND negative velocity)

**What It Teaches:**
- Risk/reward analysis (when to commit)
- Patience is a strategic asset
- How to punish overextension
- Defensive stance usage and timing
- That aggressive play can be exploited

**Combat Style:**
```
Opponent Charging:      Counter-attack
Opponent Approaching:   Defensive stance
Opponent Distant:       Neutral maintenance
Opponent Exhausted:     Attack window
```

**Training Implications:**
- Teaches learner that aggression has costs
- Rewards calculated, timely attacks over spam
- Forces development of defensive capabilities
- Teaches reading opponent state and responding
- High-skill opponent (hardest for learner)

---

### 4. Berserker - Relentless Attack Specialist

**File:** `fighters/examples/berserker.py`

**Strategy:**
- **NEVER** uses defensive stance
- **ALWAYS** extended stance when possible
- Aggressive acceleration at all times
- Minimal HP management (only retreats at < 15% HP)

**Stamina Impact:**
- Doesn't care about stamina management
- Attacks even when exhausted (just at reduced power)
- Forces opponent to exhaust attacker, not be exhausted

**What It Teaches:**
- How to survive sustained pressure
- Importance of defensive stance
- Managing constant incoming damage
- Counter-attacking from defensive position
- When to use defending stance effectively

**Combat Style:**
```
Any Range, Any HP:
  IF stamina < 20%:    Reduce acceleration but keep extended
  ELSE:                Maximum acceleration + extended
  
Distance 0-1m:         Acceleration 0.5, extended
Distance 1-2.5m:       Acceleration 1.5-3.0, extended
Distance 2.5m+:        Acceleration 1.5-4.0, extended
```

**Training Implications:**
- Most aggressive opponent in the roster
- Forces defensive development
- Teaches block/defending stance value
- Teaches conservation vs raw pressure
- Medium difficulty (learner can survive but must defend)

---

### 5. Zoner - Range Control Specialist

**File:** `fighters/examples/zoner.py`

**Strategy:**
- Maintains maximum safe distance (3-5m "zoning range")
- Pokes from range with light extended stance
- Immediately retreats when opponent closes
- Never commits to close-range combat

**Optimal Zoning Range (3-5m):**
- Distance > 4m: Advance slowly (1 m/s)
- Distance 3-4m: Hold position, attack (extended)
- Distance 2-3m: Retreat slightly while attacking
- Distance < 2m: Rapid retreat (2-3.5 m/s)

**What It Teaches:**
- Range management and spacing
- How to approach ranged opponents
- Breaking down distance control
- Reading distance and reacting appropriately
- Punishing careless approach

**Combat Style:**
```
Zoning Range (3-5m):    Extended stance + mobile (0.5-1 m/s)
Too Close (< 2m):       Retreat fast: -2.5 m/s
Too Far (> 5m):         Advance: 1-2 m/s
Opponent Charging:      Defensive retreat: -3.5 m/s
```

**Training Implications:**
- Teaches distance evaluation
- Punishes charging blindly
- Rewards calculated approach
- Medium-high difficulty
- Teaches resource management under pressure

---

### 6. Grappler - Close Combat Specialist

**File:** `fighters/examples/grappler.py`

**Strategy:**
- **Always** wants to be close (< 1.5m)
- Forces close-range combat constantly
- Uses extended stance when in range
- Never backs away unless critical HP

**Distance-Based Behavior:**
```
Close (< 1.5m):     Fight hard: extended stance
Medium (1.5-3m):    Charge forward: 2-4 m/s
Far (> 3m):         Relentless pursuit: 2-4 m/s
```

**What It Teaches:**
- Close-quarters fighting fundamentals
- How to maintain distance against rushers
- When to use extended vs defending stance
- Managing opponent who always closes
- In-fighting tactics

**Training Implications:**
- Medium difficulty (aggressive but not erratic)
- Forces distance management learning
- Teaches close-range evasion
- Complements Dodger (opposite strategy)
- Medium introductory difficulty

---

### 7. Hit-and-Run - Mobility Specialist

**File:** `fighters/examples/hit_and_run.py`

**Strategy:**
- Hits opponent briefly (1-2 ticks)
- Immediately backs away
- Repeats cycle constantly
- Never commits to sustained combat

**Attack Cycle:**
```
Phase 1: Close (< 1.5m)
  - Attack with extended (1.5 m/s acceleration)
  - Then immediately retreat
  
Phase 2: Medium (1.5-3m)
  - Approach for next attack: 1.5-2.5 m/s
  - Stamina-dependent speed
  
Phase 3: Far (> 3m)
  - Recover stamina
  - Prepare for next cycle
```

**What It Teaches:**
- Predicting mobile opponent patterns
- Setting traps for hit-and-runners
- Managing pursuit without getting trapped
- Timing predictions (where will opponent be)
- Patience in face of constant mobility

**Training Implications:**
- High difficulty (hardest to pin down)
- Forces prediction learning
- Teaches trap-setting and positioning
- Complements Dodger (similar but more aggressive)
- Last opponent to introduce

---

## Training Progression Recommended

### Phase 1: Fundamentals (Episodes 1-5,000)
**Opponents:** Training Dummy, Wanderer, Rusher
- Goal: Learn basic mechanics
- Expected Win Rate: 70-90%

### Phase 2: Core Combat (Episodes 5,000-15,000)
**Opponents:** Bumbler, Novice, Tank, **Berserker**, **Grappler**
- Goal: Learn positioning and defense
- Expected Win Rate: 50-70%

### Phase 3: Advanced Tactics (Episodes 15,000-35,000)
**Opponents:** Balanced, **Dodger**, **Stamina Manager**, **Counter Puncher**, **Zoner**, **Hit-and-Run**
- Goal: Learn prediction and resource management
- Expected Win Rate: 30-50%

### Phase 4: Elite Play (Episodes 35,000+)
**Opponents:** Rotated combination of all 10 opponents
- Goal: Mastery and versatility
- Expected Win Rate: Self-improving

---

## Fighting Each New Opponent

### vs Dodger
**Challenge:** Catching a moving target
**Strategy To Learn:**
- Predict where Dodger will move
- Use acceleration strategically
- Don't overcommit to pursuit
- Wait for Dodger to counter-attack and counter that
- Use extended stance when Dodger is vulnerable

**Counter Tactics:**
- Advance steadily to cut off space
- Use walls to corner
- Wait for Dodger to make a mistake (rare)
- This opponent teaches prediction above all

### vs Stamina Manager
**Challenge:** Adapting to opponent's stamina phases
**Strategy To Learn:**
- Read opponent's stamina level
- Attack when Stamina Manager is recovering
- Defend when they're in high-stamina phase
- Manage your own stamina efficiently
- This teaches phase-based strategy

### vs Counter Puncher
**Challenge:** Punishing your aggression
**Strategy To Learn:**
- Don't overextend (you will be punished)
- Use defensive stance more
- Be patient and calculated
- Don't commit to attacks you can't follow through
- This teaches risk/reward thinking

### vs Berserker
**Challenge:** Surviving constant pressure
**Strategy To Learn:**
- Defend constantly (defending stance)
- Use blocks to reduce damage
- Counter-attack from defensive position
- Don't get pushed to wall
- This teaches defense and survival

### vs Zoner
**Challenge:** Breaking distance control
**Strategy To Learn:**
- Don't charge blindly (you'll be kited)
- Close distance gradually
- Use stamina efficiently in approach
- Once close, capitalize
- This teaches range management

### vs Grappler
**Challenge:** Managing a close fighter
**Strategy To Learn:**
- Keep distance (back away)
- Use extended/retracted stances
- Don't let Grappler dictate range
- Strike from mid-range, retreat
- This teaches distance preservation

### vs Hit-and-Run
**Challenge:** Catching evasive opponent
**Strategy To Learn:**
- Predict movement patterns
- Set traps/corner opponent
- Don't chase blindly
- Punish predictable cycles
- This teaches advanced prediction

---

## Diversity Metrics

All 10 opponents cover these archetypes:

### Attack Patterns
- **Aggressive Constant:** Rusher, Berserker, Grappler
- **Aggressive Timed:** Counter Puncher, Hit-and-Run
- **Passive/Evasive:** Dodger
- **Range-Based:** Zoner, Stamina Manager
- **Adaptive:** Balanced
- **Stationary:** Training Dummy

### Distance Preferences
- **Close (< 2m):** Grappler, Counter Puncher (opportunistic)
- **Medium (2-4m):** Tank, Rusher, Balanced
- **Far (> 4m):** Dodger, Zoner
- **Variable:** Stamina Manager, Hit-and-Run

### Stamina Management
- **Aggressive/Wasteful:** Berserker
- **Efficient:** Stamina Manager, Dodger
- **Aware:** Tank, Rusher, Balanced, Grappler
- **Ignored:** Zoner

### Defense Usage
- **Never:** Berserker
- **Selective:** Counter Puncher, Zoner
- **Strategic:** Tank, Balanced
- **Frequent:** Dodger

---

## Population Training Integration

### Recommended Rotations

**Week 1-2:** Fundamentals
```python
opponent_pool = [
    "training_dummy",
    "wanderer", 
    "rusher"
]
```

**Week 3-4:** Intermediate
```python
opponent_pool = [
    "bumbler",
    "novice",
    "tank",
    "berserker",
    "grappler"
]
```

**Week 5-6:** Advanced
```python
opponent_pool = [
    "balanced",
    "dodger",
    "stamina_manager",
    "counter_puncher",
    "zoner",
    "hit_and_run"
]
```

**Week 7+:** Elite (Rotation)
```python
opponent_pool = [
    # All 10 fighters rotated
    # Population fights diverse set each generation
]
```

### Evolution Pressure

With diverse opponent set:
- No single strategy dominates (rock-paper-scissors dynamics)
- Population naturally diversifies
- Top performers handle multiple styles
- ELO ratings reflect true skill vs varied opponents

---

## Expected Results

### Individual Training (vs Single Opponent)

| Fighter | Dummy | Wanderer | Bumbler | Rusher | Tank | Balanced | Counter | Zoner | Others |
|---------|-------|----------|---------|--------|------|----------|---------|-------|--------|
| Parzival (Level 5) | 95% | 85% | 75% | 60% | 50% | 45% | 35% | 40% | 30-45% |
| Newly Trained | 90% | 80% | 70% | 55% | 45% | 40% | 30% | 35% | 25-40% |

### Population Training Metrics

**Healthy Population Indicators:**
- ELO Std Dev > 40
- ELO Range > 200
- No single fighter > 65% win rate
- Win rate variance > 0.15
- Top fighter rotates between matches

**Expected Timeline:**
- Generation 1-2: High variance, random strategies
- Generation 3-5: Strategies crystallizing
- Generation 6+: Diverse specialized fighters
- Generation 10+: Complex meta-game emerging

---

## Files Created

```
fighters/examples/
  ├── dodger.py                  (NEW - Evasion)
  ├── stamina_manager.py         (NEW - Resource Management)
  ├── counter_puncher.py         (NEW - Timing/Patience)
  ├── berserker.py               (NEW - Relentless Aggression)
  ├── zoner.py                   (NEW - Range Control)
  ├── grappler.py                (NEW - Close Combat)
  ├── hit_and_run.py             (NEW - Mobility)
  ├── tank.py                    (EXISTING)
  ├── rusher.py                  (EXISTING)
  └── balanced.py                (EXISTING)
```

---

## Quick Testing

### Test One Fighter
```bash
python atom_fight.py fighters/examples/dodger.py fighters/examples/rusher.py
```

### Test Multiple Matchups
```bash
for opponent in fighters/examples/*.py; do
  echo "vs $(basename $opponent)"
  python atom_fight.py fighters/examples/dodger.py "$opponent" --seed 1
done
```

### Population Training with New Fighters
```bash
python train_population.py \
  --population 12 \
  --generations 20 \
  --episodes 500 \
  --opponent-pool fighters/examples/dodger.py fighters/examples/stamina_manager.py fighters/examples/counter_puncher.py \
  --rotation-frequency 2
```

---

## Summary

These 7 new fighters provide complete coverage of combat archetypes and force diverse learning:

1. **Dodger** - Teaches pursuit and prediction
2. **Stamina Manager** - Teaches resource management and pacing
3. **Counter Puncher** - Teaches patience and risk/reward
4. **Berserker** - Teaches defense and survival
5. **Zoner** - Teaches range management
6. **Grappler** - Teaches close combat
7. **Hit-and-Run** - Teaches advanced prediction

Together with Tank, Rusher, and Balanced, they form a complete training curriculum that forces well-rounded skill development.

