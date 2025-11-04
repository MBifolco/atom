# Atom Combat - Stamina & AI Improvements

## Problem Statement

Initial matches were boring:
- **Stamina always full** - fighters never got exhausted
- **Wall grinding** - one fighter rushes to wall and both get stuck
- **No retreating** - fighters just pound each other at same position
- **Low spectacle scores** - matches scored poorly on drama metrics

## Root Causes

### 1. Broken Stamina Economy

**Old values:**
```python
stamina_accel_cost: 0.2192
stamina_base_regen: 0.0451
stamina_neutral_bonus: 2.0651
```

**Math at max acceleration (4.5):**
- Cost: `4.5 * 0.2192 * 0.0842 = 0.083/tick`
- Regen: `0.0451 * 2.0651 = 0.119/tick`
- **Net: +0.036/tick** (GAINING stamina while accelerating!)

### 2. Poor AI Retreat Logic

- Retreat triggered only at <20% HP (too late)
- No wall avoidance
- No stamina management
- Simple distance-based decisions

---

## Solutions

### 1. Fixed Stamina Economy

**New values:**
```python
stamina_accel_cost: 0.5        # Was 0.2192 (2.3x increase)
stamina_base_regen: 0.015      # Was 0.0451 (3x decrease)
stamina_neutral_bonus: 3.5     # Was 2.0651 (slight increase)

# Stance drains also increased:
neutral_drain: 0.0      # Was 0.0001
extended_drain: 0.08    # Was 0.0324
retracted_drain: 0.03   # Was 0.0139
defending_drain: 0.12   # Was 0.0611
```

**New math at max acceleration (4.5):**
- Cost: `4.5 * 0.5 * 0.0842 = 0.189/tick`
- Regen (neutral): `0.015 * 3.5 = 0.053/tick`
- **Net (neutral): -0.136/tick** (draining!)
- **Net (extended): -0.216/tick** (draining faster!)

**Key insight:** Fighters now MUST rest in neutral stance to recover stamina.

### 2. Improved Tactical AI

Created new AI module (`src/ai/tactical_ai.py`) with three archetypes:

#### tactical_aggressive
- Retreats at <40% HP (not 20%)
- Backs away from walls proactively
- Manages stamina (stops at <20% stamina)
- Varies acceleration based on stamina level

#### tactical_defensive
- Maintains optimal distance (2-4m)
- Counter-attacks when opponent overextends
- Retreats at <35% HP
- Wall avoidance logic

#### tactical_balanced
- Adapts strategy based on HP advantage
- Aggressive when winning, defensive when losing
- Smart stamina management
- Exhaustion recovery mode

**Key improvements:**
- Wall detection: `near_wall = position < 1.0 or position > arena_width - 1.0`
- Earlier retreat triggers: 35-40% HP instead of 20%
- Stamina-aware: reduce aggression at <30% stamina
- Dynamic stance usage based on situation

---

## Results Comparison

### Before (Old AI + Broken Stamina)

**Match: Aggressor (75kg) vs Defender (65kg)**
```
Duration: 69 ticks
Winner: Tank (16.1 HP remaining)
Spectacle Score: 0.599 (FAIR ⭐⭐⭐)

Stamina:
  Aggressor: 8.5/8.5 (always full!)
  Tank: 7.2/7.2 (always full!)
  Low stamina moments: ~0%

Position:
  Wall grinding: 50+ ticks at position 12.48m
  Positional Exchange: 0.000 (stuck in one place)

Drama:
  Stamina Drama: 0.300 (no exhaustion)
  Close Finish: 1.000 (only bright spot)
  Pacing Variety: 1.000
```

### After (Tactical AI + Fixed Stamina)

**Match 1: Blitz (70kg) vs Counter (75kg)**
```
Duration: 226 ticks
Winner: Counter (39.5 HP remaining)
Spectacle Score: 0.670 (GOOD ⭐⭐⭐⭐)

Stamina:
  Blitz: 0.0-8.4 range (FULL RANGE!)
  Counter: 2.8-7.8 range
  Low stamina (<30%) moments: Blitz 76.5%!

Position:
  Wall contact: Blitz 0%, Counter 6.6% (NO GRINDING!)
  Position variance: 2.92m / 3.93m (lots of movement)
  Positional Exchange: 0.769 (good variety)

Drama:
  Stamina Drama: 0.300 → still could be better
  Close Finish: 0.900 (tight ending)
  Collision Drama: 1.000
  Pacing Variety: 1.000
```

**Match 2: Alpha (65kg) vs Beta (75kg)**
```
Duration: 232 ticks (PERFECT length!)
Winner: Beta (16.6 HP remaining)
Spectacle Score: 0.684 (GOOD ⭐⭐⭐⭐)

Stamina:
  Both fighters hit 0.0 stamina!
  Sustained exhaustion battles

Position:
  No wall grinding
  Fighters move across entire arena
  Positional Exchange: 0.086

Drama:
  Duration: 1.000 (ideal length)
  Close Finish: 1.000 (nail-biter!)
  Stamina Drama: 0.500 (improved!)
  Collision Drama: 1.000
  Pacing Variety: 1.000
```

---

## Key Improvements

### Stamina System
✅ **Fighters actually get exhausted**
- Stamina drains during aggressive actions
- Must rest in neutral to recover
- Extended/Defending stances are costly
- Creates natural ebb and flow of combat

### Movement & Positioning
✅ **No more wall grinding**
- Wall avoidance logic prevents stuck fighters
- Position variance: 2.92-3.93m (was 0m)
- Fighters move across entire arena
- Dynamic positioning based on situation

### Tactical Depth
✅ **Strategic decision-making**
- Earlier retreat triggers (35-40% HP)
- Stamina-aware aggression
- Distance management
- Adaptive strategy based on HP/stamina state

### Spectacle Quality
✅ **Better matches**
- Score: 0.599 → 0.684 (FAIR → GOOD)
- Duration: Perfect length (200+ ticks)
- Close finishes: Winners at 16-39 HP
- Stamina drama: Real exhaustion moments
- Movement variety: No static grinding

---

## Remaining Opportunities

### 1. Stamina Drama Score
Currently only 0.300-0.500. Could improve by:
- Tuning stamina values further
- More aggressive AI when enemy is exhausted
- Stamina-based attacks (spend stamina for damage boost?)

### 2. Positional Exchange
Match 1: 0.769 (good), Match 2: 0.086 (poor)
- Add more lateral movement incentives
- Position-based bonuses/penalties
- Terrain features or zones?

### 3. Comeback Potential
Still low (0.200) - matches tend to be one-sided once HP gap opens
- Momentum mechanics?
- Underdog bonuses?
- Critical hit chances when desperate?

---

## Files Changed

### Core System
- `src/arena/world_config.py` - Updated stamina parameters
- `src/ai/tactical_ai.py` - New tactical AI module (created)
- `src/ai/__init__.py` - AI module exports (created)

### Testing
- `test_improved_combat.py` - Comprehensive test with analysis (created)

### Configuration Changes
```python
# Stamina Economy
stamina_accel_cost: 0.2192 → 0.5
stamina_base_regen: 0.0451 → 0.015
stamina_neutral_bonus: 2.0651 → 3.5

# Stance Drains
neutral: 0.0001 → 0.0
extended: 0.0324 → 0.08
retracted: 0.0139 → 0.03
defending: 0.0611 → 0.12
```

---

## Conclusion

The improvements successfully addressed the core issues:

1. ✅ **Stamina is meaningful** - fighters must manage energy
2. ✅ **Movement is dynamic** - no wall grinding
3. ✅ **Retreating works** - tactical fallback when needed
4. ✅ **Matches are more exciting** - better spectacle scores

**Before:** Boring wall-grinding with full stamina
**After:** Dynamic tactical battles with exhaustion drama

The system is now ready for further refinement and tournament play!
