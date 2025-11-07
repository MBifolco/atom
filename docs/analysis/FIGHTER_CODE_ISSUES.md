# Fighter Code Issues - Technical Analysis

## Overview
This document identifies specific code problems causing poor fighter performance.

---

## BERSERKER - Code Issues

### Current Behavior
- Average Collisions: 16.8 (CRITICALLY LOW for a berserker)
- Win Rate: 0/4
- All matches timeout

### Specific Code Problems

#### Problem 1: Too Conservative When Close (Lines 59-63)
```python
# Close range - maximum aggression
if distance < 1.0:
    if my_stamina_pct > 0.3:
        return {"acceleration": 0.5, "stance": "extended"}  # TOO LOW!
    else:
        return {"acceleration": 0.0, "stance": "extended"}  # STANDING STILL!
```
**Issue:** When in optimal striking range (< 1.0m), uses acceleration of only 0.5 or 0.0. A berserker should be at 2.5+ to maintain pressure.

#### Problem 2: Backs Away at Low Stamina (Lines 42-48)
```python
# Even at low stamina, keep attacking but reduce power
if my_stamina_pct < 0.2:
    if distance < 1.5:
        return {"acceleration": 0.0, "stance": "extended"}  # WRONG!
    else:
        return {"acceleration": 1.0, "stance": "neutral"}   # WRONG!
```
**Issue:** A berserker should NEVER check stamina for attack decisions. The archetype is "relentless" - stamina should be ignored.

#### Problem 3: Mid-Range Too Slow (Lines 66-72)
```python
# Mid range - aggressive advance
elif distance < 2.5:
    if my_stamina_pct > 0.4:
        return {"acceleration": 3.0, "stance": "extended"}  # Only OK
    elif my_stamina_pct > 0.2:
        return {"acceleration": 1.5, "stance": "extended"}  # TOO SLOW
```
**Issue:** Even "aggressive advance" of 3.0 is moderate. Should be 4.5+ for a berserker.

### Required Fixes
1. Increase ALL acceleration values by 2x minimum
2. Remove stamina checks - berserker attacks regardless
3. Never use acceleration < 2.0 when attacking
4. Close range (< 1.5m) should use acceleration 2.5-3.0 minimum

---

## ZONER - Code Issues

### Current Behavior
- Average Collisions: 3.2 (CRITICAL - 3 matches had ZERO)
- Win Rate: 0/4
- All matches timeout/draw

### Specific Code Problems

#### Problem 1: Extended Stance While Retreating (Lines 62-67)
```python
# Too close for safety - back away quickly
if distance < 2.0:
    if my_stamina_pct > 0.5:
        return {"acceleration": -2.5, "stance": "extended"}  # WRONG!
    else:
        return {"acceleration": -2.0, "stance": "retracted"}
```
**Issue:** Using extended stance while moving AWAY (-2.5 acceleration) doesn't create contact. Extended stance needs forward movement or stationary position to hit.

#### Problem 2: No Stationary Poking (Lines 70-87)
```python
# Optimal zoning range (3-5m) - poke from range
elif distance >= 2.0 and distance <= 5.0:
    if my_stamina_pct > 0.4:
        if distance > 4.0:
            return {"acceleration": 1.0, "stance": "extended"}   # Too slow
        elif distance < 3.0:
            return {"acceleration": -0.5, "stance": "extended"}  # Backing away
        else:
            return {"acceleration": 0.5, "stance": "extended"}   # Too slow
```
**Issue:** Even at "perfect range," the zoner keeps backing away or moving too slowly. Should HOLD POSITION (acceleration: 0.0) and POKE (extended stance).

#### Problem 3: Excessive Retreat Logic
```python
# Lines 57-58: Retreat
# Lines 61-67: Retreat
# Lines 77-78: Retreat
# Lines 84-87: Retreat
# Lines 90-94: Slow advance only when far
```
**Issue:** 80% of the code is retreat logic. A "zoner" should zone (control space), not flee.

### Required Fixes
1. Add stationary poking: `{"acceleration": 0.0, "stance": "extended"}` at range 3-4m
2. Only retreat when distance < 2.5m (not 5.0m)
3. Apply forward pressure at optimal range: `{"acceleration": 0.5, "stance": "extended"}`
4. Remove retreat from optimal zoning range (3-5m)

---

## DODGER - Code Issues

### Current Behavior
- Average Collisions: 0.0 (ZERO in both matches!)
- Win Rate: 0/2
- All matches timeout/draw

### Specific Code Problems

#### Problem 1: Only ONE Counter-Attack Condition (Lines 57-58)
```python
# Counter-attack opportunity - opponent overextended and low stamina
if distance < 2.0 and opp_hp_pct < 0.5 and my_stamina_pct > 0.5:
    return {"acceleration": 1.5, "stance": "extended"}
```
**Issue:** This requires THREE conditions simultaneously:
- Distance < 2.0 (close)
- Opponent HP < 50%
- My stamina > 50%

This almost NEVER happens in practice. Result: zero collisions.

#### Problem 2: Weak Counter-Attack (Line 58)
```python
return {"acceleration": 1.5, "stance": "extended"}
```
**Issue:** Even when counter-attacking, acceleration is only 1.5. With opponent likely moving, this doesn't close distance fast enough.

#### Problem 3: Pure Retreat Everywhere Else
```python
# Lines 36-37: Retreat at -4.0
# Lines 42-44: Retreat at -2.0
# Lines 48-50: Retreat at -2.5
# Lines 54: Retreat at -3.5
# Lines 64-66: Retreat at -2.5
# Lines 69: Retreat at -0.5
# Lines 76: Retreat at -1.5
```
**Issue:** SEVEN different retreat conditions, ONE weak counter-attack. This is "pure dodge, no counter."

### Required Fixes
1. Add counter-attack when opponent charging: `distance < 2.5 and opp_velocity > 1.0`
2. Add counter-attack when opponent low stamina: `opp_stamina < 0.3`
3. Add counter-attack after dodging: track dodges, counter after 2-3 dodges
4. Increase counter-attack acceleration to 3.5+
5. Use extended stance with HIGH acceleration when countering

---

## Pattern Analysis: Success vs Failure

### Successful Fighters Pattern
**Counter Puncher:**
- Multiple attack conditions (lines 56, 64, 71)
- Defensive baseline with offensive opportunities
- Uses acceleration 0.5-1.5 for attacks

**Hit and Run:**
- Clear attack phase (lines 57-63)
- Attack with acceleration 1.5-2.5
- Retreat is TACTICAL, not constant

### Failed Fighters Pattern
**Common Problems:**
1. Excessive retreat frequency
2. Too many stamina/HP checks that prevent action
3. Low acceleration values even when "attacking"
4. Extended stance used while retreating (doesn't make contact)
5. Overly complex conditions that rarely trigger

### The Math Problem
**Why Zero Collisions Happen:**
```
Fighter backs away at -2.0 acceleration
Opponent advances at +1.5 acceleration
Net closing speed: 1.5 - 2.0 = -0.5 (getting farther apart!)

Even with extended stance (reach ~0.5m), they never make contact.
```

**Fix:**
```
Fighter holds position at 0.0 acceleration with extended stance
Opponent advances at +1.5 acceleration
Net closing speed: 1.5 (contact inevitable)

OR

Fighter advances at +2.0 with extended stance
Opponent advances at +1.5
Net closing speed: 3.5 (aggressive engagement)
```

---

## Recommendations Summary

### Immediate Code Changes Required

**BERSERKER:**
```python
# BEFORE: Close range
return {"acceleration": 0.5, "stance": "extended"}

# AFTER: Close range
return {"acceleration": 2.5, "stance": "extended"}

# REMOVE all stamina checks for attacking
# INCREASE all acceleration values by 2x
```

**ZONER:**
```python
# BEFORE: Optimal range
return {"acceleration": -0.5, "stance": "extended"}  # Retreating

# AFTER: Optimal range
return {"acceleration": 0.0, "stance": "extended"}   # Hold and poke

# ADD forward pressure
if distance > 4.0:
    return {"acceleration": 2.0, "stance": "extended"}
```

**DODGER:**
```python
# BEFORE: One impossible counter-attack
if distance < 2.0 and opp_hp_pct < 0.5 and my_stamina_pct > 0.5:
    return {"acceleration": 1.5, "stance": "extended"}

# AFTER: Multiple practical counter-attacks
# Counter when opponent charges
if distance < 2.5 and opp_velocity > 1.0:
    return {"acceleration": 3.5, "stance": "extended"}

# Counter when opponent exhausted
if distance < 3.0 and opp_stamina_pct < 0.3:
    return {"acceleration": 3.0, "stance": "extended"}

# Counter when opponent recovering
if distance < 2.5 and opp_velocity < -0.5:
    return {"acceleration": 3.5, "stance": "extended"}
```

---

## Testing Validation

After fixes, expect:
- Berserker: 40+ collisions (highest of all fighters)
- Zoner: 25-35 collisions (moderate engagement)
- Dodger: 20-30 collisions (tactical engagement)

All three should achieve at least 1-2 wins out of 4 matches.
