# Fighter Test Suite - Comprehensive Summary Report

**Test Date:** 2025-11-05
**Test Seed:** 42
**Total Tests Conducted:** 18 matches

---

## Executive Summary

Three fighters require immediate attention: **Berserker**, **Zoner**, and **Dodger**. All three show critically low engagement rates (avg collisions < 30) and poor win rates. The issue appears to be overly passive behavior that doesn't match their intended archetypes.

**High Performers:**
- Counter Puncher (4/4 wins, 37.5 avg collisions)
- Hit and Run (2/2 wins, 45.5 avg collisions)
- Stamina Manager (2/2 wins, 47.0 avg collisions)

---

## Detailed Test Results

### Complete Match Data

| Fighter 1        | Fighter 2 | Winner              | Collisions | Duration | HP A  | HP B  | Spectacle |
|------------------|-----------|---------------------|------------|----------|-------|-------|-----------|
| berserker        | tank      | berserker (timeout) | 8          | 1000     | 93.7  | 80.4  | 0.346     |
| berserker        | rusher    | berserker (timeout) | 23         | 1000     | 60.9  | 51.4  | 0.403     |
| berserker        | balanced  | balanced (timeout)  | 12         | 1000     | 46.7  | 60.7  | 0.474     |
| berserker        | grappler  | grappler (timeout)  | 24         | 1000     | 48.5  | 49.2  | 0.446     |
| counter_puncher  | tank      | counter_puncher     | 40         | 124      | 93.7  | 0.0   | 0.489     |
| counter_puncher  | rusher    | counter_puncher     | 37         | 111      | 73.1  | 0.0   | 0.551     |
| counter_puncher  | balanced  | counter_puncher     | 42         | 127      | 93.7  | 0.0   | 0.465     |
| counter_puncher  | grappler  | counter_puncher     | 31         | 110      | 80.8  | 0.0   | 0.495     |
| zoner            | tank      | draw (timeout)      | 13         | 1000     | 93.7  | 93.7  | 0.343     |
| zoner            | rusher    | draw (timeout)      | 0          | 1000     | 93.7  | 93.7  | 0.271     |
| zoner            | balanced  | draw (timeout)      | 0          | 1000     | 93.7  | 93.7  | 0.271     |
| zoner            | grappler  | draw (timeout)      | 0          | 1000     | 93.7  | 93.7  | 0.271     |
| dodger           | tank      | draw (timeout)      | 0          | 1000     | 93.7  | 93.7  | 0.213     |
| dodger           | rusher    | draw (timeout)      | 0          | 1000     | 93.7  | 93.7  | 0.230     |
| hit_and_run      | tank      | hit_and_run         | 60         | 337      | 93.7  | 0.0   | 0.460     |
| hit_and_run      | rusher    | hit_and_run         | 31         | 101      | 74.6  | 0.0   | 0.557     |
| stamina_manager  | tank      | stamina_manager     | 57         | 158      | 93.7  | 0.0   | 0.461     |
| stamina_manager  | rusher    | stamina_manager     | 37         | 87       | 62.6  | 0.0   | 0.590     |

---

## Fighter Performance Analysis

### 1. BERSERKER
**Win Rate:** 0/4 (0.0%)
**Avg Collisions:** 16.8
**Avg Spectacle Score:** 0.417
**Low Collision Matches:** 4/4

**Critical Issues:**
- Too few collisions - fighter is too passive for a "berserker" archetype
- No wins in 4 matches
- Consistently low engagement across all opponents
- Behavior does NOT match intended "relentless all-out attack" strategy

**Root Cause Analysis:**
Looking at the code, the Berserker has several problems:
1. Uses `acceleration: 0.0` or very low values when close (lines 61-63)
2. Even when described as "relentless," backs away at low stamina (lines 42-48)
3. Wall avoidance logic interrupts aggression (lines 51-54)
4. The "berserker" pattern is too conservative - needs MORE aggression, not less

**Archetype Mismatch:** SEVERE - Should be ultra-aggressive but is actually passive

---

### 2. COUNTER PUNCHER
**Win Rate:** 4/4 (100.0%)
**Avg Collisions:** 37.5
**Avg Spectacle Score:** 0.500
**Low Collision Matches:** 0/4

**Status:** EXCELLENT - Working as intended
- Patient and defensive until opponent overextends
- Punishes aggressive mistakes effectively
- Good balance of defense and counter-attack timing
- Behavior matches archetype perfectly

---

### 3. ZONER
**Win Rate:** 0/4 (0.0%)
**Avg Collisions:** 3.2
**Avg Spectacle Score:** 0.289
**Low Collision Matches:** 4/4

**Critical Issues:**
- Critically low collisions (avg 3.2, with THREE matches at 0!)
- No wins - all draws from timeout
- Poor spectacle score
- Behavior does NOT match "poking specialist" strategy

**Root Cause Analysis:**
The Zoner is too focused on retreat:
1. Constantly backs away (lines 58, 64, 67, 78, 87)
2. "Extended" stance used while RETREATING (line 64) - doesn't make contact
3. Only advances when distance > 5.0m, but then opponent also backs away
4. No aggressive "poke" behavior - just pure evasion
5. Never commits to actual attacks

**Archetype Mismatch:** CRITICAL - Should poke from range but just runs away

---

### 4. DODGER
**Win Rate:** 0/2 (0.0%)
**Avg Collisions:** 0.0
**Avg Spectacle Score:** 0.222
**Low Collision Matches:** 2/2

**Critical Issues:**
- ZERO collisions in both matches
- Only draws from timeout
- Extremely poor spectacle score
- Behavior does NOT match "counter-attack" intention

**Root Cause Analysis:**
The Dodger is purely evasive with no offensive capability:
1. Almost all logic is retreat-focused (lines 36, 42, 48, 50, 54, 64, 66, 76)
2. Only ONE counter-attack condition (lines 57-58) that's extremely restrictive
3. Counter-attack requires: distance < 2.0 AND opp_hp < 50% AND my_stamina > 50%
4. This condition almost never triggers in practice
5. Even the "counter-attack" accelerates AWAY from opponent (acceleration: 1.5 when should be much higher)

**Archetype Mismatch:** SEVERE - Should dodge and counter, but only dodges

---

### 5. HIT AND RUN
**Win Rate:** 2/2 (100.0%)
**Avg Collisions:** 45.5
**Avg Spectacle Score:** 0.509
**Low Collision Matches:** 0/2

**Status:** EXCELLENT - Working as intended
- Successfully implements hit-retreat cycle
- Good collision rate showing actual engagement
- Behavior matches archetype perfectly
- Balances aggression with tactical retreat

---

### 6. STAMINA MANAGER
**Win Rate:** 2/2 (100.0%)
**Avg Collisions:** 47.0
**Avg Spectacle Score:** 0.525
**Low Collision Matches:** 0/2

**Status:** EXCELLENT - Working as intended
- Best spectacle score (0.525)
- Highest average collisions (47.0)
- Manages stamina while maintaining pressure
- Good performance against varied opponents

---

## Key Findings

### Collision Rate Analysis

**Healthy Collision Rates (30+):**
- Stamina Manager: 47.0 avg
- Hit and Run: 45.5 avg
- Counter Puncher: 37.5 avg

**Unhealthy Collision Rates (<30):**
- Berserker: 16.8 avg (should be highest!)
- Zoner: 3.2 avg (critical)
- Dodger: 0.0 avg (critical)

### Archetype Fidelity

**Matches Archetype:**
- Counter Puncher: Patient, defensive, punishes mistakes ✓
- Hit and Run: Mobile, hit-retreat cycles ✓
- Stamina Manager: Manages resources, sustained pressure ✓

**Does NOT Match Archetype:**
- Berserker: Should be ultra-aggressive, is actually passive ✗
- Zoner: Should poke from range, just runs away ✗
- Dodger: Should dodge and counter, only dodges ✗

### Win Rate Patterns

**Dominant (100% win rate):**
- Counter Puncher (4/4)
- Hit and Run (2/2)
- Stamina Manager (2/2)

**Struggling (0% win rate):**
- Berserker (0/4) - mostly timeouts
- Zoner (0/4) - all timeouts/draws
- Dodger (0/2) - all timeouts/draws

---

## Recommendations (Priority Order)

### PRIORITY 1: ZONER (Priority Score: 6)
**Issues:**
- 3.2 avg collisions (most are 0!)
- 0% win rate
- Doesn't match archetype

**Required Changes:**
1. Add ACTUAL poking behavior - extend stance and HOLD POSITION at range 3-4m
2. Reduce retreat frequency - only retreat when distance < 2.5m
3. Add forward pressure when at optimal range (3-5m)
4. Change acceleration when extended from negative to positive/zero
5. Add "pressure poke" pattern: extend stance + slow forward movement

### PRIORITY 2: BERSERKER (Priority Score: 6)
**Issues:**
- 16.8 avg collisions (should be highest)
- 0% win rate
- Completely wrong behavior for archetype

**Required Changes:**
1. INCREASE all acceleration values by 2-3x
2. Remove stamina checks for attack decisions - berserker doesn't care
3. When distance < 3.0m, ALWAYS use extended stance with forward acceleration
4. Remove backing away at low stamina - commit to attack
5. Wall avoidance should still attack while moving (extended stance)

### PRIORITY 3: DODGER (Priority Score: 5)
**Issues:**
- 0.0 avg collisions
- 0% win rate
- No offensive capability

**Required Changes:**
1. Add multiple counter-attack opportunities (not just one rare condition)
2. Counter-attack when: opponent charging (dist < 2.0, opp_vel > 1.0)
3. Counter-attack when: opponent low stamina (< 30%)
4. Counter-attack acceleration should be 3.0+ (current 1.5 is too weak)
5. Add "bait and punish" pattern after dodging

---

## Testing Methodology

All tests used:
- Seed: 42 (for reproducibility)
- Max ticks: 1000
- Default masses: 70kg each
- Command: `python atom_fight.py fighters/examples/{f1}.py fighters/examples/{f2}.py --seed 42`

Metrics captured:
- Winner
- Collision count
- Match duration (ticks)
- Final HP for both fighters
- Spectacle score

---

## Conclusion

The fighter suite shows a clear divide between well-designed fighters (Counter Puncher, Hit and Run, Stamina Manager) and poorly-tuned fighters (Berserker, Zoner, Dodger).

The core issue with failing fighters is **excessive passivity**. They retreat too often and lack offensive commitment. This is particularly problematic for Berserker and Zoner, whose archetypes explicitly require aggression and pressure.

**Immediate Action Required:** Fix Zoner and Berserker to increase collision rates from single digits to 30+. These fighters should be teaching diverse combat strategies but currently teach nothing due to lack of engagement.

**Success Pattern:** Successful fighters (Hit and Run, Counter Puncher) show that mixing aggression with tactical decisions creates both good collision rates and wins. The key is having OFFENSIVE options, not just defensive ones.
