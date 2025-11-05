# Reward System Improvements for Population Training

## Current Issues Analysis

### Problem 1: Timeout Win Reward Too Low
**Current:** +25 for timeout win
**Issue:** 8x lower than KO victory (+200+)
**Effect:** Creates incentive for defensive "turtle" strategy
**Evidence:** Learners develop passive play patterns

### Problem 2: Inaction Penalty Inconsistent
**Current:** -0.5/tick when damage_dealt == 0 AND damage_taken == 0
**Issue:** Penalizes legitimate scouting/distance management
**Effect:** AI avoids ranges where it can't hit (even optimal ranges)
**Evidence:** Poor distance control development

### Problem 3: Missing Stamina-Based Rewards
**Current:** No stamina management incentives
**Issue:** Stamina waste not punished, efficiency not rewarded
**Effect:** AI doesn't learn stamina economy (per IMPROVEMENTS.md)
**Evidence:** Previous training showed stamina always full

### Problem 4: Missing Stance-Based Shaping
**Current:** Reward only depends on damage dealt/taken
**Issue:** All stances equally valued
**Effect:** Incorrect stance timing, never learns defending
**Evidence:** Suboptimal attack patterns

### Problem 5: Proximity Bonus Insufficient
**Current:** Capped at +0.15/tick for distance < 3.75m
**Issue:** Far too small compared to damage rewards
**Effect:** Closing distance undervalued vs damage luck
**Evidence:** Learners don't prioritize range

---

## Proposed Solutions

### Solution 1: Increase Timeout Win Reward
**Change:** +25 → +100
**Rationale:** Creates meaningful victory path
**Effect:** 
- Timeout victory becomes valuable (not 5% of KO)
- Defensive play still viable but not dominant
- Reward: 100 for timeout, 200+ for KO (2:1 ratio)

**Implementation:**
```python
# In gym_env.py, line 213
elif truncated:
    hp_pct_diff = fighter_hp_pct - opponent_hp_pct
    if hp_pct_diff > 0:
        reward = 100.0  # Changed from 25.0
    elif hp_pct_diff < 0:
        reward = -100.0
    else:
        reward = -25.0
```

### Solution 2: Stamina-Aware Rewards
**New Component:** Stamina management rewards
**Formula:**
```python
stamina_reward = 0.0

# Penalty for wasteful low-stamina attacking
if fighter_stamina_pct < 0.3 and damage_dealt == 0:
    stamina_reward -= 0.1  # Penalize risky exhaustion
    
# Reward for active recovery in neutral
if fighter_stamina_pct < 0.6 and stance == "neutral" and prev_stamina_pct < fighter_stamina_pct:
    stamina_reward += 0.05  # Reward recovery

reward += stamina_reward
```

**Effect:**
- Teaches stamina conservation
- Rewards appropriate recovery periods
- Penalizes wasteful low-stamina attacks

### Solution 3: Stance-Based Bonuses
**New Component:** Stance timing rewards
**Formula:**
```python
stance_reward = 0.0

# Reward extended stance when dealing damage
if stance == "extended" and damage_dealt > 0:
    stance_reward += 0.2 * damage_dealt

# Reward defending stance when taking damage
if stance == "defending" and damage_taken > 0:
    stance_reward += 0.1  # Acknowledgment of defense

# Reward neutral stance during recovery
if stance == "neutral" and stamina_recovery_this_tick > 0:
    stance_reward += 0.05

reward += stance_reward
```

**Effect:**
- Encourages extended stance use when attacking
- Teaches defensive stance value
- Rewards proper recovery stance

### Solution 4: Smart Inaction Penalty
**Change:** Distinguish between "passive" and "scouting"
**Current:** -0.5 for ANY zero damage tick
**Proposed:**
```python
inaction_penalty = 0.0

# Only penalize if truly passive (far away, not approaching)
distance = abs(fighter.position - opponent.position)
min_effective_range = 2.0  # Can deal damage within this

if damage_dealt == 0 and damage_taken == 0:
    if distance < min_effective_range:
        # At striking range but doing nothing = passive
        inaction_penalty = -0.5
    elif distance > min_effective_range and distance < 5.0:
        # In approach range = scouting (no penalty)
        inaction_penalty = 0.0
    else:
        # Far away = positioning (slight penalty)
        inaction_penalty = -0.1

reward += inaction_penalty
```

**Effect:**
- Allows legitimate distance management
- Still penalizes truly passive play
- Encourages active approach vs idle waiting

### Solution 5: Proximity Bonus Enhancement
**Current:** +0.15 if distance < 3.75m
**Proposed:** Graduated proximity bonus
```python
proximity_bonus = 0.0
distance = abs(fighter.position - opponent.position)

if distance < 1.0:
    # Very close - significant bonus
    proximity_bonus = 0.3
elif distance < 2.0:
    # Close - good bonus
    proximity_bonus = 0.2
elif distance < 3.5:
    # Moderate - smaller bonus
    proximity_bonus = 0.1
elif distance < 5.0:
    # Far but approaching - minimal
    proximity_bonus = 0.05

# Only apply if actively approaching or in range
if fighter.velocity > 0 or distance < 2.0:
    reward += proximity_bonus
```

**Effect:**
- Strongly rewards close range positioning
- Encourages active approach
- Scales proximity reward by distance

---

## Complete Revised Reward Function

```python
def _calculate_episode_reward(self, terminated, truncated, damage_dealt, damage_taken):
    """
    Improved reward function with stamina and stance shaping.
    """
    fighter_hp_pct = self.fighter.hp / self.fighter.max_hp
    opponent_hp_pct = self.opponent.hp / self.opponent.max_hp
    fighter_stamina_pct = self.fighter.stamina / self.fighter.max_stamina

    reward = 0.0

    # ===== EPISODE-ENDING REWARDS =====
    
    if terminated:
        if fighter_hp_pct > opponent_hp_pct:
            # WIN: Base + time bonus + HP bonus
            time_bonus = max(0, (self.max_ticks - self.tick) / 20)
            hp_diff = fighter_hp_pct - opponent_hp_pct
            hp_bonus = hp_diff * 100
            reward = 200.0 + time_bonus + hp_bonus
        elif fighter_hp_pct == opponent_hp_pct:
            # TIE: Mutual destruction penalty
            reward = -50.0
        else:
            # LOSS: Scaled by margin
            hp_diff = opponent_hp_pct - fighter_hp_pct
            hp_penalty = hp_diff * 100
            reward = -200.0 - hp_penalty
            
    elif truncated:
        # TIMEOUT: IMPROVED VALUES
        hp_pct_diff = fighter_hp_pct - opponent_hp_pct
        if hp_pct_diff > 0:
            reward = 100.0  # Changed from 25.0
        elif hp_pct_diff < 0:
            reward = -100.0
        else:
            reward = -25.0
            
    else:
        # ===== MID-EPISODE REWARDS =====
        
        # Base damage reward
        damage_reward = (damage_dealt - damage_taken) * 2.0
        reward += damage_reward
        
        # NEW: Stamina management reward
        if fighter_stamina_pct < 0.3 and damage_dealt == 0:
            # Penalize risky low-stamina moments without payoff
            reward -= 0.1
        elif fighter_stamina_pct > 0.5 and fighter_stamina_pct < 0.7:
            # Reward maintaining mid-range stamina (balanced)
            reward += 0.05
        
        # NEW: Stance-based rewards
        if self.current_stance == "extended" and damage_dealt > 0:
            # Reward extended stance when it deals damage
            reward += 0.1
        elif self.current_stance == "defending" and damage_taken > 0:
            # Reward defending stance for damage reduction
            if damage_taken < expected_damage:
                reward += 0.05
        
        # IMPROVED: Smart inaction penalty
        distance = abs(self.fighter.position - self.opponent.position)
        if damage_dealt == 0 and damage_taken == 0:
            if distance < 2.0:
                # At striking range but inactive = bad
                reward -= 0.5
            elif distance > 5.0:
                # Very far and inactive = light penalty
                reward -= 0.1
            # else: in approach range = no penalty
        
        # IMPROVED: Proximity bonus
        if distance < 1.0:
            proximity_bonus = 0.3
        elif distance < 2.0:
            proximity_bonus = 0.2
        elif distance < 3.5:
            proximity_bonus = 0.1
        else:
            proximity_bonus = 0.0
            
        if proximity_bonus > 0:
            reward += proximity_bonus
    
    return reward
```

---

## Implementation Checklist

### Phase 1: Quick Wins (Low Risk)
- [ ] Increase timeout win reward: +25 → +100
- [ ] Add smart inaction penalty (distance-aware)
- [ ] Add proximity bonus enhancement

### Phase 2: Moderate Changes (Medium Risk)
- [ ] Add stamina management rewards
- [ ] Add stance-based bonuses

### Phase 3: Full Integration
- [ ] Test with population training
- [ ] Monitor ELO progression
- [ ] Validate strategy diversity

---

## Expected Impact

### Before Improvements
- AI learns defensive "turtle" strategy
- Timeout wins rare (poor reward)
- Stamina management not learned
- Stance timing suboptimal
- Win rate vs diverse opponents: 30-40%

### After Improvements
- Multiple viable strategies (aggressive, balanced, defensive)
- Timeout wins more valuable and common
- Stamina efficiency rewarded
- Better stance usage
- Win rate vs diverse opponents: 40-55%
- Better diversity in population

---

## Validation Metrics

### Per Episode
- Track reward breakdown:
  - Terminal reward (win/loss/timeout)
  - Damage reward contribution
  - Stamina reward contribution
  - Stance reward contribution
  - Proximity reward contribution

### Population Level
- ELO progression (should improve)
- Strategy diversity (should increase)
- Win rate variance (should increase)
- Stamina management metrics (should improve)

### Behavioral
- Stance usage frequency (defending should increase)
- Timeout wins vs KO wins (should approach 1:2 ratio)
- Distance management (should improve)
- Recovery patterns (should become visible)

---

## References

- Current implementation: `training/src/gym_env.py` lines 184-250
- Related: `docs/IMPROVEMENTS.md` (stamina fixes)
- Training loop: `training/src/trainers/ppo_trainer.py`

