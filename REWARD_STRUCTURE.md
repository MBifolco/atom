# Atom Combat - Reward Structure

Complete documentation of the reinforcement learning reward system.

## Philosophy

The reward structure is designed to:
1. **Prioritize winning** - Terminal rewards are much larger than mid-episode rewards
2. **Encourage engagement** - Small proximity bonus guides agents toward combat
3. **Reward skill** - Damage differential rewards tactical play
4. **Punish avoidance** - Inaction penalty prevents passive strategies

## Reward Breakdown

### Terminal Rewards (Episode End)

#### Win (fighter HP > opponent HP)
```python
base_reward = 200.0
time_bonus = max(0, (max_ticks - tick) / 20)  # Up to +50 for quick wins
hp_bonus = (fighter_hp_pct - opponent_hp_pct) * 100  # Up to +100 for dominant win

total = 200 + time_bonus + hp_bonus  # Range: 200 to 350
```

**Examples:**
- Perfect win (100% HP, opponent at 0%, tick 50): 200 + 47.5 + 100 = **347.5**
- Close win (60% HP, opponent at 50%, tick 500): 200 + 25 + 10 = **235**
- Narrow win (51% HP, opponent at 50%, tick 900): 200 + 5 + 1 = **206**

#### Loss (fighter HP < opponent HP)
```python
base_penalty = -200.0
hp_penalty = (opponent_hp_pct - fighter_hp_pct) * 100  # Up to -100 for bad loss

total = -200 - hp_penalty  # Range: -300 to -200
```

**Examples:**
- Total domination (0% HP, opponent at 100%): -200 - 100 = **-300**
- Close loss (50% HP, opponent at 51%): -200 - 1 = **-201**

#### Tie (both at same HP)
```python
reward = -50.0  # Discourage mutual destruction
```

#### Timeout (max_ticks reached)
```python
if fighter_hp_pct > opponent_hp_pct:
    reward = 25.0  # Winning on points (much less than KO to encourage finishing)
elif fighter_hp_pct < opponent_hp_pct:
    reward = -100.0  # Losing on points
else:
    reward = -25.0  # Exact tie (discourage stalling)
```

### Mid-Episode Rewards (Each Tick)

#### 1. Damage Differential (Primary)
```python
reward = (damage_dealt - damage_taken) * 2.0
```

**Examples:**
- Hit for 5 damage, take 0: **+10** reward
- Hit for 5 damage, take 3: **+4** reward
- Take 5 damage, deal 0: **-10** reward

**Impact:** A single good collision can give +10 to +20 reward

#### 2. Proximity Bonus (Guidance)
```python
distance = abs(fighter_position - opponent_position)
close_distance = arena_width * 0.3  # ~3.75 meters

if distance < close_distance:
    proximity_bonus = 0.15 * (1.0 - distance / close_distance)
    reward += proximity_bonus  # Range: 0.0 to +0.15
```

**Purpose:** Guides agent toward combat without dominating objective
- At distance 0: **+0.15** per tick
- At 3.75m: **+0.0** per tick
- Beyond 3.75m: **+0.0** per tick

**Max accumulation:** 0.15 × 1000 ticks = **150 total** (less than 1 good hit!)

#### 3. Inaction Penalty (Anti-Avoidance)
```python
if damage_dealt == 0 and damage_taken == 0:
    reward -= 0.5  # Strong penalty for passive play (increased from -0.2)
```

**Why -0.5 per tick?**
- Over 1000 ticks of avoidance: **-500 total**
- This is worse than fighting and losing (typically -300 to -400)
- Prevents local minimum where fighter learns to avoid all combat
- Forces engagement while still allowing tactical retreats

**Net Effect:**
- Close + attacking: +0.15 + damage_reward = **positive**
- Close + not attacking: +0.15 - 0.5 = **-0.35** (strong penalty!)
- Far away: -0.5 = **-0.5** (strong penalty)

## Reward Scales Comparison

| Scenario | Reward | Ticks to Equal 1 Hit |
|----------|--------|---------------------|
| **Win by KO** | **+200 to +350** | N/A (terminal) |
| Win by TKO (timeout) | +25 | N/A (terminal) |
| Lose match | -200 to -300 | N/A (terminal) |
| Deal 5 damage | +10 | 1 tick |
| Be close (0m) | +0.15/tick | 67 ticks |
| Be passive | **-0.5/tick** | N/A (penalty) |
| **Avoid combat for 1000 ticks** | **-500** | **Worse than losing!** |

**Key:** KO (knockout before timeout) is **8-14x more valuable** than TKO (technical knockout at timeout). This strongly encourages finishing fights decisively rather than running out the clock.

## Design Rationale

### Why Small Proximity Bonus?

**Problem:** Without proximity bonus, agents can get stuck in local minimum (avoiding all combat)

**Solution:** Small bonus (+0.15 max) that:
- Guides agent toward opponent (reward shaping)
- But doesn't override main objective
- Combined with inaction penalty (-0.2), encourages active engagement
- One good hit (+10) worth 67 ticks of proximity

### Why Damage Differential?

**Reward both attacking AND defending:**
- Deal 5, take 0: +10 (perfect execution)
- Deal 5, take 3: +4 (good trade)
- Deal 3, take 5: -4 (bad trade)
- Deal 0, take 5: -10 (purely defensive)

This teaches the agent that **skill matters** - it's not just about hitting, but hitting WITHOUT getting hit.

### Why Inaction Penalty?

Without it, agent could:
1. Get close (proximity bonus)
2. Stay neutral stance (no risk)
3. Collect small rewards forever

With inaction penalty (-0.2), agent MUST take action to get positive reward.

### Why Heavy Terminal Rewards?

Terminal rewards (±200 to ±350) are **10-20x larger** than typical mid-episode rewards. This ensures:
- Winning is the primary objective
- Tactics (mid-episode) serve strategy (winning)
- Agent doesn't optimize for points, but for victory

## Training Implications

### Early Training (Episodes 1-100)
Agent discovers:
1. Proximity bonus → move toward opponent
2. Inaction penalty → take action
3. Damage differential → hitting is good

### Mid Training (Episodes 100-1000)
Agent learns:
4. Attacking while avoiding damage → skill development
5. Stamina management → don't exhaust
6. Stance timing → when to extend/defend

### Late Training (Episodes 1000+)
Agent optimizes:
7. Winning consistently → terminal reward optimization
8. Quick wins → time bonus
9. Dominant wins → HP differential bonus

## Common Issues

### Agent Avoids Combat
**Symptoms:** Low damage dealt, timeouts, negative rewards
**Diagnosis:** Proximity bonus too small OR inaction penalty too small
**Fix:** Increase proximity bonus or inaction penalty

### Agent Trades Hits Equally
**Symptoms:** High damage both ways, mutual destruction, ties
**Diagnosis:** Not learning defensive skill
**Fix:** Increase damage differential multiplier (currently 2.0)

### Agent Spams Attacks
**Symptoms:** High stamina drain, low damage output when exhausted
**Diagnosis:** Not learning stamina management
**Fix:** Ensure stamina affects damage (already implemented in arena)

## Testing Reward Structure

Run these commands to validate rewards:
```bash
# Test proximity bonus
python -c "
from training.src.gym_env import AtomCombatEnv
# Create env and check proximity calculations
"

# Monitor training rewards
tail -f training/outputs/logs/your_model_*.log | grep Reward
```

## Future Improvements

Potential enhancements (not yet implemented):
- **Stance diversity bonus**: Reward using different stances (prevents spamming one stance)
- **Movement efficiency**: Penalize excessive movement (encourages precision)
- **Comeback bonus**: Extra reward for recovering from low HP
- **Style bonus**: Reward specific fighting styles (aggressive, defensive, etc.)
