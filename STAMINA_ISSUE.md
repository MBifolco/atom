# Stamina Exhaustion Issue

## Summary

Stamina regeneration **is working correctly**, but trained fighters are exhausting themselves by constant high acceleration.

## Evidence

From `parzival_vs_balanced` replay analysis:

### Early Match (Ticks 100-110)
- Stamina regenerating: +0.04 to +0.05 per tick ✓
- Low acceleration: -0.10 to -0.25 m/s²
- **System working as designed**

### Mid-Late Match (Ticks 200-999)
- Stamina: **0.00 (pinned)**
- Acceleration: **4.32 m/s² (constant)**
- Stance: neutral
- **Acceleration cost > regeneration**

## The Math

At 70kg mass, neutral stance:

```
Regeneration = 0.015 × 3.5 (neutral bonus) = 0.0525 per tick
Cost (4.32 accel) = 4.32 × 0.5 (cost) × 0.0842 (dt) × 1.0 (mass factor) = 0.182 per tick

Net = -0.13 per tick (draining!)
```

Once stamina hits 0, the fighter can't recover because acceleration cost exceeds regen.

## Impact

At 0% stamina (`arena_1d.py:176-178`):
- **Damage multiplier = 0.25** (only 25% of normal damage)
- Extended stance blocked (can't attack without stamina)
- Velocity reduced by 50% when stamina hits 0

The fighter is crippling itself for 80% of the match!

## Root Cause

**Training reward structure doesn't penalize stamina exhaustion:**

From `REWARD_STRUCTURE.md`:
- Proximity bonus: +0.15/tick for being close
- Damage differential: Main reward source
- Inaction penalty: -0.2/tick for no damage

**Missing:** Penalty for low stamina or reward for stamina management

## Why Training Didn't Learn This

The AI optimized for:
1. Getting close (proximity bonus)
2. Dealing damage (damage reward)
3. Winning (terminal reward)

But never learned that **exhausting stamina reduces damage output** because:
- No explicit stamina management reward
- The stamina→damage penalty is indirect
- Short-term rewards (proximity) override long-term strategy (stamina management)

## Solutions

### Option 1: Add Stamina Management Reward

```python
# In gym_env.py step()
stamina_pct = self.fighter.stamina / self.fighter.max_stamina

# Penalty for low stamina
if stamina_pct < 0.2:
    reward -= (0.2 - stamina_pct) * 0.5  # Up to -0.1 per tick at 0%

# Small bonus for maintaining good stamina
stamina_bonus = stamina_pct * 0.05  # Up to +0.05 at 100%
reward += stamina_bonus
```

### Option 2: Increase Stamina Regeneration

```python
# In world_config.py
stamina_base_regen: float = 0.03  # Was 0.015 (double it)
stamina_neutral_bonus: float = 5.0  # Was 3.5
```

This would give: `0.03 × 5.0 = 0.15 regen/tick` (enough to cover moderate acceleration)

### Option 3: Reduce Acceleration Cost

```python
# In world_config.py
stamina_accel_cost: float = 0.3  # Was 0.5 (40% reduction)
```

### Option 4: Make Stamina More Observable

Add stamina percentage to observation space (already done) and increase its importance by normalizing to 0-1 range (already done).

## Resolution (Implemented)

**Approach taken:** Increase stamina regeneration without explicit rewards

Changed in `src/arena/world_config.py`:
```python
stamina_base_regen: float = 0.03  # Was 0.015 (doubled)
```

**New stamina economics:**
- Neutral stance regen: **0.105 per tick** (was 0.0525)
- Break-even acceleration: **2.49 m/s²** (was 1.25 m/s²)
- At max acceleration (4.38 m/s²): Net **-0.079 per tick** (was -0.13)

This allows fighters to:
1. Sustain moderate acceleration (~2.5 m/s²) indefinitely
2. Still need to manage stamina at high acceleration
3. Recover stamina much faster when resting

**Why no explicit stamina rewards:**
The fighter should learn stamina management naturally through experiencing:
- Reduced damage output at low stamina
- Inability to use extended stance at 0 stamina
- Faster recovery when resting in neutral

Adding explicit stamina rewards would make the learning artificial rather than emergent from combat mechanics.

## Tests

Created `test_stamina_regen.py` which confirms:
- ✓ Stamina regenerates in neutral with no movement
- ✓ Stamina decreases with acceleration
- ✓ Stamina recovers when resting
- ✓ Stamina caps at max_stamina

The mechanics work - the AI just needs to learn to use them!

## Curriculum Updates

Also updated training curriculum in both PPO and SAC trainers to match `OPPONENT_PROGRESSION.md`:
- **7 levels total** (was 4): Training Dummy → Wanderer → Bumbler → Novice → Rusher → Tank → Balanced
- **Progressive difficulty**: Win rate requirements from 95% down to 50%
- **Graduation tests**: Each level must maintain win rate against ALL previous levels

This ensures comprehensive training and prevents catastrophic forgetting.
