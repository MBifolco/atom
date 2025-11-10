# Mass Convergence Problem

**Date**: 2025-11-09
**Training Run**: `outputs/progressive_20251108_183723`
**Generations Analyzed**: 0-21

## Problem Summary

Population-based training has converged almost entirely to maximum mass (85.0kg), eliminating diversity and creating a monoculture of heavyweight fighters.

## Data

### Mass Distribution (15 Exported Fighters)
- **Minimum**: 75.2kg (only 1 fighter)
- **Maximum**: 85.0kg
- **Mean**: 84.0kg
- **Median**: 85.0kg

### Weight Class Breakdown
| Weight Class | Count | Percentage |
|--------------|-------|------------|
| 60-70kg | 0 | 0% |
| 70-80kg | 1 | 7% |
| 80-85kg | 3 | 20% |
| 85.0kg (max) | **11** | **73%** |

### All Fighter Masses (sorted)
```
75.2, 82.4, 83.5, 83.7, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0
```

### Recent Generations (G17-G21)
**All 4 recent fighters are at exactly 85.0kg** - showing complete convergence.

## Root Causes

### 1. HP Advantage is Overwhelming
- 85kg fighter: 125.5 HP
- 60kg fighter: 48.0 HP
- **Difference**: 2.6x more health for heavy fighters
- In combat, this massive HP pool dominates outcomes

### 2. Stamina Penalty is Too Weak
- 85kg fighter: 5.8 stamina
- 60kg fighter: 12.4 stamina
- **Trade-off**: 2.1x more stamina for light fighters
- Current reward structure doesn't sufficiently penalize low stamina
- Heavy fighters can still win without optimal stamina management

### 3. Selection Pressure Creates Positive Feedback Loop
1. Heavy fighters win more often (due to HP advantage)
2. Winners selected for breeding
3. Offspring inherit heavy mass (with small variation ±5kg)
4. Mass variation gets clamped at 85kg ceiling
5. Population converges to maximum mass
6. Diversity eliminated

### 4. Evolution Mutation is Insufficient
From `population_trainer.py:859-860`:
```python
mass_variation = np.random.uniform(-5, 5)
new_mass = np.clip(parent.mass + mass_variation, *self.mass_range)
```
- Variation: ±5kg
- If parent is 85kg, offspring can be 80-85kg (not 80-90kg)
- Ceiling at 85kg creates one-way pressure toward maximum

## Consequences

### Loss of Strategic Diversity
All fighters converge to similar strategy:
- High HP tank approach
- Absorb damage and outlast opponent
- Stamina management less critical
- Less interesting combat dynamics

### Reduced Training Effectiveness
- Population becomes homogeneous
- Self-play between similar fighters = limited learning
- No exposure to diverse strategies
- Champions don't face variety in opponents

### Unrealistic Meta
In a real competitive environment, there should be:
- **Heavyweights**: High HP, low stamina (tanks)
- **Middleweights**: Balanced stats (all-rounders)
- **Lightweights**: Low HP, high stamina (agile fighters)

Different weight classes should have different viable strategies.

## Potential Solutions (Not Yet Implemented)

### Option 1: Mass-Based ELO Brackets
- Separate ELO ratings per 10kg weight class
- Fighters only compete within their bracket
- Selection pressure within each bracket
- Maintains diversity across brackets

### Option 2: Increase Stamina Importance
- Stronger penalties for running out of stamina
- Higher rewards for stamina-efficient combat
- Make stamina regeneration slower
- Increase stamina costs for acceleration/stances

### Option 3: Mass Handicapping in Rewards
- Bonus rewards for lighter fighters winning
- Penalty for heavier fighters winning (expected to win)
- Similar to ELO: upset victories worth more

### Option 4: Enforce Mass Diversity in Selection
- Always keep at least 2 fighters in each weight bracket
- Protected slots for underrepresented masses
- Prevents complete convergence

### Option 5: Different Training Opponents by Mass
- Train heavy fighters against other heavy fighters
- Train light fighters against other light fighters
- Prevents cross-weight dominance from eliminating light fighters

### Option 6: Widen Mass Variation in Evolution
- Increase mutation: ±5kg → ±10kg
- Allow exploration of full mass range
- Risk: doesn't fix selection pressure problem

## Recommendation Priority

1. **High Priority**: Increase stamina importance (Option 2) - addresses root cause
2. **High Priority**: Mass-based ELO brackets (Option 1) - preserves diversity
3. **Medium Priority**: Mass handicapping (Option 3) - balances selection
4. **Low Priority**: Enforce diversity (Option 4) - band-aid solution
5. **Low Priority**: Widen variation (Option 6) - doesn't address selection pressure

## Configuration Parameters

**Current Settings:**
```python
mass_range: (60.0, 85.0)  # 25kg range
mass_variation: ±5kg      # Evolution mutation
```

**HP/Stamina Formulas** (from `world_config.py`):
```python
hp_min: 47.9535
hp_max: 125.4919
stamina_min: 5.7635
stamina_max: 12.3595

# HP increases with mass
hp = hp_min + (mass - min_mass) * (hp_max - hp_min) / (max_mass - min_mass)

# Stamina decreases with mass
stamina = stamina_max - (mass - min_mass) * (stamina_max - stamina_min) / (max_mass - min_mass)
```

## Next Steps

1. Analyze generation 29 results to see if trend continues
2. Decide on solution approach
3. Implement changes
4. Re-run training with diversity-preserving mechanisms
5. Monitor mass distribution across generations
