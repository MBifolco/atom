# Comprehensive Fighter Behavior Analysis Report

## Executive Summary

After running all 49 fighter combinations (7x7 matrix) and analyzing telemetry data in detail, several critical issues have been identified that affect combat dynamics.

## Key Findings

### 1. Wall Grinding Problem
**CRITICAL ISSUE**: Multiple fighters are getting stuck against walls, particularly the right wall.

- **Berserker**: Spends 984/1000 ticks near right wall in multiple matches
- **Rusher**: Similar wall-sticking behavior (981/1000 ticks)
- **Grappler**: Also prone to wall grinding (956/1000 ticks in some matches)

**Root Cause**: Starting position for Fighter B is 10.0m (in a 12m arena), leaving only 2m to the right wall. Aggressive fighters immediately push to the wall and get stuck.

### 2. Movement Patterns by Fighter Type

#### Tank
- **Expected**: Stand ground, defensive stance, counter-attack when close
- **Actual**: ✅ Successfully stands ground, uses defending stance
- **Issues**: Sometimes gets pushed to walls by aggressive opponents

#### Rusher
- **Expected**: Constant forward pressure with extended stance
- **Actual**: ⚠️ Gets stuck on right wall due to starting position
- **Collision Rate**: High (200-350) when not wall-stuck

#### Balanced
- **Expected**: Adaptive behavior based on situation
- **Actual**: ✅ Shows varied stance usage (extended/neutral/defending)
- **Issues**: Matches often timeout due to cautious approach

#### Grappler
- **Expected**: Close distance and stick to opponent
- **Actual**: ⚠️ Wall grinding issues, but high collision rates when working
- **Collision Rate**: Variable (50-200)

#### Berserker
- **Expected**: Maximum aggression, ignore stamina, always attacking
- **Actual**: ❌ MAJOR ISSUE - Gets stuck on right wall immediately
- **Collision Rate**: Very high (700+) but due to wall grinding, not proper combat

#### Zoner
- **Expected**: Control distance, poke from range
- **Actual**: ⚠️ Low collision rates (2-30), needs more aggressive poking
- **Issues**: Too much retreat, not enough standing ground to poke

#### Dodger
- **Expected**: Dodge and counter-attack
- **Actual**: ✅ Much improved from zero collisions, now 100-900
- **Collision Rate**: Highly variable based on opponent

## Detailed Match Analysis

### High Collision Matches (Good Combat)
1. **Tank vs Tank**: 901 collisions - Mirror match with good engagement
2. **Dodger vs Berserker**: 895 collisions - But mostly wall grinding
3. **Tank vs Berserker**: 811 collisions - Wall grinding issue

### Low Collision Matches (Poor Engagement)
1. **Zoner vs Berserker**: 2 collisions - Zoner keeps distance, Berserker stuck
2. **Zoner vs Rusher**: 6-11 collisions - Both maintaining distance
3. **Zoner vs Grappler**: 8-9 collisions - Zoner successfully avoiding

### Behavioral Anomalies

#### Wall Positions Beyond Arena Bounds
Several fighters show positions > 12.0m (arena width):
- Berserker: 12.4m (physically impossible)
- Rusher: 12.4m
- Grappler: 12.2m

This suggests physics engine issues with wall collisions.

#### Excessive Passivity
In many matches, fighters spend 80-95% of time stationary:
- Tank vs Berserker: Both 94-96% stationary
- Grappler vs Zoner: Both 92-93% stationary

This is due to:
1. Wall grinding (stuck in place)
2. Close-range deadlock (both pushing against each other)
3. Excessive standoffs at distance

## Fighter-Specific Issues

### Berserker
```python
# Current issue in berserker.py:
# Near wall check at 1.0m is too close
near_left_wall = my_position < 1.0  # Should be 1.5-2.0
near_right_wall = my_position > arena_width - 1.0  # Should be arena_width - 2.0

# Wall escape acceleration too low
return {"acceleration": 4.0, "stance": "extended"}  # Should be 5.0 and maybe neutral
```

### Rusher
```python
# Similar wall detection issue
near_left_wall = my_position < 1.2  # Still too close
near_right_wall = my_position > arena_width - 1.2
```

### Zoner
```python
# Too passive at optimal range
return {"acceleration": 0.0, "stance": "extended"}  # Needs small movements to maintain distance
```

## Recommendations

### Immediate Fixes Required

1. **Fix Wall Detection Distance**
   - All fighters: Increase wall detection to 2.0m from walls
   - Use stronger acceleration (5.0) to escape walls
   - Consider using neutral/defending stance when escaping walls

2. **Adjust Starting Positions**
   - Fighter A: 3.0m (instead of 2.0m)
   - Fighter B: 9.0m (instead of 10.0m)
   - This gives both fighters more room to maneuver

3. **Fix Berserker Specifically**
   ```python
   # Wall escape should be:
   if near_left_wall:
       return {"acceleration": 5.0, "stance": "neutral"}  # Not extended
   if near_right_wall:
       return {"acceleration": -5.0, "stance": "neutral"}
   ```

4. **Improve Zoner**
   - Add small movements even at optimal range
   - More aggressive forward movement when too far
   - Better poke timing

5. **Physics Engine Check**
   - Investigate why positions exceed arena bounds
   - Check wall collision physics

### Testing Recommendations

1. Test with adjusted starting positions
2. Run matches with fixed wall detection
3. Monitor for wall grinding specifically
4. Check collision counts improve for Zoner

## Summary Statistics

### Average Collisions by Fighter (as Fighter A)
- Tank: 571
- Dodger: 549
- Berserker: 544
- Balanced: 191
- Grappler: 98
- Rusher: 97
- Zoner: 95

### Win Rates (from 7 matches each)
- Grappler: 6/7 wins
- Dodger: 5/7 wins
- Zoner: 5/7 wins
- Balanced: 4/7 wins
- Rusher: 3/7 wins
- Tank: 1/7 wins
- Berserker: 1/7 wins

### Critical Issues
- 🚨 Berserker wall grinding (984/1000 ticks at wall)
- 🚨 Starting positions too close to walls
- 🚨 Zoner collision rate too low (avg 95)
- ⚠️ Many matches timing out (1000 ticks)
- ⚠️ Excessive standoffs in some matchups

## Conclusion

While the fighters' basic behaviors have improved significantly from the initial implementation, critical issues remain:

1. **Wall grinding** is the most severe problem, affecting multiple fighters
2. **Starting positions** need adjustment to prevent immediate wall issues
3. **Zoner** needs to be more aggressive with poking
4. **Physics issues** allow positions beyond arena bounds

Once these issues are fixed, the fighter ecosystem should show:
- More dynamic movement
- Better spacing
- Fewer timeouts
- More varied combat scenarios
- Proper archetype representation