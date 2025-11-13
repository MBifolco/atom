# Curriculum Graduation System

**Status**: Production (Nov 2024)
**Purpose**: Prevent lucky streaks and ensure consistent skill mastery before level advancement

---

## Overview

The graduation system determines when fighters advance between curriculum levels. It uses a **dual-requirement approach** that checks both recent performance and overall consistency to prevent lucky streaks from triggering premature graduation.

## Hardcore Graduation Requirements

Fighters must maintain **elite standards** (80-88% win rates) throughout all 5 curriculum levels:

| Level | Name | Recent WR | Overall WR | Episodes Window | Min Episodes |
|-------|------|-----------|------------|----------------|--------------|
| 1 | Fundamentals | 90% | 75% | 50 | 200 |
| 2 | Basic Skills | 88% | 73% | 50 | 300 |
| 3 | Intermediate | 85% | 70% | 50 | 400 |
| 4 | Advanced | 83% | 68% | 50 | 500 |
| 5 | Expert | 80% | 65% | 50 | 600 |

**Key Points**:
- Standards remain high (80-90%) throughout entire curriculum
- No easy levels - even final level requires 80% recent win rate
- Both requirements must pass simultaneously
- Overall threshold is always 15% below recent threshold

---

## Dual-Requirement System

### Why Two Requirements?

**Problem**: Single threshold allows lucky streaks
- Fighter has 30% overall win rate
- Gets lucky and wins last 10 episodes
- Graduates despite being consistently weak

**Solution**: Require BOTH recent AND overall thresholds
- **Recent Win Rate**: Measures current skill (last 50 episodes)
- **Overall Win Rate**: Ensures sustained performance (all episodes at level)

### How It Works

```python
def should_graduate(level):
    # Check minimum episodes first
    if episodes_at_level < level.min_episodes:
        return False

    # Check recent window
    recent_episodes = last_50_episodes
    if len(recent_episodes) < level.graduation_episodes:
        return False

    # Calculate win rates
    recent_win_rate = sum(recent_episodes) / len(recent_episodes)
    overall_win_rate = total_wins / total_episodes

    # Both must pass
    recent_passed = recent_win_rate >= level.graduation_win_rate
    overall_threshold = max(0.5, level.graduation_win_rate - 0.15)
    overall_passed = overall_win_rate >= overall_threshold

    return recent_passed and overall_passed
```

### Example Scenarios

**Scenario 1: Lucky Streak (FAILS)**
```
Episodes at level: 250
Total wins: 75 (30% overall)
Recent wins: 45/50 (90% recent)

Recent: 90% ✓ (passes 88% threshold)
Overall: 30% ✗ (needs 73%, has 30%)
Result: FAILS - Not graduating
```

**Scenario 2: Consistent Performance (PASSES)**
```
Episodes at level: 350
Total wins: 260 (74.3% overall)
Recent wins: 45/50 (90% recent)

Recent: 90% ✓ (passes 88% threshold)
Overall: 74.3% ✓ (passes 73% threshold)
Result: PASSES - Graduating!
```

**Scenario 3: Plateau After Early Success (FAILS)**
```
Episodes at level: 500
Total wins: 350 (70% overall)
Recent wins: 40/50 (80% recent)

Recent: 80% ✗ (needs 88%, has 80%)
Overall: 70% ✗ (needs 73%, has 70%)
Result: FAILS - Needs more training
```

---

## Log Spam Prevention

### Problem

Original implementation logged graduation check on EVERY episode when recent threshold was met:

```
Episode 180500: 🎓 GRADUATION CHECK ❌ FAILED (overall too low)
Episode 180501: 🎓 GRADUATION CHECK ❌ FAILED (overall too low)
Episode 180502: 🎓 GRADUATION CHECK ❌ FAILED (overall too low)
Episode 180503: 🎓 GRADUATION CHECK ❌ FAILED (overall too low)
...
```

This created thousands of log lines when approaching graduation.

### Solution

Only log graduation checks in two cases:
1. **When actually graduating** (always log this)
2. **Every 100 episodes** (when recent threshold is met)

```python
will_graduate = recent_passed and overall_passed
should_log = will_graduate or (recent_passed and episodes_at_level % 100 == 0)

if should_log:
    status = "✅ PASSED" if will_graduate else "❌ FAILED (overall too low)"
    logger.info(f"🎓 GRADUATION CHECK {status}")
    logger.info(f"   Recent WR: {recent_win_rate:.2%} (need {threshold:.1%}) {'✓' if recent_passed else '✗'}")
    logger.info(f"   Overall WR: {overall_win_rate:.2%} (need {overall_threshold:.1%}) {'✓' if overall_passed else '✗'}")
```

**Result**: Clean logs with periodic updates instead of spam.

---

## Percentage Display Precision

### Problem

Using `.1%` format caused confusion:
```
Recent WR: 75.0% (need 75.0%) ✗  # Why did this fail?
```

The actual value was 74.99%, which rounds to 75.0% but still fails the check.

### Solution

Use `.2%` format for clarity:
```
Recent WR: 74.99% (need 75.00%) ✗  # Now it's clear why it failed
```

---

## Metrics Reset Between Levels

When advancing to a new level, all metrics are reset:

```python
def advance_level():
    # Advance to next level
    self.progress.current_level += 1

    # Reset metrics
    self.progress.episodes_at_level = 0
    self.progress.wins_at_level = 0
    self.progress.recent_episodes = []

    # Log the reset for debugging
    logger.info("📊 METRICS RESET FOR NEW LEVEL:")
    logger.info(f"   Episodes at level: {self.progress.episodes_at_level}")
    logger.info(f"   Wins at level: {self.progress.wins_at_level}")
    logger.info(f"   Recent episodes buffer: {len(self.progress.recent_episodes)} episodes")
```

**Why Reset?**:
- Each level teaches different skills
- Performance against Level 1 opponents doesn't predict Level 2 performance
- Fresh start ensures graduation based on current level mastery

---

## Opponent Rotation

Each curriculum level has multiple opponents (4-9 per level). During training:

1. **Random Selection**: Opponents are randomly selected each episode
2. **Uniform Distribution**: All opponents at a level appear with equal frequency
3. **Single Episode**: Each episode uses one opponent (not mixed)
4. **Win Rate Aggregation**: Win rates are calculated across ALL opponents combined

**Example (Level 2 with 6 opponents)**:
```
Episodes 1-300: Random selection from 6 opponents
Win rate: Calculated across all 300 episodes regardless of opponent
Graduation: Must achieve 88% win rate against the MIX of all 6 opponents
```

**Why This Works**:
- Prevents overfitting to single opponent
- Ensures generalization across opponent types
- More robust fighters emerge

---

## Implementation Details

### File Location
`src/training/trainers/curriculum_trainer.py`

### Key Functions

**`should_graduate()`** - Lines 599-644
- Checks both recent and overall win rates
- Implements log spam prevention
- Returns True only when both thresholds met

**`advance_level()`** - Lines 646-704
- Resets metrics for new level
- Creates new environments for vmap (GPU)
- Updates opponent models

**`_on_step()`** - Lines 174-211
- Called after each episode
- Updates win/loss tracking
- Checks graduation criteria
- Advances level when ready

### Configuration

Graduation thresholds defined in `_create_default_curriculum()` (lines 272-362):

```python
CurriculumLevel(
    name="Basic Skills",
    opponents=[...],  # List of opponent paths
    min_episodes=300,  # Must complete at least this many
    graduation_win_rate=0.88,  # Recent threshold (88%)
    graduation_episodes=50,  # Size of recent window
    description="..."
)
```

Overall threshold calculated as:
```python
overall_threshold = max(0.5, graduation_win_rate - 0.15)
# For 88% recent → 73% overall
# For 80% recent → 65% overall
```

---

## Expected Training Progression

### Level 1: Fundamentals (90% / 75%)

**Opponents**: 4 stationary targets
**Expected Duration**: ~180,000 episodes

**Progression**:
- Episodes 1-50: Learning basics, ~30% win rate
- Episodes 50-100: Improving, ~60% win rate
- Episodes 100-150: Mastering, ~85% win rate
- Episodes 150-180K: Fine-tuning to reach 90% recent, 75% overall

**Graduation Check**:
```
Episode 180,000:
  Recent: 45/50 wins (90.00%) ✓
  Overall: 135,000/180,000 wins (75.00%) ✓
  → GRADUATES to Level 2
```

### Level 2: Basic Skills (88% / 73%)

**Opponents**: 6 simple movement patterns
**Expected Duration**: ~5,000-10,000 episodes (was 300 with old 80% requirement)

**Progression**:
- Episodes 1-100: Reset, learning new opponents, ~50% win rate
- Episodes 100-500: Improving against movement, ~70% win rate
- Episodes 500-2000: Mastering pursuit/evasion, ~85% win rate
- Episodes 2000-10K: Fine-tuning to reach 88% recent, 73% overall

**Graduation Check**:
```
Episode 8,500:
  Recent: 44/50 wins (88.00%) ✓
  Overall: 6,205/8,500 wins (73.00%) ✓
  → GRADUATES to Level 3
```

### Level 3-5: Similar Pattern

Each level requires:
1. **Learning phase** (~100-500 episodes): Win rate drops, learning new opponents
2. **Improvement phase** (~500-2000 episodes): Win rate climbs to 70-80%
3. **Mastery phase** (~2000-15000 episodes): Fine-tuning to meet both thresholds

**Total Expected Time**:
- Level 1: ~180,000 episodes (unchanged from before)
- Levels 2-5: ~40,000-60,000 episodes combined (was ~2,000 before)
- **Total**: ~220,000-240,000 episodes (vs ~182,000 before)

**Training Time**:
- CPU (8 envs): ~24 hours total
- GPU (250 envs): ~45 minutes total

---

## Troubleshooting

### Fighter Not Graduating

**Symptom**: Stuck at a level for many episodes

**Check**:
1. **Look at logs every 100 episodes** for graduation check status
2. **Recent win rate too low?** Fighter needs more training
3. **Overall win rate too low?** Fighter had rough start, needs more episodes to average up
4. **Recent passing but overall failing?** This is working as intended - prevents lucky streaks

**Solutions**:
- Keep training - overall win rate will catch up eventually
- If stuck for 50K+ episodes, check reward function
- If recent win rate not improving, may need to adjust hyperparameters

### Log Spam

**Symptom**: Thousands of graduation check messages

**Cause**: Using old version without log spam fix

**Solution**: Update to latest version (Nov 2024+) which only logs every 100 episodes

### Premature Graduation

**Symptom**: Fighter graduates despite seeming weak

**Check**:
- Are you using the old single-threshold system?
- Update to dual-requirement system (Nov 2024+)

**Prevention**:
- Dual-requirement system specifically prevents this
- Overall threshold ensures sustained performance

---

## Comparison: Old vs New System

### Old System (Before Nov 2024)

**Graduation Requirements**:
- Level 1: 90% / 75% (same)
- Level 2: 80% / 65%
- Level 3: 75% / 60%
- Level 4: 60% / 50%
- Level 5: 50% / 50%

**Problems**:
1. Later levels too easy (50-60% requirements)
2. Single threshold only (recent win rate)
3. Lucky streaks caused premature graduation
4. Log spam on every episode
5. Fighters rushed through levels 2-5 in ~2,000 episodes

**Training Time**: ~182,000 episodes total

### New System (Nov 2024+)

**Graduation Requirements**:
- Level 1: 90% / 75% (same)
- Level 2: 88% / 73% (harder)
- Level 3: 85% / 70% (harder)
- Level 4: 83% / 68% (much harder)
- Level 5: 80% / 65% (much harder)

**Improvements**:
1. Elite standards maintained throughout (80-90%)
2. Dual-requirement system (recent AND overall)
3. Lucky streaks prevented by overall threshold
4. Log spam fixed (only logs every 100 episodes)
5. Fighters must demonstrate mastery at each level

**Training Time**: ~220,000-240,000 episodes total

**Impact**:
- More training required (+20-30%)
- Much stronger fighters emerge
- Consistent performance enforced
- Better generalization to new opponents

---

## References

- Implementation: `src/training/trainers/curriculum_trainer.py`
- Progressive Training Guide: `docs/PROGRESSIVE_TRAINING.md`
- Training Configuration: `docs/TRAINING_CONFIGURATION.md`

---

## Summary

The curriculum graduation system ensures fighters achieve **consistent, sustained mastery** at each level before advancing. Key features:

✅ **Dual-requirement system** prevents lucky streaks
✅ **Hardcore standards** (80-88%) maintained throughout
✅ **Log spam prevention** for clean, readable logs
✅ **Precise percentage display** for debugging clarity
✅ **Metrics reset** between levels for fair evaluation
✅ **Multi-opponent rotation** for generalization

**Result**: Stronger, more robust fighters that truly master each level before advancing.
