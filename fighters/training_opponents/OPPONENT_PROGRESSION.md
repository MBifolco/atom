# Opponent Progression Guide

Training curriculum for AI fighters. Progress through levels as win rate improves.

---

## Level 1: Training Dummy ⭐
**File:** `training_dummy.py`

**Behavior:**
- Stands completely still
- Neutral stance
- Does nothing

**What AI Should Learn:**
- Move into range
- Use extended stance
- Land hits
- Basic collision mechanics

**Goal:** 100% win rate in <200 ticks

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/training_dummy.py --output level1_graduate --episodes 5000 --max-ticks 300
```

---

## Level 2: Wanderer ⭐⭐
**File:** `wanderer.py`

**Behavior:**
- Moves randomly
- Random stance changes (mostly neutral)
- No strategy
- Sometimes accidentally hits

**What AI Should Learn:**
- Positioning matters
- Predict opponent movement
- Time attacks when opponent is vulnerable
- Movement patterns

**Goal:** 90%+ win rate

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/wanderer.py --output level2_graduate --episodes 5000 --max-ticks 400
```

---

## Level 3: Bumbler ⭐⭐⭐
**File:** `bumbler.py`

**Behavior:**
- Tries to fight but poorly
- Moves toward opponent but no retreat
- Uses extended stance too early (wastes stamina)
- No stamina management
- No defensive awareness

**What AI Should Learn:**
- Timing is critical
- Stamina management matters
- Exploit poor timing
- Counter predictable aggression

**Goal:** 80%+ win rate

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/bumbler.py --output level3_graduate --episodes 7000 --max-ticks 500
```

---

## Level 4: Novice ⭐⭐⭐⭐
**File:** `novice.py`

**Behavior:**
- Competent fundamentals
- Proper range control
- Good stance timing
- Basic stamina awareness
- Wall avoidance
- But: predictable pattern, no adaptation

**What AI Should Learn:**
- Compete with competent opponent
- Exploit predictability
- Mix up timing
- Read opponent patterns

**Goal:** 70%+ win rate

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/novice.py --output level4_graduate --episodes 10000 --max-ticks 600
```

---

## Level 5: Rusher ⭐⭐⭐⭐⭐
**File:** `rusher.py` *(already exists)*

**Behavior:**
- Aggressive pressure fighter
- Constantly advances
- Strikes when close
- Backs away from walls
- Retreats when HP critical

**What AI Should Learn:**
- Counter aggression
- Use defensive stance
- Wait for openings
- Manage being pressured

**Goal:** 60%+ win rate

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/rusher.py --output level5_graduate --episodes 15000 --max-ticks 800
```

---

## Level 6: Tank ⭐⭐⭐⭐⭐⭐
**File:** `tank.py` *(already exists)*

**Behavior:**
- Defensive counter-puncher
- Maintains optimal distance (2-4m)
- Defends when opponent charges
- Counter-attacks on openings
- Strategic positioning

**What AI Should Learn:**
- Break through defense
- Create openings
- Patience
- Don't overcommit

**Goal:** 55%+ win rate

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/tank.py --output level6_graduate --episodes 20000 --max-ticks 1000
```

---

## Level 7: Balanced ⭐⭐⭐⭐⭐⭐⭐
**File:** `balanced.py` *(already exists)*

**Behavior:**
- Adaptive tactician
- Aggressive when winning
- Defensive when losing
- Smart stamina management
- Adapts to situation

**What AI Should Learn:**
- Handle adaptive opponent
- Can't exploit single pattern
- Must be versatile
- Read situation and adapt

**Goal:** 50%+ win rate (competitive)

**Training Command:**
```bash
python train_fighter.py --opponent ../fighters/balanced.py --output level7_graduate --episodes 30000 --max-ticks 1000
```

---

## Level 8: Self-Play 🏆
**File:** Your own trained fighters

**Behavior:**
- Unpredictable
- Learned strategies
- No hardcoded patterns

**What AI Should Learn:**
- Meta-game awareness
- Novel strategies
- Continuous improvement

**Goal:** Consistently improve generation over generation

**Training Command:**
```bash
# Train against previous generation
python train_fighter.py --opponent ../fighters/level7_graduate.py --output gen1 --episodes 20000

# Then train against gen1
python train_fighter.py --opponent gen1.py --output gen2 --episodes 20000

# Continue iterating
```

---

## Curriculum Training (Automated)

To automatically progress through levels:

```bash
# Train through entire curriculum
python train_curriculum.py --start-level 1 --end-level 7 --output curriculum_graduate
```

*(Future: implement this script)*

---

## Multi-Opponent Training

Once you've graduated individual levels, train against multiple:

```bash
# Train against levels 1-4 simultaneously
python train_fighter.py --opponents ../fighters/training_dummy.py ../fighters/wanderer.py ../fighters/bumbler.py ../fighters/novice.py --output multi_trained --episodes 20000

# Train against all tactical fighters
python train_fighter.py --opponents ../fighters/rusher.py ../fighters/tank.py ../fighters/balanced.py --output tactical_graduate --episodes 30000
```

---

## Expected Training Times (10 cores)

| Level | Episodes | Time | Total Cumulative |
|-------|----------|------|------------------|
| 1 | 5,000 | 10 min | 10 min |
| 2 | 5,000 | 10 min | 20 min |
| 3 | 7,000 | 15 min | 35 min |
| 4 | 10,000 | 20 min | 55 min |
| 5 | 15,000 | 30 min | 1h 25m |
| 6 | 20,000 | 45 min | 2h 10m |
| 7 | 30,000 | 1h 15m | 3h 25m |

**Full curriculum:** ~3-4 hours on 10 cores

---

## Signs of Readiness to Graduate

**Ready to move up:**
- ✅ Win rate above threshold
- ✅ Consistent performance (not lucky)
- ✅ Reward steadily increasing
- ✅ Episode length stabilized
- ✅ No further improvement (plateaued)

**Not ready yet:**
- ❌ Win rate below threshold
- ❌ High variance in results
- ❌ Still improving rapidly
- ❌ Episode length erratic

---

## Testing Your Fighter

After training at each level:

```bash
# Test against the level opponent (should win often)
python atom_fight.py your_fighter.py fighters/levelX.py --html test_levelX.html

# Test against next level opponent (preview difficulty)
python atom_fight.py your_fighter.py fighters/levelX+1.py --html preview.html

# Run multiple matches to check win rate
for i in {1..10}; do
    python atom_fight.py your_fighter.py fighters/levelX.py
done
```

---

## Philosophy

**Why this progression works:**
1. Each level teaches specific skills
2. Can't skip levels (each builds on previous)
3. Provides achievable milestones
4. Maintains motivation (consistent progress)
5. Prevents frustration (not too hard)
6. Mirrors human learning (fundamentals → tactics → strategy)

**The goal:** By level 7, your AI should be competitive with human-designed tactical fighters!
