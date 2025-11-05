# Training Configuration Guide

## Quick Configuration Reference

### Configuration 1: Fast Baseline (4-6 hours)
```bash
python train_population.py \
  --population 8 \
  --generations 20 \
  --episodes 500 \
  --max-ticks 800 \
  --opponent-pool fighters/examples/rusher.py fighters/examples/tank.py fighters/examples/balanced.py \
  --evolution-freq 1 \
  --keep-top 0.5 \
  --output baseline_pop
```

**Result:** Baseline improvement metrics
**Pros:** Fast, reveals obvious issues
**Cons:** Limited skill development

---

### Configuration 2: Recommended Standard (12-16 hours)
```bash
python train_population.py \
  --population 12 \
  --generations 30 \
  --episodes 1000 \
  --max-ticks 1000 \
  --opponent-pool fighters/examples/training_dummy.py \
                   fighters/examples/wanderer.py \
                   fighters/examples/rusher.py \
                   fighters/examples/tank.py \
                   fighters/examples/balanced.py \
                   fighters/examples/berserker.py \
                   fighters/examples/grappler.py \
  --evolution-freq 2 \
  --keep-top 0.5 \
  --output standard_pop
```

**Result:** Good skill development, diverse strategies
**Pros:** Balanced learning, strategy diversity
**Cons:** Moderate time requirement

---

### Configuration 3: Comprehensive Training (24-32 hours)
```bash
python train_population.py \
  --population 16 \
  --generations 40 \
  --episodes 2000 \
  --max-ticks 1000 \
  --opponent-pool fighters/examples/training_dummy.py \
                   fighters/examples/wanderer.py \
                   fighters/examples/bumbler.py \
                   fighters/examples/novice.py \
                   fighters/examples/rusher.py \
                   fighters/examples/tank.py \
                   fighters/examples/balanced.py \
                   fighters/examples/dodger.py \
                   fighters/examples/stamina_manager.py \
                   fighters/examples/counter_puncher.py \
                   fighters/examples/berserker.py \
                   fighters/examples/zoner.py \
                   fighters/examples/grappler.py \
                   fighters/examples/hit_and_run.py \
  --evolution-freq 2 \
  --keep-top 0.4 \
  --mutation-rate 0.15 \
  --output comprehensive_pop
```

**Result:** Elite fighters, complex strategies
**Pros:** Best possible fighters, meta-game development
**Cons:** Very long training time

---

### Configuration 4: Curriculum-Based (Phased)

#### Phase 1: Fundamentals (4-6 hours)
```bash
python train_population.py \
  --population 8 \
  --generations 15 \
  --episodes 1000 \
  --opponent-pool fighters/examples/training_dummy.py \
                   fighters/examples/wanderer.py \
  --output phase1_pop
```

#### Phase 2: Core Combat (8-10 hours)
```bash
python train_population.py \
  --population 12 \
  --generations 20 \
  --episodes 1200 \
  --opponent-pool fighters/examples/rusher.py \
                   fighters/examples/tank.py \
                   fighters/examples/balanced.py \
                   fighters/examples/berserker.py \
  --init-from phase1_pop \
  --output phase2_pop
```

#### Phase 3: Advanced (10-12 hours)
```bash
python train_population.py \
  --population 16 \
  --generations 25 \
  --episodes 1500 \
  --opponent-pool fighters/examples/dodger.py \
                   fighters/examples/stamina_manager.py \
                   fighters/examples/counter_puncher.py \
                   fighters/examples/zoner.py \
                   fighters/examples/grappler.py \
                   fighters/examples/hit_and_run.py \
  --init-from phase2_pop \
  --output phase3_pop
```

**Total Time:** 22-28 hours
**Advantage:** Each phase builds on previous (reinforces learning)
**Disadvantage:** Manual progression management

---

## Parameter Explanations

### Population Size
- **8:** Minimum diversity, fastest training
- **12:** Balanced (recommended)
- **16:** Maximum diversity, slowest training

### Generations
- Each generation: evolve population, replace weak fighters
- More generations = more refinement
- Diminishing returns after 30+ generations

### Episodes Per Training Session
- Per fighter, per opponent matchup
- More episodes = deeper learning
- But: 2000 episodes takes 2x time vs 1000

### Max Ticks (Episode Length)
- Longer = more time to learn strategy
- Standard: 800-1000 ticks
- Shorter (300): Training is faster but less nuanced
- Longer (1500): Rewards extreme patience, slow training

### Opponent Pool Rotation
- Fighters train against random subset each generation
- More opponents = more diverse learning
- More opponents = slower training

### Evolution Frequency
- 1: Evolve every generation (aggressive)
- 2: Evolve every other generation (standard)
- 3: Evolve every 3 generations (conservative)

Higher = more stable learning, slower evolution

### Keep Top (Selection Pressure)
- 0.3: Keep top 30%, aggressive evolution
- 0.5: Keep top 50%, balanced (standard)
- 0.7: Keep top 70%, conservative

Lower = stronger selection pressure, more aggressive evolution

---

## Opponent Pool Recommendations

### For Speed (Training Dummy Only)
```python
[training_dummy.py]
```
- Fastest training
- Learns basic mechanics
- Not useful for real combat

### For Fundamentals
```python
[training_dummy.py, wanderer.py, bumbler.py, novice.py]
```
- Good baseline learning
- Covers basic patterns
- 4-8 hours training

### For Intermediate
```python
[rusher.py, tank.py, balanced.py, berserker.py, grappler.py]
```
- Core combat skills
- Covers main archetypes
- 8-12 hours training

### For Comprehensive (RECOMMENDED)
```python
[rusher.py, tank.py, balanced.py, berserker.py, grappler.py,
 dodger.py, stamina_manager.py, counter_puncher.py, zoner.py, hit_and_run.py]
```
- All archetypes covered
- Forces diverse learning
- 12-24 hours training

### For Expert (All Opponents)
```python
[training_dummy.py, wanderer.py, bumbler.py, novice.py,
 rusher.py, tank.py, balanced.py, berserker.py, grappler.py,
 dodger.py, stamina_manager.py, counter_puncher.py, zoner.py, hit_and_run.py]
```
- Maximum diversity
- Brutal training
- 20-32 hours training
- Best final results

---

## Monitoring Training

### Key Metrics to Track

#### 1. Population ELO
```
Generation 1:  Average ELO = 1500 (starting)
Generation 5:  Average ELO = 1520 (should increase)
Generation 10: Average ELO = 1560 (steady improvement)
Generation 20: Average ELO = 1600+ (good learning)
```

**Interpretation:**
- Increasing ELO = learning is working
- Stalling ELO = plateau reached or opponents too hard
- Decreasing ELO = bad changes, too aggressive evolution

#### 2. ELO Spread
```
Generation 1:  Min: 1400, Max: 1600 (Range: 200)
Generation 10: Min: 1450, Max: 1650 (Range: 200)
Generation 20: Min: 1500, Max: 1750 (Range: 250)
```

**Interpretation:**
- Increasing spread = diversity (good)
- Decreasing spread = convergence (bad)
- Target: Range > 150, StdDev > 40

#### 3. Win Rate Variance
```
Healthy: Fighter A: 45%, Fighter B: 52%, Fighter C: 60%
Bad:     Fighter A: 72%, Fighter B: 70%, Fighter C: 71%
```

**Interpretation:**
- High variance (30%+ range) = diversity (good)
- Low variance (< 10% range) = convergence (bad)

#### 4. Opponent Performance
Monitor fighter performance vs each opponent:
```
Fighter Alpha vs:
  - Rusher: 55% (good)
  - Tank: 45% (okay)
  - Dodger: 25% (weak)
  - Stamina Manager: 40% (okay)
```

**Interpretation:**
- Varied performance = versatile fighter
- Similar performance across all = specialized fighter
- Very low on any = bad matchup

---

## Expected Progression

### With Good Configuration

**Early Generations (1-5):**
- Random strategies emerge
- High variance in ELO (150+ range)
- No clear winner
- Many experiments

**Middle Generations (6-15):**
- Strategies crystallizing
- Some fighters clearly stronger
- Top fighters handle 2-3 opponents well
- Weaker fighters start to specialize

**Late Generations (16-30):**
- Meta-game emerges
- Rock-paper-scissors dynamics visible
- Top fighters handle 50%+ of opponents
- Clear skill hierarchy (ELO range > 200)

**Elite Generations (30+):**
- Refined strategies
- Top fighters 60-70% win rate vs diverse opponents
- Population diversity stable
- New strategies still emerging (slowly)

---

## Common Issues and Solutions

### Issue: ELO Not Increasing
**Causes:**
- Opponents too hard
- Evolution too aggressive
- Reward function not encouraging improvement
- Not enough training episodes

**Solutions:**
- Reduce evolution frequency (1 → 2)
- Reduce opponent difficulty
- Increase keep_top (0.5 → 0.7)
- Increase episodes per training

### Issue: All Fighters Converging (Low Variance)
**Causes:**
- Strong selection pressure
- Limited opponent variety
- Evolution too aggressive
- Not enough genetic diversity

**Solutions:**
- Increase opponent pool size
- Increase mutation rate
- Increase keep_top ratio
- Add more diverse opponents

### Issue: Training Too Slow
**Causes:**
- Too many episodes
- Too large population
- Too many opponents
- Too long episode length

**Solutions:**
- Reduce episodes (2000 → 1000)
- Reduce population (16 → 12)
- Reduce opponent pool
- Reduce max_ticks

### Issue: Fighters Weak vs All Opponents
**Causes:**
- Reward function broken
- Opponents too difficult
- Not enough training time
- Configuration mismatch

**Solutions:**
- Start with simpler opponents
- Increase training time
- Check reward function
- Use curriculum progression

---

## Advanced Tuning

### For Aggressive Evolution
```python
keep_top: 0.3        # Keep only top 30%
evolution_freq: 1    # Evolve every generation
mutation_rate: 0.2   # High mutation
```
Effect: Fast adaptation, high risk of losing good strategies

### For Conservative Learning
```python
keep_top: 0.7        # Keep top 70%
evolution_freq: 3    # Evolve every 3 generations
mutation_rate: 0.05  # Low mutation
```
Effect: Stable learning, slow evolution, less diversity risk

### For Diversity
```python
keep_top: 0.5        # Balanced
opponent_pool: [large diverse set]
mutation_rate: 0.15  # Moderate mutation
```
Effect: Multiple viable strategies, good learning

### For Speed
```python
population: 8
episodes: 500
opponent_pool: [small set]
evolution_freq: 1
```
Effect: Fast training, limited quality, good for testing

---

## Recommended Starting Configuration

For first-time training:
```bash
python train_population.py \
  --population 12 \
  --generations 30 \
  --episodes 1000 \
  --max-ticks 1000 \
  --opponent-pool fighters/examples/training_dummy.py \
                   fighters/examples/rusher.py \
                   fighters/examples/tank.py \
                   fighters/examples/balanced.py \
                   fighters/examples/dodger.py \
                   fighters/examples/stamina_manager.py \
                   fighters/examples/berserker.py \
                   fighters/examples/grappler.py \
  --evolution-freq 2 \
  --keep-top 0.5 \
  --output my_population
```

**Time Required:** 12-16 hours
**Expected Result:** Competitive fighters with diverse strategies
**Difficulty:** Medium (good learning curve)

---

## Files

- Implementation: `training/train_population.py`
- Trainer class: `training/src/trainers/population/population_trainer.py`
- ELO system: `training/src/trainers/population/elo_tracker.py`

