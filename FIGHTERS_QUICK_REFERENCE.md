# New Hardcoded Fighters - Quick Reference

## Files Created

All files in: `/home/biff/eng/atom/fighters/examples/`

```
dodger.py              - Evasion specialist (kiting, prediction)
stamina_manager.py     - Resource management (pacing, efficiency)
counter_puncher.py     - Timing specialist (patience, punishing mistakes)
berserker.py           - Relentless attacker (teaches defense)
zoner.py               - Range control specialist (poking from distance)
grappler.py            - Close combat specialist (in-fighting)
hit_and_run.py         - Mobility specialist (advanced prediction)
```

## Quick Fighter Comparison

| Fighter | Distance | Style | Stamina | Defense | Best For |
|---------|----------|-------|---------|---------|----------|
| **Dodger** | Far (3-5m) | Evasion | Efficient | Frequent | Teaching pursuit |
| **Stamina Mgr** | Variable | Phased | Optimal | Strategic | Teaching pacing |
| **Counter P** | Medium (2-3m) | Patience | Moderate | High | Teaching risk/reward |
| **Berserker** | Close (0-2m) | Constant | Wasteful | Never | Teaching defense |
| **Zoner** | Far (3-5m) | Poking | Aware | Selective | Teaching distance |
| **Grappler** | Close (0-2m) | Pursuit | Aware | Strategic | Teaching in-fighting |
| **Hit-Run** | Variable | Mobile | Efficient | Selective | Teaching prediction |

## Training with New Fighters

### Quick Test (30 minutes)
```bash
cd /home/biff/eng/atom
python train_population.py \
  --population 8 \
  --generations 5 \
  --episodes 200 \
  --opponent-pool fighters/examples/rusher.py fighters/examples/dodger.py fighters/examples/berserker.py \
  --output test_pop
```

### Standard Training (12-16 hours)
```bash
python train_population.py \
  --population 12 \
  --generations 30 \
  --episodes 1000 \
  --opponent-pool fighters/examples/training_dummy.py \
                   fighters/examples/wanderer.py \
                   fighters/examples/rusher.py \
                   fighters/examples/tank.py \
                   fighters/examples/balanced.py \
                   fighters/examples/berserker.py \
                   fighters/examples/grappler.py \
  --output standard_pop
```

### Comprehensive Training (24-32 hours)
```bash
python train_population.py \
  --population 16 \
  --generations 40 \
  --episodes 2000 \
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
  --output comprehensive_pop
```

## Test Individual Fighters

```bash
# Test one matchup
python atom_fight.py fighters/examples/dodger.py fighters/examples/rusher.py

# Test all new fighters vs Tank
for f in dodger stamina_manager counter_puncher berserker zoner grappler hit_and_run; do
  echo "vs $f:"
  python atom_fight.py fighters/examples/tank.py fighters/examples/$f.py --seed 1
done

# Tournament with new fighters
python atom tournament fighters/examples/*.py
```

## Key Differences

### vs Tank (Defensive)
- **Dodger:** Evades away (mobile vs static)
- **Stamina Mgr:** Phases between attack/recovery
- **Counter P:** Waits for mistakes
- **Berserker:** Relentless pressure
- **Zoner:** Pokes from range
- **Grappler:** Forces close combat
- **Hit-Run:** Hit-retreat cycle

### vs Rusher (Aggressive)
- **Dodger:** Matches mobility, kites more
- **Stamina Mgr:** Exhausts Rusher, then attacks
- **Counter P:** Punishes overextension
- **Berserker:** Mutual aggression (stamina war)
- **Zoner:** Creates distance advantage
- **Grappler:** Also close (strength war)
- **Hit-Run:** More erratic than Rusher

### vs Balanced (Adaptive)
- **Dodger:** Evades adaptively
- **Stamina Mgr:** Better resource use
- **Counter P:** More patient
- **Berserker:** More aggressive
- **Zoner:** Better range management
- **Grappler:** Also adaptive
- **Hit-Run:** More unpredictable

## Expected Win Rates

Against a trained fighter (after learning to beat Tank/Rusher/Balanced):

| Fighter | vs Dodger | vs StaminaMgr | vs Counter | vs Berserker | vs Zoner | vs Grappler | vs Hit-Run |
|---------|-----------|---------------|-----------|--------------|----------|-------------|-----------|
| Tank | 25% | 40% | 45% | 60% | 35% | 45% | 30% |
| Rusher | 30% | 35% | 25% | 50% | 40% | 45% | 35% |
| Balanced | 35% | 45% | 50% | 65% | 45% | 50% | 40% |

(These are estimates; actual results depend on training quality)

## Implementation Steps

1. **Verify files created:** All 7 .py files in fighters/examples/
2. **Test single matches:** Run atom_fight.py with each fighter
3. **Start training:** Run standard or comprehensive training config
4. **Monitor progress:** Check logs for ELO progression and diversity
5. **Analyze results:** Compare strategies and win rates vs opponent pool

## Documentation

- **NEW_FIGHTERS_GUIDE.md** - Detailed guide for each fighter
- **REWARD_IMPROVEMENTS.md** - Reward system analysis
- **TRAINING_CONFIGURATION.md** - Training setup guide
- **ANALYSIS_SUMMARY.md** - Executive summary
- **This file** - Quick reference

## Next Actions

1. Verify files created (done: all 7 fighters confirmed)
2. Test with quick training run (30 min)
3. If successful, run standard training (12-16 hours)
4. Compare results vs baseline
5. Implement reward improvements if needed
6. Run comprehensive training (24-32 hours)

