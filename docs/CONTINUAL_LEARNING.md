# Continual Learning & Catastrophic Forgetting

## The Problem

**Catastrophic forgetting** occurs when an RL model trained on a sequence of tasks forgets how to perform earlier tasks after training on later ones.

### What Happened in parzival_1.0.11

1. **Level 4 (Novice)**: Model BARELY graduated
   - Final episodes showed very close wins (1-7 HP remaining)
   - Episode 996 was even a TIE (both fighters at 0 HP)
   - Win rate just barely met the 70% requirement

2. **Level 5 (Rusher)**: Trained for 500 episodes EXCLUSIVELY against rusher
   - Rusher has aggressive, high-pressure fighting style
   - Model adapted to counter this specific behavior
   - 100% win rate against rusher achieved

3. **Graduation Tests**: FAILED to beat novice
   - Model completely forgot how to handle novice's different fighting style
   - Training detected catastrophic forgetting and stopped curriculum

## Root Cause

The curriculum was training on **only the current opponent** at each level:

```python
model = train_fighter(
    opponent_files=[level['opponent']],  # Only rusher!
    ...
)
```

Even though the code:
- Loaded the previous level's model (continual learning)
- Used a low learning rate (1e-4)

...500 episodes of focused training on rusher's unique behavior overwrote what was learned about novice.

## The Solution

**Mix ALL previous opponents into training** at each level.

### Before (parzival_1.0.11)
- Level 1: Train vs training_dummy (500 episodes)
- Level 2: Train vs wanderer ONLY (500 episodes)
- Level 3: Train vs bumbler ONLY (500 episodes)
- Level 4: Train vs novice ONLY (500 episodes)
- Level 5: Train vs rusher ONLY (500 episodes) ❌ Forgets novice!

### After (Fixed)
- Level 1: Train vs training_dummy (500 episodes)
- Level 2: Train vs [training_dummy, wanderer] (500 episodes mixed)
- Level 3: Train vs [training_dummy, wanderer, bumbler] (500 episodes mixed)
- Level 4: Train vs [training_dummy, wanderer, bumbler, novice] (500 episodes mixed)
- Level 5: Train vs [training_dummy, wanderer, bumbler, novice, rusher] (500 episodes mixed)

With 10 parallel environments, level 5 training will cycle through all 5 opponents:
- Env 0: training_dummy
- Env 1: wanderer
- Env 2: bumbler
- Env 3: novice
- Env 4: rusher
- Env 5: training_dummy (repeats)
- etc.

This ensures the model continuously practices against ALL opponent styles, not just the current level.

## Code Changes

### PPO Trainer (training/src/trainers/ppo/trainer.py:654-662)

```python
# Build opponent list: include ALL previous levels + current level
# This prevents catastrophic forgetting by mixing in previous opponents
opponent_files = []
for prev_idx in range(level_idx + 1):
    opponent_files.append(curriculum[prev_idx]['opponent'])

if len(opponent_files) > 1:
    print(f"📚 Training against {len(opponent_files)} opponents (prevents forgetting)")
    print(f"   Mix: {', '.join(Path(f).stem for f in opponent_files)}")
```

### SAC Trainer (training/src/trainers/sac/trainer.py:647-655)

Same change applied to keep both algorithms consistent.

## Expected Impact

1. **Slower per-level progress**: Model will take longer to master the current opponent since it's also fighting previous opponents
2. **Better retention**: Model will maintain ability to beat all previous opponents
3. **More robust strategy**: Model must develop generalized fighting skills, not opponent-specific exploits
4. **Graduation tests should pass**: Model won't forget previous levels

## Trade-offs

### Pros
- Prevents catastrophic forgetting
- Develops more robust, generalized fighting skills
- Graduation tests more likely to pass
- Final model can handle diverse opponent styles

### Cons
- Training takes longer (more diverse experience needed)
- May require more episodes per level to achieve win rate requirements
- Final model may be less specialized against any single opponent

## Alternative Approaches (Not Implemented)

1. **Experience Replay**: Save episodes from previous levels, replay during training
2. **Elastic Weight Consolidation (EWC)**: Protect important weights from changing
3. **Progressive Neural Networks**: Freeze old networks, add new columns for new tasks
4. **Weighted Sampling**: Focus 80% on current level, 20% on previous levels

The current solution (mix all opponents) is the simplest and most effective for our use case.

## Validation

To verify the fix works:
1. Train a new curriculum run (e.g., `parzival_1.0.12`)
2. Check console output shows: `📚 Training against N opponents (prevents forgetting)`
3. Verify graduation tests pass at each level
4. Compare final model performance vs parzival_1.0.11

## References

- Curriculum training: `OPPONENT_PROGRESSION.md`
- Reward structure: `REWARD_STRUCTURE.md`
- Original issue: parzival_1.0.11 logs showing 100% rusher win but novice failure
