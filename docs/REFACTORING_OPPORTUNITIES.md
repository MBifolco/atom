# Refactoring Opportunities in Training Code

## Large Methods Identified

### Curriculum Trainer (src/training/trainers/curriculum_trainer.py)

1. **`_build_curriculum()`** - 100 lines (line 302)
   - **Could extract**: Individual level builders
   - **Benefit**: Each level construction testable separately
   - **Impact**: ~50 statements easier to test

2. **`update_progress()`** - 80 lines (line 577)
   - **Could extract**: Graduation check logic, win rate calculation
   - **Benefit**: Core graduation logic testable independently
   - **Impact**: ~30 statements

3. **`advance_level()`** - 73 lines (line 710)
   - **Could extract**: Environment cleanup, new env creation
   - **Benefit**: Level transition logic isolated
   - **Impact**: ~25 statements

4. **`train()`** - 73 lines (line 837)
   - **Could extract**: Training loop, progress logging
   - **Benefit**: Core training logic testable
   - **Impact**: ~30 statements

5. **`create_envs_for_level()`** - 60 lines (line 459)
   - **Could extract**: Vmap env creation, DummyVecEnv creation
   - **Benefit**: Environment setup testable separately
   - **Impact**: ~20 statements

### Population Trainer (src/training/trainers/population/population_trainer.py)

1. **`_train_single_fighter_parallel()`** - 263 lines! (line 36)
   - **Could extract**:
     - Environment setup (~50 lines)
     - Model initialization (~40 lines)
     - Training loop (~60 lines)
     - Results collection (~30 lines)
   - **Benefit**: Massive complexity reduction
   - **Impact**: ~150 statements easier to test

2. **`train_fighters_parallel()`** - 225 lines (line 676)
   - **Could extract**:
     - Process pool setup
     - Task distribution
     - Results aggregation
   - **Impact**: ~100 statements

3. **`evolve_population()`** - 139 lines (line 1074)
   - **Could extract**:
     - Fitness calculation
     - Parent selection
     - Mutation logic
   - **Impact**: ~60 statements

4. **`train()`** - 153 lines (line 1519)
   - **Could extract**:
     - Generation loop
     - Evolution step
     - Checkpointing
   - **Impact**: ~60 statements

## Total Potential Impact

**Extractable helper methods**: ~20-30
**Statements made easier to test**: ~600-800
**Estimated refactoring time**: 6-8 hours
**Testing time saved**: ~4-5 hours

## Recommendation

**For Current Session**:
Continue testing existing code to reach 50%. We're at 30%, need 960 more statements.

**For Future Refactoring**:
1. Start with `_train_single_fighter_parallel()` (263 lines → 4-5 focused methods)
2. Refactor `_build_curriculum()` (100 lines → level builders)
3. Break down `evolve_population()` (139 lines → selection/mutation/fitness)

Each refactoring would:
- ✅ Follow PERMANENT_CONTEXT (reusability over duplication)
- ✅ Make testing easier
- ✅ Improve code maintainability
- ✅ Enable better unit testing

## Priority for Next Session

**High**: Refactor `_train_single_fighter_parallel()` (biggest win)
**Medium**: Refactor `_build_curriculum()`, `evolve_population()`
**Low**: Other 50-100 line methods (already manageable)
