# Training Code Cleanup Plan

## Dead Code Found (Can Be Removed)

### 1. SBX (Stable-Baselines JAX) - ~50 lines
**Location**: `curriculum_trainer.py:20-27`
```python
# Remove this:
try:
    from sbx import PPO
    _using_sbx = True
except ImportError:
    from stable_baselines3 import PPO
    _using_sbx = False
```

**Replace with**: Just `from stable_baselines3 import PPO`

**Why**: SBX requires JAX < 0.7.0, incompatible with our ROCm JAX 0.7.1. We never use it successfully.

**Impact**: ~20 statements removed, cleaner imports

---

### 2. SAC Imports in Population Trainer - ~10 lines
**Location**: `population_trainer.py:21, 86, 105`
```python
from stable_baselines3 import PPO, SAC  # Remove SAC
```

**Why**: We removed SAC trainer entirely

**Impact**: ~5 statements

---

### 3. Retracted Stance in Training Code - ~30 lines
**Locations**:
- `vmap_env_wrapper.py:144` - stance_names includes "retracted"
- `vmap_env_wrapper.py:126` - action space still 0-3.99
- `opponents_jax.py:28-30` - stationary_retracted_jax function
- `opponents_jax.py:267, 295, 410` - retracted references
- `replay_recorder.py:394, 399` - old stance mapping comments

**Why**: We removed retracted from the arena (3-stance system now)

**Impact**: ~20 statements, fixes inconsistency

---

### 4. use_jax/use_jax_jit Flags - ~20 lines
**Location**: `gym_env.py:46-47, 70-71`
```python
use_jax: bool = False,
use_jax_jit: bool = False
```

**Why**: We deleted old arena implementations (arena_1d.py, arena_1d_jax.py). Only Arena1DJAXJit exists now.

**Impact**: ~15 statements, simpler API

---

### 5. Device='auto' Logic - ~15 lines
**Location**: `curriculum_trainer.py:532, 558`
```python
actual_device = "cpu" if self.device == "auto" else self.device
```

**Why**: Always forces CPU anyway. GPU training uses JAX physics, not PyTorch models.

**Impact**: ~10 statements, simpler code

---

### 6. SubprocVecEnv in Curriculum - ~30 lines
**Location**: `curriculum_trainer.py:500-521`
**Comments say**: "ALWAYS use DummyVecEnv... SubprocVecEnv has too many pickle/process issues"

**Why**: Dead code path, never used

**Impact**: ~20 statements

---

### 7. Old Stance Indices in Vmap - ~5 lines
**Location**: `vmap_env_wrapper.py:559`
```python
# Comment says: 0=neutral, 1=extended, 2=retracted, 3=defending
# Should be: 0=neutral, 1=extended, 2=defending
```

**Impact**: Documentation fix

---

## Total Removable

**Estimated statements**: ~120-150
**Files affected**: 5-6
**New coverage after cleanup**: ~31-32% (with same tests)
**Statements to test for 50%**: ~860 instead of ~960

## Cleanup Priority

**High Priority (Do Now)**:
1. ✅ Remove SBX imports/fallback (~20 stmts)
2. ✅ Remove SAC imports (~5 stmts)
3. ✅ Remove retracted stance code (~20 stmts)
4. ✅ Remove use_jax/use_jax_jit flags (~15 stmts)

**Medium Priority**:
5. Remove device='auto' logic (~10 stmts)
6. Remove SubprocVecEnv dead code (~20 stmts)

**Low Priority**:
7. Fix stance documentation

## Benefits

1. **Less code to test** (~120-150 fewer statements)
2. **Simpler codebase** (no confusing legacy options)
3. **Clearer intent** (one way to do things)
4. **Easier to reach 50%** coverage target
5. **Follows PERMANENT_CONTEXT** (refactor for reusability, remove duplication)

## Risk Assessment

**Risk**: LOW
- All removed code is:
  - Unused (dead paths)
  - Legacy (superseded by better implementations)
  - Documented as problematic (SubprocVecEnv comment)

**Testing**: Current tests will verify nothing breaks

## Recommendation

Clean up items 1-4 now (saves ~60 statements), then continue testing. This gets us closer to 50% without writing more tests.
