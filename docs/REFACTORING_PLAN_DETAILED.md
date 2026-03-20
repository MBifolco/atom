# Detailed Refactoring Plan for 50% Coverage

## Current Status
- Coverage: 38.35% (1,440/3,755 statements)
- Target: 50% (1,878 statements)
- Gap: 438 statements (11.65%)

## Strategy: Refactor Large Methods → Test Small Methods → Reach 50%

---

## Phase 1: Refactor `_train_single_fighter_parallel()` (263 lines)

**File**: `src/training/trainers/population/population_trainer.py:36-298`

### Current Structure (Monolithic)
```python
def _train_single_fighter_parallel(...):  # 263 lines
    # 1. Thread config (10 lines)
    # 2. Imports (15 lines)
    # 3. Config setup (10 lines)
    # 4. Opponent loading (50 lines)
    # 5. Env creation - vmap vs CPU (80 lines)
    # 6. Model init/load (40 lines)
    # 7. Training loop (30 lines)
    # 8. Results collection (28 lines)
```

### Proposed Refactoring

#### Extract: `_configure_process_threading()`
```python
def _configure_process_threading() -> None:
    """Set thread limits for subprocess to prevent CPU oversubscription."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # ... (10 lines total)
```
**Benefit**: Testable, reusable, clear purpose

#### Extract: `_load_opponent_models(opponent_data, algorithm)`
```python
def _load_opponent_models(opponent_data: List[Tuple], algorithm: str) -> List[Tuple]:
    """Load opponent models from paths for training."""
    from stable_baselines3 import PPO
    opponent_models = []
    for opp_name, opp_mass, opp_path in opponent_data:
        opp_model = PPO.load(opp_path, device="cpu")
        opponent_models.append((opp_name, opp_mass, opp_model))
    return opponent_models
```
**Benefit**: Easy to test with mock models

#### Extract: `_create_vmap_environment(fighter_mass, opponent_models, config, n_vmap_envs, max_ticks)`
```python
def _create_vmap_environment(fighter_mass, opponent_models, config, n_vmap_envs, max_ticks):
    """Create JAX vmap vectorized environment for GPU training."""
    from src.atom.training.trainers.curriculum_trainer import VmapEnvAdapter
    from src.atom.training.vmap_env_wrapper import VmapEnvWrapper

    # Configure JAX memory
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'
    # ... (30 lines)

    return VmapEnvAdapter(vmap_env)
```
**Benefit**: Vmap setup testable independently

#### Extract: `_create_cpu_environment(fighter_mass, opponent_models, config, n_envs, max_ticks)`
```python
def _create_cpu_environment(fighter_mass, opponent_models, config, n_envs, max_ticks):
    """Create CPU-based parallel environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    # ... (40 lines)

    return DummyVecEnv(env_fns)
```
**Benefit**: CPU mode testable separately

#### Extract: `_initialize_ppo_model(vec_env, model_path, logs_dir, fighter_name)`
```python
def _initialize_ppo_model(vec_env, model_path, logs_dir, fighter_name):
    """Initialize or load PPO model for training."""
    from pathlib import Path
    from stable_baselines3 import PPO

    if Path(model_path).exists():
        model = PPO.load(model_path, env=vec_env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            # ... hyperparams (20 lines)
        )
    return model
```
**Benefit**: Model creation testable with mocks

#### Extract: `_run_training_episodes(model, episodes, vec_env, fighter_name)`
```python
def _run_training_episodes(model, episodes, vec_env, fighter_name):
    """Execute training for specified number of episodes."""
    timesteps = episodes * vec_env.num_envs * 250  # rough estimate
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model
```
**Benefit**: Training loop isolated

#### Extract: `_collect_fighter_statistics(vec_env, fighter_name, opponent_models)`
```python
def _collect_fighter_statistics(vec_env, fighter_name, opponent_models):
    """Collect training statistics from environment."""
    stats = {
        "name": fighter_name,
        "episodes": get_episode_count(vec_env),
        "avg_reward": get_average_reward(vec_env),
        # ... (15 lines)
    }
    return stats
```
**Benefit**: Stats gathering testable

#### New Structure
```python
def _train_single_fighter_parallel(...):  # Now ~40 lines!
    _configure_process_threading()
    config = _reconstruct_config(config_dict)
    opponent_models = _load_opponent_models(opponent_data, algorithm)

    if use_vmap:
        vec_env = _create_vmap_environment(...)
    else:
        vec_env = _create_cpu_environment(...)

    model = _initialize_ppo_model(...)
    model = _run_training_episodes(...)
    stats = _collect_fighter_statistics(...)

    _save_final_model(model, output_path)

    return stats
```

**Impact**:
- Main function: 263 → 40 lines
- Extracted: 6-7 helper functions averaging 20-40 lines each
- Each helper: Independently testable
- Coverage gain: ~150-200 statements easier to cover

---

## Phase 2: Refactor `evolve_population()` (139 lines)

**File**: `src/training/trainers/population/population_trainer.py:1074-1212`

### Current Structure
```python
def evolve_population(self):  # 139 lines
    # 1. Fitness calculation (30 lines)
    # 2. Parent selection (25 lines)
    # 3. Mutation logic (40 lines)
    # 4. Child creation (30 lines)
    # 5. Survivor selection (14 lines)
```

### Proposed Refactoring

#### Extract: `_calculate_fitness_scores(fighters)`
```python
def _calculate_fitness_scores(self, fighters):
    """Calculate fitness scores for all fighters based on ELO and damage."""
    fitness_scores = {}
    for fighter in fighters:
        elo = self.elo_tracker.fighters[fighter.name].elo
        damage_ratio = self.elo_tracker.fighters[fighter.name].damage_ratio
        fitness = elo * 0.7 + damage_ratio * 100 * 0.3
        fitness_scores[fighter.name] = fitness
    return fitness_scores
```

#### Extract: `_select_parents(fitness_scores, num_parents)`
```python
def _select_parents(self, fitness_scores, num_parents):
    """Select parents using tournament selection."""
    # Tournament selection logic (20 lines)
    return selected_parents
```

#### Extract: `_create_child_fighter(parent, child_name, mutation_rate)`
```python
def _create_child_fighter(self, parent, child_name, mutation_rate):
    """Create child fighter from parent with mutation."""
    # Mutation logic (30 lines)
    return child_fighter
```

#### Extract: `_select_survivors(fighters, population_size)`
```python
def _select_survivors(self, fighters, population_size):
    """Select survivors for next generation."""
    # Keep top N by fitness (10 lines)
    return survivors
```

**Impact**:
- Main function: 139 → 25 lines
- Extracted: 4 helper methods
- Coverage gain: ~70 statements easier

---

## Phase 3: Refactor `_build_curriculum()` (100 lines)

**File**: `src/training/trainers/curriculum_trainer.py:302-401`

### Extract Level Builders
```python
def _create_fundamentals_level():
    """Create Level 1: Fundamentals."""
    return [
        CurriculumLevel(name="Stationary Neutral", ...),
        CurriculumLevel(name="Stationary Extended", ...),
        # ... (20 lines)
    ]

def _create_basic_skills_level():
    """Create Level 2: Basic Skills."""
    # ... (20 lines)

def _create_intermediate_level():
    """Create Level 3: Intermediate."""
    # ... (20 lines)

def _create_advanced_level():
    """Create Level 4: Advanced."""
    # ... (20 lines)

def _create_expert_level():
    """Create Level 5: Expert."""
    # ... (20 lines)

def _build_curriculum(self):  # Now 15 lines!
    levels = []
    levels.extend(_create_fundamentals_level())
    levels.extend(_create_basic_skills_level())
    levels.extend(_create_intermediate_level())
    levels.extend(_create_advanced_level())
    levels.extend(_create_expert_level())
    return levels
```

**Impact**:
- Main function: 100 → 15 lines
- Extracted: 5 level builders
- Coverage gain: ~50 statements easier

---

## Phase 4: Update Tests

After refactoring, write tests for each extracted method:

```python
# test_population_training_refactored.py

def test_configure_process_threading():
    """Test thread configuration sets environment variables."""
    _configure_process_threading()
    assert os.environ['OMP_NUM_THREADS'] == '1'

def test_load_opponent_models(tmp_path):
    """Test loading opponent models from paths."""
    # Create mock model
    # Test loading

def test_create_vmap_environment():
    """Test vmap environment creation."""
    # Mock vmap wrapper
    # Verify configuration

def test_calculate_fitness_scores():
    """Test fitness calculation from ELO and damage."""
    # ... easy to test now!
```

**Impact**: Each small method = 1-3 simple tests = +10-30 statements coverage per method

---

## Expected Outcomes

### Statements Made Testable
- Phase 1: ~200 statements (from 263-line method)
- Phase 2: ~70 statements (from 139-line method)
- Phase 3: ~50 statements (from 100-line method)
- **Total**: ~320 statements easier to test

### Coverage Projection
- Current: 38.35% (1,440 stmts)
- After refactoring tests: 45-52% (1,690-1,950 stmts)
- Confidence: HIGH (small methods = easy tests)

### Time Estimate
- Refactoring: 4-6 hours
- Writing tests for refactored code: 3-4 hours
- **Total**: 7-10 hours to reach 50%+

### Benefits Beyond Coverage
1. **Maintainability**: Smaller methods easier to understand
2. **Reusability**: Helper methods usable elsewhere
3. **Debugging**: Isolated failures easier to diagnose
4. **Follows PERMANENT_CONTEXT**: Refactor for reusability

---

## Execution Plan for Next Session

### Session 1: Core Refactoring (4-6 hours)
1. ✅ Refactor `_train_single_fighter_parallel()` → 6 methods
2. ✅ Refactor `evolve_population()` → 4 methods
3. ✅ Refactor `_build_curriculum()` → 5 level builders
4. ✅ Refactor `train_fighters_parallel()` → 3-4 methods
5. ✅ Run existing tests to ensure nothing broke

### Session 2: Test Writing (3-4 hours)
1. ✅ Write tests for all extracted methods
2. ✅ Achieve 45-50% coverage
3. ✅ Fix any failing tests
4. ✅ Document refactoring decisions

---

## Notes

**Why This Works**:
- Small methods (10-40 lines) are easy to test
- Single responsibility = simpler mocking
- Pure functions where possible
- Follows software engineering best practices

**Risk Mitigation**:
- Keep original method signatures (no breaking changes)
- Extract carefully (preserve behavior)
- Test after each extraction
- Can roll back if needed

**Alignment with PERMANENT_CONTEXT**:
- ✅ "ALWAYS refactor for reusability rather than duplicate code"
- ✅ "Write tests after you fix a bug to ensure its covered"
- ✅ "Prefer pure functions for JAX compatibility"

Ready for next session!
