# JAX Optimization Best Practices

**Goal**: Maximize training performance while maintaining code quality and stability

**Audience**: Developers using JAX for RL training acceleration

---

## Quick Reference

### Current Status
- ✅ **Phase 1**: JAX Physics Engine (validated)
- ✅ **Phase 2**: SBX Training (2.6x speedup - PRODUCTION READY)
- ✅ **Phase 3**: JAX JIT + vmap (2.15x physics with batch=500)

### Performance Achieved
- **Baseline**: 1,091 steps/sec (SB3 + Python)
- **Current**: 2,828 steps/sec (SBX + Python)
- **Speedup**: 2.59x

### Potential Gains
- **Level 1** (Multi-env): 6-8x total (2-3 days work)
- **Level 2** (vmap): 10-15x total (1-2 weeks work)
- **Level 3** (GPU): 50-500x total (2-4 weeks work, high risk)

---

## Decision Framework

### When to Optimize

Use this flowchart to decide which optimization level to pursue:

```
Is training too slow?
├─ No → Ship current code (2.6x is great!)
└─ Yes
    └─ How slow?
        ├─ 10-30 min → Probably fine, ship it
        ├─ 30min-2hr → Consider Level 1 (multi-env)
        ├─ 2-8 hours → Do Level 1, consider Level 2 (vmap)
        └─ >8 hours → Do Level 1+2, evaluate GPU
```

### Optimization Priority

**Always prioritize in this order:**

1. **Algorithm improvements** (better reward, curriculum, architecture)
   - 10-100x potential speedup
   - Examples: Better reward shaping, hierarchical RL, curriculum learning

2. **Multi-environment parallelization** (Level 1)
   - 2-3x speedup
   - 1 hour effort
   - Zero risk

3. **Code profiling** (find bottlenecks)
   - Varies
   - 2-4 hours
   - Identifies where to optimize

4. **JAX vmap** (Level 2)
   - 2-5x additional speedup
   - 1-2 days effort
   - Low risk

5. **GPU acceleration** (Level 3)
   - 10-100x potential
   - 2-5 days effort
   - High risk (experimental)

**Why this order?** Algorithm improvements give the biggest gains with least code complexity.

---

## Best Practices by Optimization Level

### Level 0: Code Quality (Always Do This)

**Profile Before Optimizing**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your training code here
trainer.train()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

**Write Tests**:
```python
# Always validate physics correctness
def test_jax_physics_parity():
    """Ensure JAX matches Python physics exactly."""
    python_result = arena_python.step(actions)
    jax_result = arena_jax.step(actions)

    assert np.allclose(python_result, jax_result, atol=1e-5)
```

**Benchmark Everything**:
```python
import time

def benchmark(fn, name, iterations=100):
    """Benchmark a function."""
    start = time.time()
    for _ in range(iterations):
        fn()
    elapsed = time.time() - start
    print(f"{name}: {elapsed/iterations*1000:.2f}ms per call")
```

### Level 1: Production JAX (Currently Here)

**Use SBX for Training**:
```python
from sbx import PPO

# Simple drop-in replacement for SB3
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    device="auto"  # Uses JAX automatically
)

# Result: 2-3x speedup, zero code changes
```

**Keep Physics Simple**:
```python
# ✅ Good: Use Python physics for simplicity
env = AtomCombatEnv(
    opponent_func,
    use_jax=False,
    use_jax_jit=False
)

# ❌ Bad: Premature optimization
# Don't use JAX physics unless you need >10x speedup
```

**Monitor Performance**:
```python
from stable_baselines3.common.callbacks import BaseCallback

class PerformanceMonitor(BaseCallback):
    """Track training throughput."""
    def __init__(self):
        super().__init__()
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % 10000 == 0:
            elapsed = time.time() - self.start_time
            sps = self.num_timesteps / elapsed
            print(f"Performance: {sps:.0f} steps/sec")
        return True
```

### Level 2: Multi-Environment Scaling

**Use SubprocVecEnv**:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create multiple parallel environments
def make_env(seed):
    def _init():
        env = AtomCombatEnv(opponent_func, seed=seed)
        return Monitor(env)
    return _init

# Use 1-2x your CPU core count
import os
n_cores = os.cpu_count()
n_envs = min(n_cores, 16)  # Cap at 16 for diminishing returns

envs = SubprocVecEnv([make_env(42 + i) for i in range(n_envs)])

# Adjust hyperparameters
model = PPO(
    "MlpPolicy",
    envs,
    n_steps=512 // n_envs,  # Keep total steps constant
    batch_size=64
)
```

**Benchmark to Find Optimal Count**:
```bash
# Test different environment counts
python benchmark_multi_env.py --envs 1 2 4 8 16 32

# Look for the sweet spot where scaling efficiency > 60%
```

**Common Pitfalls**:
- ❌ Too many envs (>32): Overhead dominates, efficiency drops
- ❌ Too few envs (<4): Not utilizing available CPU cores
- ❌ Not adjusting n_steps: Batch size becomes too large
- ✅ Sweet spot: 8-16 envs for most systems

### Level 3: JAX vmap Integration

**When to Use vmap**:
```python
# ✅ Use vmap when:
# - You need 10-20x speedup
# - Training takes >2 hours
# - You're comfortable debugging JAX

# ❌ Don't use vmap when:
# - Current speed is acceptable
# - Training takes <30 minutes
# - You need maximum stability
```

**Using VmapEnvWrapper**:
```python
from src.training.vmap_env_wrapper import VmapEnvWrapper

# Create vectorized environment
env = VmapEnvWrapper(
    n_envs=100,  # Run 100 episodes in parallel!
    opponent_decision_func=opponent_func,
    config=WorldConfig(),
    max_ticks=250
)

# Use with SBX
model = PPO("MlpPolicy", env, device="auto")
model.learn(total_timesteps=1_000_000)
```

**Debugging vmap Issues**:
```python
# Issue: "Can't JIT compile X"
# Solution: Ensure all inputs are JAX arrays or primitives

# ❌ Bad: Python strings in JIT
stance = "neutral"  # Can't JIT

# ✅ Good: Integer encoding
STANCE_NEUTRAL = 0
stance = 0  # Can JIT

# Issue: "TracerBoolConversionError"
# Solution: Use jnp.where instead of if/else

# ❌ Bad: Python control flow
if collision:
    damage = calculate_damage()

# ✅ Good: JAX conditional
damage = jnp.where(collision, calculate_damage(), 0.0)
```

**Performance Tuning**:
```python
# Tune batch size for optimal throughput
for batch_size in [50, 100, 250, 500, 1000]:
    env = VmapEnvWrapper(n_envs=batch_size, ...)
    # Benchmark...

# Look for:
# - Scaling efficiency > 50x at batch=500
# - Diminishing returns beyond batch=1000
```

### Level 4: GPU Acceleration

**Before GPU Setup, Ask**:
1. Is CPU training too slow? (>4 hours per run)
2. Do I have 1-2 days for setup/debugging?
3. Am I comfortable with experimental features?
4. Is my ROCm version compatible? (5.7+, 6.x)

**If ALL YES, proceed. If ANY NO, stick with CPU.**

**GPU Setup Checklist**:
```bash
# 1. Verify ROCm
rocm-smi --showproductname

# 2. Backup current JAX
pip freeze | grep jax > jax_backup.txt

# 3. Install JAX with ROCm
pip uninstall jax jaxlib -y
pip install --upgrade "jax[rocm61]" \
    -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# 4. Set environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# 5. Verify
python -c "import jax; print(jax.devices())"

# Success: [RocmDevice(id=0)]
# Failure: [CpuDevice(id=0)] → Rollback
```

**Using GPU in Code**:
```python
import jax

# Check GPU availability
devices = jax.devices()
has_gpu = jax.default_backend() in ['rocm', 'cuda', 'gpu']

if has_gpu:
    print(f"✅ Using GPU: {devices}")
else:
    print("⚠️  Using CPU (GPU not available)")

# Force GPU (if needed)
jax.config.update('jax_platform_name', 'rocm')

# Everything automatically uses GPU now
result = vmap_step_batch(states, actions)  # Runs on GPU!
```

**GPU Troubleshooting**:
```python
# Issue: Out of memory
# Solution: Reduce batch size or enable memory preallocation

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Issue: Slow GPU performance
# Solution: Ensure batch size is large enough (>100)

# Small batches: GPU overhead dominates
env = VmapEnvWrapper(n_envs=10)  # ❌ Too small for GPU

# Large batches: GPU parallelism utilized
env = VmapEnvWrapper(n_envs=500)  # ✅ Good for GPU
```

---

## Code Patterns

### Pattern 1: Functional JAX (Required for JIT)

```python
# ❌ Bad: Mutable state
class Fighter:
    def __init__(self):
        self.hp = 100

    def take_damage(self, amount):
        self.hp -= amount  # Mutation!

# ✅ Good: Immutable state
@chex.dataclass
class FighterState:
    hp: float

def apply_damage(state: FighterState, amount: float) -> FighterState:
    return state.replace(hp=state.hp - amount)  # Returns new state
```

### Pattern 2: Integer Encoding (Required for JIT)

```python
# ❌ Bad: String enums
stance = "neutral"  # Can't index arrays

# ✅ Good: Integer encoding
STANCE_NEUTRAL = 0
STANCE_EXTENDED = 1
STANCE_RETRACTED = 2
STANCE_DEFENDING = 3

stance = STANCE_NEUTRAL  # Can index arrays

# Pre-compute stance-dependent values
stance_reach = jnp.array([1.0, 1.5, 0.5, 0.8])
reach = stance_reach[stance]  # Fast indexing
```

### Pattern 3: Conditional Logic (JAX-safe)

```python
# ❌ Bad: Python if/else in JIT
@jit
def step(state, action):
    if state.hp <= 0:  # TracerBoolConversionError!
        return state
    # ...

# ✅ Good: jnp.where
@jit
def step(state, action):
    # Calculate next state
    next_state = update_state(state, action)

    # Conditionally return original or next
    is_dead = state.hp <= 0
    return jax.tree_map(
        lambda old, new: jnp.where(is_dead, old, new),
        state,
        next_state
    )
```

### Pattern 4: Vectorization (vmap)

```python
# ❌ Bad: Python loop
def step_all_envs(states, actions):
    results = []
    for state, action in zip(states, actions):
        results.append(step_single(state, action))
    return results

# ✅ Good: vmap
from jax import vmap

step_all_envs = vmap(step_single)

# Usage
results = step_all_envs(states, actions)  # Parallel!
```

### Pattern 5: Debugging JAX

```python
# Enable debugging checks (slow, but catches errors)
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# Disable JIT for debugging
with jax.disable_jit():
    result = jax_function(inputs)  # Easier to debug

# Print in JAX (debugging only)
from jax.experimental import host_callback

@jit
def step(state):
    # Print intermediate values
    host_callback.id_print(state.hp, what="HP")
    return next_state
```

---

## Performance Checklist

### Before Optimizing
- [ ] Profile code to find bottlenecks
- [ ] Write correctness tests
- [ ] Establish baseline performance
- [ ] Document current training time

### Level 1: Production Ready
- [ ] Replace SB3 with SBX (2-3x speedup)
- [ ] Verify training still works
- [ ] Benchmark new performance
- [ ] Update documentation

### Level 2: Multi-Environment
- [ ] Implement SubprocVecEnv
- [ ] Benchmark 1, 2, 4, 8, 16, 32 envs
- [ ] Find optimal environment count
- [ ] Adjust hyperparameters (n_steps)
- [ ] Verify scaling efficiency >60%

### Level 3: JAX vmap
- [ ] Convert physics to pure JAX (no strings, immutable)
- [ ] Add JIT compilation
- [ ] Test physics parity (Python vs JAX)
- [ ] Implement VmapEnvWrapper
- [ ] Benchmark batch sizes (50, 100, 500, 1000)
- [ ] Integrate with SBX
- [ ] Verify 10-20x total speedup

### Level 4: GPU
- [ ] Verify ROCm installation
- [ ] Backup current setup
- [ ] Install JAX with ROCm support
- [ ] Verify GPU detection
- [ ] Benchmark GPU vs CPU
- [ ] Document GPU-specific issues
- [ ] Decide: keep or rollback

---

## Common Mistakes

### Mistake 1: Premature Optimization

```python
# ❌ Bad: Optimizing without profiling
"My code is slow, let me rewrite everything in JAX!"

# ✅ Good: Profile first
profiler.run('trainer.train()')
# Oh, 80% of time is in reward calculation, not physics!
# Optimize reward calculation instead
```

### Mistake 2: Breaking Correctness

```python
# ❌ Bad: Optimization changes behavior
# Before: reward = hp_diff * 10
# After: reward = hp_diff * 100  # Oops, changed reward scale!

# ✅ Good: Test correctness
def test_physics_parity():
    assert np.allclose(python_result, jax_result, atol=1e-5)
```

### Mistake 3: Not Measuring

```python
# ❌ Bad: Assume optimization worked
"I added JAX, it must be faster!"

# ✅ Good: Benchmark
baseline = benchmark_training()
optimized = benchmark_training_with_jax()
speedup = baseline / optimized
print(f"Speedup: {speedup:.2f}x")
```

### Mistake 4: Over-Optimizing

```python
# ❌ Bad: Spending 2 weeks for 10% gain
# Time spent: 80 hours
# Speedup: 1.1x
# Was it worth it? No.

# ✅ Good: Effort vs reward
# Time spent: 2 hours
# Speedup: 2.6x (SBX)
# Was it worth it? Yes!
```

### Mistake 5: GPU for Small Batches

```python
# ❌ Bad: GPU with 1-10 episodes
env = VmapEnvWrapper(n_envs=1)  # GPU slower than CPU!

# ✅ Good: GPU with 100+ episodes
env = VmapEnvWrapper(n_envs=500)  # GPU 10-50x faster
```

---

## Recommended Workflow

### Week 1: Establish Baseline
```bash
# Day 1: Profile and benchmark
python benchmark_end_to_end.py

# Day 2: Implement SBX
# Replace SB3 with SBX in trainers

# Day 3: Verify and benchmark
python benchmark_end_to_end.py
# Verify 2-3x speedup

# Day 4-5: Production testing
# Run full curriculum with SBX
# Ensure everything works
```

### Week 2: Multi-Environment (Optional)
```bash
# Day 1: Implement SubprocVecEnv
# Modify trainer to support multiple envs

# Day 2: Benchmark scaling
python benchmark_multi_env.py --envs 1 2 4 8 16 32

# Day 3: Find optimal count
# Identify best env count (probably 8-16)

# Day 4: Integrate
# Update trainer with optimal env count

# Day 5: Verify
# Run full training, verify 6-8x total speedup
```

### Week 3: vmap Integration (Optional, Advanced)
```bash
# Day 1-2: Convert physics to JAX
# Remove strings, make immutable, add JIT

# Day 3: Test physics parity
python tests/test_jax_physics_parity.py

# Day 4: Implement VmapEnvWrapper
# Create vectorized environment wrapper

# Day 5: Benchmark
python benchmark_jax_vmap.py

# Day 6-7: Integrate with SBX
# Adapt wrapper for SBX, test training
```

### Week 4: GPU Setup (Optional, Expert)
```bash
# Day 1: Research compatibility
# Check ROCm version, JAX support

# Day 2: Attempt installation
# Follow GPU_SETUP_GUIDE.md

# Day 3-4: Debug issues
# Fix library paths, version conflicts

# Day 5: Benchmark
# Compare CPU vs GPU performance

# Decide: Keep or rollback
```

---

## Decision Matrix

| Training Time | Recommended Action | Expected Gain | Effort |
|--------------|-------------------|---------------|---------|
| <10 minutes | Ship it! | N/A | 0 |
| 10-30 min | Use SBX (already done) | 2.6x | Done |
| 30min-2hr | Add multi-env (8-16) | 6-8x | 1 day |
| 2-4 hours | Multi-env + consider vmap | 10-15x | 1 week |
| 4-8 hours | Multi-env + vmap | 15-20x | 2 weeks |
| >8 hours | All + evaluate GPU | 50-500x | 3-4 weeks |

---

## Performance Targets

### Realistic Targets (CPU)

| Level | Configuration | Steps/sec | Total Speedup |
|-------|--------------|-----------|---------------|
| 0 | SB3 + Python | 1,091 | 1.0x (baseline) |
| 1 | SBX + Python | 2,828 | 2.6x ✅ |
| 2 | SBX + 16 envs | 7,000 | 6.4x |
| 3 | SBX + vmap(100) | 15,000 | 13.7x |

### Optimistic Targets (GPU)

| Level | Configuration | Steps/sec | Total Speedup |
|-------|--------------|-----------|---------------|
| 1 | SBX + GPU | 5,000 | 4.6x |
| 2 | SBX + GPU + vmap(500) | 50,000 | 45.8x |
| 3 | PureJaxRL + GPU | 100,000+ | 91.7x |

---

## Resources

### Documentation
- `docs/JAX_OPTIMIZATION_ROADMAP.md` - Complete optimization roadmap
- `docs/GPU_SETUP_GUIDE.md` - GPU installation guide
- `docs/INTEGRATION_AND_GPU_RESULTS.md` - Results and analysis

### Benchmarks
- `benchmark_end_to_end.py` - End-to-end training benchmark
- `benchmark_multi_env.py` - Multi-environment scaling
- `benchmark_jax_vmap.py` - JAX vmap parallelization
- `demo_jax_scaling.py` - Complete scaling demonstration

### Code
- `src/arena/arena_1d_jax_jit.py` - JAX JIT physics engine
- `src/training/vmap_env_wrapper.py` - Vectorized environment wrapper
- `src/training/gym_env.py` - Gym environment with JAX support

### Tests
- `tests/test_jax_physics_parity.py` - Physics correctness tests
- `test_jax_jit.py` - JIT compilation tests
- `test_vmap_wrapper.py` - vmap wrapper tests

---

## Summary

**What We've Achieved**:
- ✅ 2.6x speedup with SBX (production ready)
- ✅ JAX physics engine validated
- ✅ JIT + vmap proven (2.15x with parallelization)
- ✅ Complete optimization roadmap

**What's Available**:
- 🎯 Multi-env scaling (6-8x total, 1 day)
- 🎯 vmap integration (10-15x total, 1-2 weeks)
- 🎯 GPU acceleration (50-500x total, 2-4 weeks, risky)

**Recommendation**:
1. **Ship current code** with SBX (2.6x is great!)
2. **If training too slow**, add multi-env (1 day work, 2.5x additional gain)
3. **If still too slow**, evaluate vmap and GPU based on time budget

**Remember**: Premature optimization is the root of all evil. Optimize only when needed, measure everything, and maintain correctness!
