# JAX Optimization Roadmap: Getting Full Value

**Goal**: Extract maximum performance from JAX + GPUs + parallelization

**Current Status**: 2.59x speedup (SBX only)
**Potential**: 100-1000x speedup with full JAX pipeline + GPU

---

## Understanding the Performance Landscape

### Current Performance (CPU)

| Configuration | Performance | Use Case |
|--------------|-------------|----------|
| Python + SB3 | 1,091 steps/sec | Baseline |
| Python + SBX | 2,828 steps/sec | **Current best** |
| JAX Physics (single) | 10,065 tps | Too small to benefit |
| JAX Physics (batch=500) | 122,947 tps | Needs integration |

### The JAX Performance Hierarchy

```
Level 0: Python (baseline)
  ↓ 2.6x
Level 1: SBX Training (JAX NN + Python physics)    ← You are here
  ↓ 2-5x
Level 2: SBX + vmap Physics (JAX NN + batched JAX physics)
  ↓ 5-10x
Level 3: Full JAX Pipeline (end-to-end JAX, single GPU)
  ↓ 10-50x
Level 4: Multi-GPU JAX (pmap across devices)
  ↓ 2-8x per GPU
Total Potential: 100-1000x
```

---

## Optimization Levels (Ranked by Effort/Reward)

### 🟢 Level 1: Multi-Environment Training (Easiest)
**Effort**: < 1 hour
**Speedup**: 2-3x additional
**Status**: No code changes needed, just use more envs

**Implementation**:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Instead of 1 environment
env = make_env()

# Use 10-20 parallel environments
envs = SubprocVecEnv([make_env for _ in range(10)])

# Expected speedup: 2-3x
# Works with current SBX + Python physics
```

**Why it works**:
- CPU cores are underutilized with 1 env
- SubprocVecEnv parallelizes across cores
- Each env runs in separate process
- No GPU needed

**Benchmark**:
```python
# Current: 2,828 steps/sec with 1 env
# With 10 envs: ~7,000-10,000 steps/sec (estimated)
```

---

### 🟡 Level 2: vmap Physics Integration (Medium)
**Effort**: 1-2 days
**Speedup**: 2-5x additional (physics only)
**Status**: Requires custom SBX integration

**The Problem**:
- Current: SBX calls gym env in a loop (Python overhead)
- JAX vmap: Can run 100+ episodes in parallel
- Need to bridge: SBX → vmap batch → JAX physics

**Solution Architecture**:
```python
# Current (slow):
for env in envs:
    obs, reward = env.step(action)  # Python loop

# Optimized (fast):
obs, rewards = vmap_step_batch(states, actions)  # Single JAX call
```

**Implementation Plan**:

1. **Create Vectorized Environment Wrapper**:
```python
class VmapEnvWrapper:
    """Wraps multiple envs as a single vectorized JAX environment."""

    def __init__(self, n_envs, opponent_func, config):
        self.n_envs = n_envs
        # Initialize JAX states for all envs
        self.jax_states = self._init_batch(n_envs)

    def reset(self):
        """Reset all envs in parallel."""
        # vmap across all environments
        return vmap(self._reset_single)()

    def step(self, actions):
        """Step all envs in parallel with vmap."""
        # actions: [n_envs, action_dim]
        # returns: obs, rewards, dones, infos (all batched)
        return vmap(self._step_single)(self.jax_states, actions)
```

2. **Integrate with SBX**:
```python
# SBX expects VecEnv interface
# But internally batches all operations

from sbx import PPO

env = VmapEnvWrapper(n_envs=100, opponent_func=..., config=...)
model = PPO("MlpPolicy", env)

# SBX will call env.step(actions) where actions is [100, action_dim]
# VmapEnvWrapper uses vmap to run all 100 episodes in parallel on JAX
```

**Expected Speedup**:
- Physics: 2-5x (from vmap parallelization)
- Overall: ~4-6x total (2.6x SBX × 2x vmap)

---

### 🟠 Level 3: Full JAX Training Pipeline (Hard)
**Effort**: 3-5 days
**Speedup**: 10-50x with GPU
**Status**: Requires major refactoring

**Option A: PureJaxRL**

Pure JAX RL library optimized for JAX environments:

```python
# Install PureJaxRL
pip install purejaxrl

# Full JAX training loop
import jax
import jax.numpy as jnp
from purejaxrl import PPO

# Define Gymnax-compatible environment
class AtomCombatGymnax:
    def reset(self, rng):
        # Pure JAX reset
        return state, obs

    def step(self, rng, state, action):
        # Pure JAX step (no Python)
        return next_state, obs, reward, done, info

# Train with PureJaxRL
config = {
    "num_envs": 1024,  # Can go very high with JAX!
    "num_steps": 128,
    "total_timesteps": 10_000_000,
}

# Entire training loop is JIT-compiled
train_fn = jax.jit(make_train(config))

# Run on GPU
rng = jax.random.PRNGKey(0)
train_state = train_fn(rng)  # All on GPU
```

**Why it's faster**:
- No Python loops in training
- Entire rollout collection in JAX
- All data on GPU (no CPU↔GPU transfers)
- Can run 1000+ envs in parallel

**Expected Performance (GPU)**:
- 10,000 - 100,000 steps/sec
- 100-1000x faster than baseline

**Option B: Custom JAX Training Loop**

Build your own end-to-end JAX training:

```python
@jax.jit
def training_step(train_state, batch_states, batch_actions):
    """One training iteration - fully JIT compiled."""

    # Collect rollouts (vmap across envs)
    next_states, rewards, dones = vmap(env_step)(batch_states, batch_actions)

    # Compute advantages
    advantages = compute_gae(rewards, dones, train_state.value_fn)

    # Update policy (PPO)
    new_policy_params = update_policy(
        train_state.policy_params,
        batch_states,
        batch_actions,
        advantages
    )

    return train_state.replace(policy_params=new_policy_params)

# Run entire training in JAX
for step in range(total_steps):
    train_state = training_step(train_state, states, actions)
```

---

### 🔴 Level 4: GPU Acceleration (Hardest)
**Effort**: 2-5 days (may fail)
**Speedup**: 10-100x
**Status**: Experimental ROCm support

**The Challenge**: AMD GPU + JAX

**Current Situation**:
- You have: AMD GPU (gfx1032)
- You have: ROCm 5.x or 6.x
- Missing: JAX with ROCm support

**Installation Path**:

```bash
# Check ROCm version
rocm-smi --showproductname

# Uninstall CPU JAX
pip uninstall jax jaxlib -y

# Attempt 1: Prebuilt wheels (may not exist for your ROCm version)
pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Attempt 2: Build from source (complex)
git clone https://github.com/google/jax.git
cd jax
python build/build.py --enable_rocm --rocm_path=/opt/rocm
pip install dist/*.whl

# Verify
python -c "import jax; print(jax.devices())"
# Should show: [RocmDevice(id=0)]
```

**Potential Issues**:
1. **ROCm version mismatch**: JAX might not support your ROCm version
2. **Build complexity**: Building from source is non-trivial
3. **Driver issues**: ROCm drivers can be finicky
4. **Limited testing**: JAX ROCm support less mature than CUDA

**If Successful**:
```python
# Force GPU device
import jax
jax.config.update('jax_platform_name', 'rocm')

# Run benchmarks
devices = jax.devices()  # [RocmDevice(id=0)]

# Everything automatically uses GPU
result = vmap_step_batch(states, actions)  # Runs on GPU!
```

**Expected Speedup**:
- Single episode: 5-10x vs CPU
- Vectorized (batch=1000): 50-100x vs CPU
- Training: 20-50x vs CPU

---

### 🟣 Level 5: Multi-GPU Training (Expert)
**Effort**: 5-10 days
**Speedup**: 2-8x per GPU
**Status**: Only worth it if Level 4 works

**Using pmap (parallel map across devices)**:

```python
import jax
from jax import pmap

# Assume 2 GPUs
devices = jax.devices()  # [RocmDevice(0), RocmDevice(1)]

# Replicate training across GPUs
@pmap
def parallel_training_step(train_state, batch):
    """Each GPU trains on different data."""
    return training_step(train_state, batch)

# Split batch across GPUs
batch_per_gpu = batch.reshape(n_gpus, batch_size // n_gpus, ...)

# Train on all GPUs simultaneously
updated_states = parallel_training_step(train_states, batch_per_gpu)
```

**Expected Speedup**:
- Linear scaling up to 4-8 GPUs
- 2 GPUs: ~2x speedup
- 4 GPUs: ~3-4x speedup (diminishing returns)

---

## Recommended Path Forward

### Phase A: Quick Wins (1-2 hours)
**Do This First**:

1. **Use more environments** (Level 1):
   ```python
   # Change from 4 to 16 environments
   envs = SubprocVecEnv([make_env for _ in range(16)])

   # Expected: 2-3x additional speedup
   # Result: ~7,000-10,000 steps/sec
   ```

2. **Benchmark to confirm**:
   ```bash
   python benchmark_multi_env.py --envs 1 4 8 16 32
   ```

**Expected Total**: 2.6x (SBX) × 2.5x (multi-env) = **6.5x speedup**

---

### Phase B: vmap Integration (1-2 days)
**Medium Effort, High Reward**:

1. Create `VmapEnvWrapper` for SBX
2. Test with 100+ parallel episodes
3. Benchmark physics speedup

**Expected Total**: 2.6x (SBX) × 3x (vmap) = **8x speedup**

---

### Phase C: GPU Setup (2-5 days)
**High Effort, Uncertain Reward**:

Only attempt if:
- ✅ You need >10x speedup
- ✅ You're comfortable debugging ROCm issues
- ✅ You can afford 2-5 days of potential troubleshooting

**Steps**:
1. Try prebuilt ROCm wheels
2. If fails, build JAX from source
3. Debug driver/compatibility issues
4. Benchmark GPU vs CPU

**Expected Total**: 2.6x (SBX) × 20x (GPU) = **50x speedup**

---

### Phase D: Full JAX Pipeline (3-5 days)
**Expert Level**:

Only if GPU works and you need maximum performance:

1. Implement Gymnax environment
2. Integrate PureJaxRL
3. Run 1000+ parallel episodes on GPU
4. Benchmark end-to-end

**Expected Total**: **100-1000x speedup**

---

## Practical Next Steps

### Option 1: Easy Wins (Recommended)
**Time**: 1-2 hours
**Gain**: 6-8x total speedup

```python
# Just increase environments - no code changes!
n_envs = 16  # Instead of 4

# Train as usual
trainer = CurriculumTrainer(..., n_envs=16)
trainer.train()

# Enjoy 6-8x faster training
```

### Option 2: vmap Integration
**Time**: 1-2 days
**Gain**: 8-10x total speedup

```python
# Create vectorized environment wrapper
env = VmapEnvWrapper(n_envs=100, use_jax_jit=True)

# Train with SBX
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1_000_000)
```

### Option 3: Full JAX Adventure
**Time**: 1-2 weeks
**Gain**: 100-1000x total speedup
**Risk**: High (experimental)

```bash
# Setup GPU
./setup_jax_rocm.sh

# Use PureJaxRL
python train_purejaxrl.py --num_envs 1024 --device gpu
```

---

## Performance Estimates

| Level | Implementation | Time | CPU Speedup | GPU Speedup |
|-------|---------------|------|-------------|-------------|
| Current | SBX only | Done | 2.6x | N/A |
| **Level 1** | Multi-env | 1h | **6x** | N/A |
| Level 2 | vmap | 1-2d | 8x | 15x |
| Level 3 | Full JAX | 3-5d | 10x | 100x |
| Level 4 | GPU | 2-5d | N/A | 50x |
| Level 5 | Multi-GPU | 5-10d | N/A | 200x |

---

## Benchmarking Strategy

### Test 1: Multi-Environment Scaling
```bash
for n_envs in 1 2 4 8 16 32; do
    echo "Testing $n_envs environments..."
    python benchmark_curriculum.py --n_envs $n_envs --episodes 1000
done
```

### Test 2: vmap Batch Size Scaling
```bash
for batch_size in 10 50 100 250 500 1000; do
    echo "Testing batch size $batch_size..."
    python benchmark_vmap_training.py --batch_size $batch_size
done
```

### Test 3: GPU vs CPU
```bash
# CPU
python benchmark_jax_vmap.py --device cpu --batch_size 500

# GPU (if setup works)
python benchmark_jax_vmap.py --device gpu --batch_size 500
```

---

## Key Insights

### What Makes JAX Fast

1. **JIT Compilation**: Compiles to optimized XLA
2. **Vectorization (vmap)**: Parallel execution across batches
3. **GPU Acceleration**: Massive parallelism on GPU
4. **No Python Loops**: Everything in compiled code

### Where JAX Struggles

1. **Small workloads**: JIT overhead not amortized
2. **Single episodes**: No parallelization benefit
3. **Python interop**: Back to CPU for each step

### Sweet Spot for JAX

- **Large batches**: 100-1000 parallel episodes
- **Pure JAX**: No Python in hot loop
- **GPU**: Massive parallelism
- **Long training**: JIT cost amortized over millions of steps

---

## Recommended Action Plan

**Week 1**: Quick Wins
- Day 1: Increase to 16 environments → 6x speedup ✅
- Day 2: Benchmark and tune
- Day 3: Measure training time improvements

**Week 2** (Optional): vmap Integration
- Day 1-2: Build VmapEnvWrapper
- Day 3: Test with SBX
- Day 4: Benchmark and optimize
- Day 5: Document findings

**Week 3** (Optional): GPU Adventure
- Day 1-2: Attempt JAX ROCm installation
- Day 3: Debug issues (if any)
- Day 4: Benchmark GPU performance
- Day 5: Decide if worth keeping

**Decision Points**:
- ✅ Do Week 1 (easy win)
- 🤔 Do Week 2 if you need 8-10x
- ⚠️ Do Week 3 if you need 50-100x and can afford risk

---

**Bottom Line**: Start with Level 1 (multi-env) for immediate 6x speedup, then decide if you need more!
