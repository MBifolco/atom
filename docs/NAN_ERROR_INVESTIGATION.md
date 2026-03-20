# NaN Error Investigation and Solution

## Problem Description

During curriculum training at Level 5 (expert fighters), PPO training would fail with NaN errors in specific environments (notably environments 10 and 18 out of 64). The error occurred during the creation of Normal distributions in the policy network, not immediately but after many successful training steps.

### Error Symptoms
```
ValueError: Expected parameter scale (Tensor of shape (64, 2)) of distribution Normal to satisfy the constraint GreaterThan(lower_bound=0.0), but found invalid values:
tensor([[7.5994e-01, 9.9993e-01],
        ...
        [nan, nan],  # Environment 10
        ...
        [nan, nan],  # Environment 18
        ...])
```

## Investigation Process

### 1. Initial Hypotheses (Incorrect)

**Dimension Mismatch Theory**
- Thought: Model trained on 13 dimensions, environment providing 9
- Why wrong: Would cause immediate errors, not gradual NaN appearance
- User insight: "wouldn't that have thrown many more errors immediately?"

**Quick Fix Attempts (Bad Solutions)**
- Hard clipping rewards to [-10, 10]
- Reverting to 9-dimensional observations
- Why wrong: Destroys the carefully designed reward signal

### 2. Root Cause Analysis

Created multiple diagnostic scripts to identify the actual cause:

#### Key Findings

**It's the PATTERN, not the SIZE of rewards:**

| Test Case | Reward Range | Result |
|-----------|--------------|---------|
| Constant 5000 reward | 5000 | ✅ No NaN |
| Oscillating ±10 | -10 to 10 | ✅ No NaN |
| Rare spikes ±100 | -100 to 100 | ✅ No NaN |
| Expert fighter patterns | -200 to 200 | ❌ NaN after accumulation |

**The Compound Effect:**
1. **Reward Scale**: Terminal rewards range from -200 to +200
2. **Damage Multipliers**: Mid-episode damage rewards multiplied by 10x
3. **Close-range Bonuses**: Additional 2x multiplier for close combat
4. **Expert Fighters**: Create consistent high-damage patterns
5. **Accumulation**: Gradients compound over thousands of steps
6. **Specific Environments**: Some environments (10, 18) hit problematic state-action combinations

### 3. Why NaN Occurs

The NaN appears during Normal distribution creation in PPO's policy network:

```python
# In PPO forward pass
mean_actions = policy_network(observations)  # Shape: (64, 2)
std = torch.exp(log_std)                     # Can become 0 or inf
distribution = Normal(mean_actions, std)     # NaN if std invalid
```

**Gradient Explosion Chain:**
1. Large rewards (up to 300+ per step) → Large policy gradients
2. Large gradients → Extreme weight updates
3. Extreme weights → Extreme network outputs
4. Extreme outputs → log_std becomes very negative
5. exp(very negative) → Near zero std
6. Normal(mean, ~0) → Numerical instability → NaN

## Solution: VecNormalize Wrapper

### What VecNormalize Does

VecNormalize uses **running statistics** (Welford's algorithm) to normalize rewards without knowing their range:

```python
# How it works internally
class RunningMeanStd:
    def update(self, batch):
        batch_mean = np.mean(batch)
        batch_var = np.var(batch)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

# Normalization
normalized_reward = (reward - running_mean) / (sqrt(running_var) + 1e-8)
normalized_reward = clip(normalized_reward, -10, 10)  # Safety clip at 10 std devs
```

### Why VecNormalize Works

1. **No Maximum Needed**: Adapts to any reward scale dynamically
2. **Preserves Relationships**: 200 reward is still 2x of 100 reward after normalization
3. **Gradient Stability**: Normalized rewards typically in [-2, 2] range
4. **Outlier Protection**: Clips at ±10 standard deviations

### Example Evolution

| Episode | Raw Rewards | Running Mean | Running Std | Normalized Range |
|---------|------------|--------------|-------------|------------------|
| 1 | [-10, 10] | 0.0 | 7.5 | [-1.3, 1.3] |
| 10 | [-100, 100] | 5.2 | 45.3 | [-2.3, 2.1] |
| 100 | [-200, 200] | -3.1 | 87.6 | [-2.2, 2.3] |
| 1000 | [-200, 200] | 2.4 | 91.2 | [-2.2, 2.2] |

## Implementation

### Files Modified

1. **`src/training/trainers/curriculum_trainer.py`**
```python
from stable_baselines3.common.vec_env import VecNormalize

# Wrap environment with VecNormalize
self.envs = VecNormalize(
    self.envs,
    norm_obs=False,      # Observations already normalized
    norm_reward=True,    # Normalize rewards
    clip_obs=10.0,      # Safety clip
    clip_reward=10.0,   # Clip at ±10 std deviations
    gamma=0.99          # Discount factor
)
```

2. **`src/training/vmap_env_wrapper.py`**
```python
# Safety clip only for extreme outliers (not hard clipping)
rewards = np.clip(rewards, -1000.0, 1000.0)
```

3. **`src/training/utils/stable_ppo_config.py`**
```python
def get_stable_ppo_config():
    return {
        "learning_rate": 3e-5,  # Reduced for stability
        "max_grad_norm": 0.5,   # Gradient clipping
        "target_kl": 0.01,      # Early stopping for PPO updates
        # ...
    }
```

## Why NOT to Use Hard Clipping

**Problem with `np.clip(reward, -10, 10)`:**

| Scenario | Original Reward | Hard Clipped | VecNormalize |
|----------|-----------------|--------------|--------------|
| Dominant win | 200 | 10 | ~2.2σ |
| Close win | 15 | 10 | ~0.2σ |
| Small damage | 2 | 2 | ~0.02σ |
| Big loss | -150 | -10 | ~-1.7σ |

Hard clipping destroys the signal - a dominant win becomes indistinguishable from a barely-win.

## Debugging Commands

```bash
# Test if rewards are causing issues
python archived/diagnostics/nan/diagnose_training_nan.py

# Check reward patterns vs size
python archived/diagnostics/nan/diagnose_reward_patterns.py

# Test VecNormalize effectiveness
python archived/diagnostics/nan/test_proper_nan_fix.py

# See how VecNormalize adapts
python archived/diagnostics/nan/explain_vecnormalize.py
```

## Key Takeaways

1. **NaN in RL is rarely about absolute values** - it's about patterns and accumulation
2. **Don't destroy your reward signal** - normalize, don't clip
3. **VecNormalize is the standard solution** - it's built for exactly this problem
4. **Running statistics are powerful** - no need to know min/max ahead of time
5. **The error location is misleading** - NaN in environments 10, 18 doesn't mean those environments are broken, they just hit the breaking point first

## Prevention Checklist

- [ ] Use VecNormalize for environments with large/unknown reward ranges
- [ ] Set appropriate learning rates (3e-5 or lower for large rewards)
- [ ] Enable gradient clipping (`max_grad_norm=0.5`)
- [ ] Consider target_kl for PPO to prevent large updates
- [ ] Monitor log_std values - if approaching -10, you're heading for trouble
- [ ] Test with extended training runs - NaN often appears after many steps

## References

- [Stable Baselines3 VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
- [Welford's Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Section on reward scaling
