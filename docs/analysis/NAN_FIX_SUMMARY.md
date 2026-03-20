# NaN Error Fix Summary

## Root Cause
The NaN errors during curriculum training (especially at Level 5) were caused by **gradient explosion** due to extremely large reward values:

1. **Unscaled rewards**: Terminal rewards could be 200+ for wins, -200 for losses
2. **Mid-episode damage rewards**: Multiplied by 10.0, plus 2x bonus for close-range hits
3. **No reward normalization**: Rewards could accumulate to 300+ per step
4. **Expert fighters at Level 5**: Deal more consistent damage, producing larger rewards more frequently

These large rewards caused gradients to explode during PPO backpropagation, leading to NaN in the policy network's Normal distribution creation.

## Solution Applied

### 1. Reward Clipping
- Added `np.clip(reward, -10.0, 10.0)` to both gym_env.py and VmapEnvWrapper
- Prevents any single reward from being too large

### 2. Reward Scaling
- Reduced terminal rewards: Win=10, Loss=-10 (was 200, -200)
- Reduced damage multiplier: 0.5x (was 10x)
- Reduced close-range bonus: 0.1x (was 2x)

### 3. Stable PPO Hyperparameters
- Reduced learning rate: 3e-5 (was 5e-5)
- Added target_kl=0.01 for early stopping of PPO updates
- Increased entropy coefficient to 0.01 for better exploration
- Kept gradient clipping at 0.5

## Test Results
✅ Rewards now properly bounded to [-10, 10]
✅ PPO can train without NaN errors
✅ Model parameters remain valid during training

## Why This Wasn't Immediate
The user correctly noted that if it was just a dimension mismatch, errors would appear immediately. Instead, this was a numerical stability issue that accumulated over time:
- NaN appeared only after many training steps
- Only affected specific environments (10, 18 out of 64)
- Occurred during backpropagation, not forward pass
- Expert fighters at Level 5 produced more extreme reward patterns

The fix ensures rewards stay in a reasonable range that PPO's gradient-based optimization can handle stably.