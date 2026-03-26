# Redesign Analysis: What Would We Do Differently?

Written after 4 SAC attempts, 8 PPO runs, and extensive iteration on the training pipeline.

## What We Have

A 1D physics-based combat game where two fighters move on a line, switch between 3 stances (neutral/extended/defending), and try to reduce the opponent's HP to zero within 250 ticks. Fighters are trained via RL and fight both hand-crafted opponents and each other.

**Current results:**
- PPO (256-256): Graduates all 8 curriculum levels including Expert, 40-60% WR against expert fighters
- SAC: Entropy collapses in every run. Learns to beat easy movers (100%) but fails against stationary/defensive opponents (0%). Never graduated Phase 1 after 42K episodes.

## The Core Problem

SAC isn't failing because of curriculum structure, replay buffers, or hyperparameters. It's failing because **the action space doesn't suit off-policy learning.**

### Why the 4D logit action space is the root issue

Our action space is `[acceleration, logit_neutral, logit_extended, logit_defending]`. The stance is selected by argmax over the 3 logits. This was designed to solve PPO's deterministic inference collapse — and it works for PPO because:

1. PPO's Gaussian policy samples from logits stochastically during training
2. The argmax converts continuous noise into discrete stance choices
3. PPO's on-policy nature means it only needs the current policy's samples

For SAC, this design is pathological:

1. **SAC optimizes a Q-function over continuous actions.** The Q-function tries to assign value to specific (acceleration, logit0, logit1, logit2) tuples. But the actual stance is determined by argmax, so Q(0.5, 1.1, 1.0, 0.9) and Q(0.5, 2.0, 0.1, 0.1) produce the same stance (neutral) but the Q-function treats them as completely different actions with different values. The Q-function has to learn a discontinuous value landscape across logit space.

2. **The entropy bonus is computed over the continuous logit space, not the discrete stance space.** SAC thinks there's high entropy when logits are spread out, even if they always produce the same argmax stance. As the policy sharpens one logit, the entropy signal says "you're becoming deterministic" and the auto-tuner reduces the coefficient to compensate — causing entropy collapse.

3. **Replay buffer actions are continuous logits.** When SAC replays old experience, it evaluates new Q-values for the stored (state, logit) pairs. But the relevant information is the stance (argmax), not the specific logit values. Two stored transitions with different logits but the same stance are treated as different actions, fracturing the learning signal.

This is why SAC crushes approach_slow (only needs to move forward, stance doesn't matter much) but fails against stationary opponents (needs precise stance switching to deal damage without taking it back). The Q-function can't learn the value of stance switching because the action space makes it appear continuous when it's actually discrete.

## What Would We Do Differently

### 1. Action Space: Separate continuous and discrete

**Current:** `[acceleration, logit_neutral, logit_extended, logit_defending]` (4D continuous Box)

**Better:** Use a **hybrid action space** — continuous acceleration + discrete stance:
```
acceleration: Box(-1, 1)    # continuous
stance: Discrete(3)         # categorical: neutral, extended, defending
```

SB3 and SBX both support `MultiDiscrete` and `Dict` action spaces. For SAC specifically, use **SAC-Discrete** or **SAC with action masking** that handles the categorical stance natively.

**Why this fixes SAC:**
- Q-function evaluates 3 discrete stances, not a continuous logit space
- Entropy is computed over the actual stance distribution (3 categories)
- Replay buffer stores discrete stances, not logit vectors
- No more argmax discontinuity in the value landscape

**Impact on PPO:** PPO would also benefit — the current logit-argmax is a workaround for PPO's deterministic inference gap. With proper discrete action support, PPO wouldn't need the workaround.

**Implementation:** SBX doesn't have built-in SAC-Discrete. Options:
- Use `sbx.TQC` with a custom discrete head
- Use CleanRL's SAC-Discrete implementation
- Keep continuous acceleration + discretize stance in the env wrapper (env accepts `{accel: float, stance: int}`, model outputs `[accel, stance_logit0, stance_logit1, stance_logit2]` and env does the argmax)

The third option is closest to what we have — but the key change is making the Q-function aware that stance is categorical. This requires a custom network architecture with separate heads for continuous (acceleration) and discrete (stance) actions.

### 2. Observation Space: Add opponent direction explicitly

**Current 14D obs includes:**
- `distance` (unsigned, always positive)
- `relative_velocity` (signed, positive = approaching)

**Missing:** explicit `direction` (-1 or +1, which side is the opponent on)

The circle_left/circle_right asymmetry (100% vs 2.6% WR) proves the policy can't generalize across directions. Adding `direction` as obs[14] would let the policy learn "chase toward opponent" as a direction-agnostic concept.

**Also consider:**
- `your_velocity` is already in obs[1] and is signed — but the policy may not be connecting velocity sign with opponent direction
- Wall distances (obs[9], obs[10]) implicitly encode position but don't directly tell you which way to go

### 3. Reward Function: Much simpler

**Current:** 6 components with per-level scaling (proximity, inaction, stance, stamina, damage, terminal). We already identified that proximity and inaction mislead against expert opponents.

**Better:** Start with nearly raw reward:
```
reward = damage_dealt - damage_taken + win_bonus - loss_bonus
```

Add exactly ONE shaping term: a small approach reward that decays with training time (curriculum level or timestep count). This bootstraps initial approach behavior without creating reward hacking.

The complex reward function was built iteratively to solve specific training problems. But each component adds a local optimum the policy can exploit. The 6-component reward explains why SAC-trained fighters beat movers (high proximity + damage reward) but fail against stationary targets (proximity reward is zero when you're already close, inaction penalty for waiting).

### 4. Physics: Consider simplification

The physics engine is well-built and parity-verified. But some features create unnecessary difficulty for RL:

**Hit cooldown (5 ticks):** Creates a timing problem. The fighter must learn "I just hit, wait 5 ticks, then hit again." This is hard for both PPO and SAC because the cooldown state isn't directly observable.

**Consider adding to observation:**
- `ticks_since_last_hit` (0-5) — lets the policy learn timing
- `can_hit` (boolean) — even simpler, just tells if cooldown is expired

**Stance stamina drain:** Extended stance costs 0.08 stamina/tick. Defending costs 0. This creates a stamina management problem that's important for good play but hard to learn from reward alone.

**Consider:** Making the first few curriculum levels use relaxed physics (no stamina drain, no hit cooldown) to let the fighter learn basic approach/attack, then introduce these mechanics in later levels. This is curriculum learning applied to physics, not just opponents.

### 5. Algorithm: PPO is actually fine, SAC needs a different approach

**PPO with 256-256 graduated all 8 levels in Run 8.** It's working. The 70% Expert WR after 18K episodes is a real achievement.

**SAC's failure is not algorithmic — it's the action space mismatch.** If we fix the action space (option 1 above), SAC would likely work. Without fixing it, no amount of hyperparameter tuning will help.

**If starting from scratch today:**
1. Start with PPO + hybrid action space (discrete stance + continuous accel)
2. Use a simpler reward function (damage + win/loss + small approach bonus)
3. Add `direction` and `ticks_since_last_hit` to observations (16D)
4. If PPO plateaus at Expert level, add imitation learning warmstart from expert fighters
5. Only then try SAC-Discrete if PPO + imitation isn't enough

### 6. Training Structure: What the curriculum got right

The curriculum approach is correct. The key insights that worked:

1. **Per-opponent mastery prevents gaming aggregate WR** — this was a critical fix
2. **8 progressive levels with increasing complexity** — the right structure for PPO
3. **Population training after curriculum** — adds adaptive opponent diversity
4. **Anchor evaluation prevents forgetting** — the retention scoring works

What we'd keep: the entire curriculum + population pipeline. What we'd change: the action space, reward, and observation.

### 7. Population Training: What to improve

The population phase ran but fighters developed a monoculture. Improvements:

1. **Start with diverse initialization** — not all from the same curriculum graduate. Initialize half from curriculum, half random.
2. **Include curriculum opponents in evaluation** (already implemented via anchor scoring)
3. **Larger population** (16+) for more diversity
4. **Consider league-style training** where some fighters are trained as explicit counters

## Priority Order for Implementation

If I were starting the next iteration:

1. **Add opponent direction to observation space** (30 minutes, fixes circle_left/right asymmetry)
2. **Simplify reward to damage + terminal + small approach** (1 hour, removes reward hacking)
3. **Add ticks_since_last_hit to observation** (30 minutes, enables timing learning)
4. **Run PPO with these changes** (2 hours on Colab, compare to Run 8)
5. **If SAC needed: implement hybrid action space** (2-3 days, requires custom network)
6. **If Expert WR still low: add imitation learning warmstart** (1 week)

Items 1-4 are low-risk, high-impact changes that improve PPO too. Item 5 is the real SAC fix but requires more work. Item 6 is the nuclear option.

## The Honest Assessment

PPO with 256-256 network is a legitimate approach that produces fighters capable of beating hand-crafted experts 40-60% of the time. That's a real accomplishment.

SAC's failure is instructive — it's not that SAC is a bad algorithm, it's that our action space creates a fundamentally hostile learning landscape for off-policy Q-learning. The 4D continuous logit space with argmax stance selection was a clever PPO hack that accidentally made SAC impossible.

The biggest single improvement would be **fixing the action space to properly separate continuous and discrete actions.** Everything else is incremental.
