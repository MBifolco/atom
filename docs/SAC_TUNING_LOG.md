# SAC Tuning Log

## Run SAC-1: Baseline (commit d6fbca9)

**Config:**
- learning_rate: 3e-4
- buffer_size: 100,000
- learning_starts: 1,000
- batch_size: 256
- tau: 0.005
- ent_coef: "auto" (auto-tuned, no floor)
- train_freq: 1
- gradient_steps: 1
- net_arch: [256, 256]

**Results:**
- L1 Fundamentals: Graduated at 5,079 episodes (51.7% WR)
- L2 Basic Skills: Graduated at ~10,000 episodes
- L3 Intermediate: STUCK at 49,500 episodes, 22% WR
  - 4/5 opponents mastered
  - stamina_efficient: 1.6% WR (9,419 episodes, 151 wins)

**Diagnosis:**
- ent_coef collapsed from 0.96 → 0.003 (near-zero exploration)
- Policy became near-deterministic, couldn't adapt to stamina_efficient
- 54K gradient updates but no improvement on L3 after first few thousand
- Negative mean rewards (-200 to -600) on recent episodes

**Root cause:** SAC's auto entropy tuning has no floor — it drove entropy
to near-zero once the policy found a decent strategy for easy opponents.
The policy then couldn't explore different tactics for harder opponents.

---

## Run SAC-2: Entropy floor + faster training (next run)

**Changes from SAC-1:**
1. `ent_coef`: "auto_0.1" → auto-tune but with floor of 0.1 (prevents collapse)
2. `train_freq`: 1 → 4 (train every 4 steps, 4x less overhead)
3. `gradient_steps`: 1 → 2 (2 gradient updates per training call)
4. `buffer_size`: 100,000 → 300,000 (more replay capacity)
5. `learning_starts`: 1,000 → 2,000 (more warmup for 250-env batches)

**Rationale:**
- Entropy floor ensures ongoing exploration even after policy stabilizes
- train_freq=4 reduces Python overhead by 4x (SAC was 7x slower than PPO)
- gradient_steps=2 compensates for less frequent training
- Larger buffer stores more experience from all 250 envs
- Longer warmup avoids early Q-function overestimation

**Expected effect:**
- Faster wall-clock training (~2x speedup from train_freq=4)
- Maintained exploration through curriculum (ent_coef stays ≥0.1)
- Better sample reuse from larger buffer

---

## Run SAC-3: Phased Mixed Curriculum + auto_0.1 (commit dd03e96)

**Config changes from SAC-2:**
- Phased mixed curriculum: 3 phases accumulating opponents instead of 8 sequential levels
- Phase 1: 12 opponents (L1+L2), Phase 2: 24, Phase 3: 38
- Replay buffer preserved across phases
- Fixed reward weights across all phases
- ent_coef: "auto_0.1" (initial value 0.1, auto-tuned from there)

**Results:**
- Phase 1: NOT graduated after 42,396 episodes, 7.4M timesteps, ~75 minutes
- Overall WR: 37% and DECLINING (was 43% at ep 4000)
- 6/12 opponents mastered, 6 stuck at 0% WR
- ent_coef collapsed: 0.10 → 0.0017 (same as SAC-1)

**Diagnosis:**
- `"auto_0.1"` in SBX only sets the INITIAL value — there is no floor
- Auto-tuning drove entropy to 0.0017 despite starting at 0.1
- Policy locked into beating easy movers (100% WR) but 0% vs stationary/defensive
- Mixed curriculum didn't help because the core issue is entropy collapse
- WR declining over time = policy getting worse, not better

**Mastered:** approach_slow (100%), shuttle_medium (100%), circle_left (100%),
approach_extended (92%), forward_mover (15%), stationary_neutral (5%)

**Stuck at 0%:** stationary_extended, stationary_defending, flee_always,
circle_right, flee_defending, backward_mover

---

## Run SAC-4: Fixed entropy coefficient (next run)

**Changes from SAC-3:**
1. `ent_coef`: "auto_0.1" → 0.1 (FIXED, no auto-tuning)

**Rationale:**
- Three runs of auto-tuned entropy all collapsed to <0.01
- SBX auto-tuning has no floor mechanism
- Fixed ent_coef=0.1 guarantees ongoing exploration
- This is the standard fix for entropy collapse in SAC literature

**Expected effect:**
- ent_coef stays at 0.1 throughout training (no collapse)
- Policy maintains exploration diversity against all opponents
- Should break through the 0% WR barrier on stationary/defensive opponents
