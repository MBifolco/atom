# Post-Run7 Investigation: Training vs Runtime Performance Gap

## The Problem

Run 7 completed the full curriculum (L1-L7) and 20 generations of population training. The top fighter, Blaze_Fox_G15 (ELO 1569, 89% WR against population peers), was exported and tested against the expert fighters. Results were disappointing:

| Opponent | atom_fight.py (deterministic) | atom_fight.py (stochastic export) |
|----------|:---:|:---:|
| boxer | 0/5 | 0/10 |
| slugger | 0/5 | 0/10 |
| swarmer | 0/5 | 0/10 |
| counter_puncher | 5/5 | 0/10 |
| out_fighter | 0/5 | 10/10 |

This was surprising given that the curriculum graduate had ~24% WR against these fighters during training, and the population evolved specifically to beat each other.

## Investigation

### Step 1: Deterministic Inference Gap (known issue)

The first issue was that `atom_fight.py` and the exported fighter template both used `deterministic=True` for `model.predict()`. With the 4D logit action space, the Gaussian mean converges to a single stance — the fighter locks into one stance forever.

**Fix**: Switched exported fighters and test harness to `deterministic=False` (stochastic sampling). This is how the model was trained and evaluated.

### Step 2: Catastrophic Forgetting in Population Training

Tested both the curriculum graduate (pre-population) and Blaze_Fox (post-population) in stochastic mode:

| Opponent | Curriculum Graduate | Blaze Fox (Post-Pop) | Delta |
|----------|:---:|:---:|:---:|
| stationary_neutral | 20% | **100%** | +80 |
| approach_slow | 0% | **100%** | +100 |
| reactive_defender | 0% | **100%** | +100 |
| hp_adaptive | 80% | **100%** | +20 |
| counter_puncher | 20% | **60%** | +40 |
| out_fighter | 0% | **50%** | +50 |
| boxer | 20% | **40%** | +20 |
| **forward_mover** | **100%** | 10% | **-90** |
| **slugger** | **100%** | 30% | **-70** |
| **swarmer** | **100%** | 0% | **-100** |

Population training improved performance against harder opponents but caused forgetting of how to beat aggressive rushdown fighters (swarmer, slugger, forward_mover). This is expected — population fighters evolved to beat each other (similar PPO-style play), not curriculum opponents.

**Future fix**: Include curriculum opponents in the population evaluation mix to prevent forgetting.

### Step 3: Training Env vs Runtime Arena Gap

Even after the stochastic fix, we observed a large gap between the training env (`AtomCombatEnv`) and the runtime arena (`MatchOrchestrator` via `atom_fight.py`):

| Test (vs boxer, 50 seeds, stochastic) | Win Rate |
|---|---|
| Training env (AtomCombatEnv) | **20%** |
| Runtime arena (MatchOrchestrator) | **2%** |

Same model, same opponent, same physics engine, same seed range — 10x performance difference.

### Step 4: Physics Parity Verification

Investigated whether the two systems use different physics:

- **Physics engine**: Both use `Arena1DJAXJit` from the same file. Identical damage, stamina, collision, and velocity calculations.
- **World config**: Both use the same `WorldConfig` singleton. Same constants.
- **Starting positions**: Both start at position 2.0 vs 10.0 with mass 70.0.
- **Arena initialization**: Both use `Arena1DJAXJit(fighter_a, fighter_b, config, seed=seed)`.
- **Opponent state**: Both use `generate_snapshot()` to build the opponent's view.

**Conclusion**: 100% physics parity confirmed.

### Step 5: Observation Comparison

Captured observations tick-by-tick from both systems with the same seed:

```
TRAINING ENV obs (first 4 ticks):
  tick 0: [ 2.     0.     1.     1.     8.     0.     1.     1.    12.476  2.     10.476  0.     0.   ]
  tick 1: [ 2.03   0.358  1.     0.997  7.965 -0.416  1.     1.    12.476  2.03   10.446  0.     0.   ]
  tick 2: [ 2.088  0.687  1.     0.971  7.898 -0.8    1.     1.    12.476  2.088  10.388  0.     0.   ]
  tick 3: [ 2.147  0.704  1.     0.972  7.824 -0.872  1.     1.    12.476  2.147  10.329  0.     0.   ]

RUNTIME ARENA obs (first 4 ticks):
  tick 0: [ 2.     0.     1.     1.     8.     0.     1.     1.    12.476  2.     10.476  0.     0.   ]
  tick 1: [ 2.03   0.358  1.     0.997  7.965 -0.416  1.     1.    12.476  2.03   10.446  0.     0.   ]
  tick 2: [ 2.088  0.687  1.     0.971  7.898 -0.8    1.     1.    12.476  2.088  10.388  0.     0.   ]
  tick 3: [ 2.147  0.704  1.     0.972  7.824 -0.872  1.     1.    12.476  2.147  10.329  0.     0.   ]
```

**Observations are byte-identical** for the first 4 ticks. The physics and observation builder produce the same output.

### Step 6: Root Cause Found — `recent_damage` (obs[12])

The 13-dimensional observation space was:

```
[0]  you_position
[1]  you_velocity
[2]  you_hp (normalized)
[3]  you_stamina (normalized)
[4]  distance_to_opponent
[5]  relative_velocity
[6]  opponent_hp (normalized)
[7]  opponent_stamina (normalized)
[8]  arena_width
[9]  wall_distance_left
[10] wall_distance_right
[11] opponent_stance
[12] cumulative_episode_damage_dealt    <-- THE PROBLEM
```

**During training** (gym_env.py / vmap_env_wrapper.py): obs[12] was set to `self.episode_damage_dealt` — a running total that grows throughout the fight. After landing 50 damage, obs[12] = 50.0.

**During inference** (exported fighter via `build_observation_from_snapshot`): obs[12] was always **0.0** because the runtime snapshot protocol doesn't carry cumulative episode state — it only contains the current tick's instantaneous state.

The PPO policy network was trained on millions of observations where obs[12] correlated with winning. It likely learned behavioral patterns like:
- "obs[12] is low → I'm not landing hits → be more aggressive"
- "obs[12] is high → I'm winning → maintain pressure"
- "obs[12] = 0 for many ticks → something is wrong"

At inference, the model permanently saw obs[12] = 0.0, interpreting every tick as "I haven't dealt any damage" even after dealing significant damage. This corrupted signal caused systematically wrong action selection.

### Why the gap was ~10x, not ~2x

The training env's `AtomCombatEnv` passes obs[12] correctly (it has access to `episode_damage_dealt`). The runtime arena's `build_observation_from_snapshot` defaults obs[12] to 0.0. This explains why the same model performs dramatically worse in the runtime arena — it's seeing corrupted state.

The gap is amplified because obs[12] is **correlated with fight momentum**. Early in a fight, both systems see obs[12] ≈ 0.0. But once the fighter lands its first hit, the training env's obs[12] jumps up, reinforcing the behavior, while the runtime's stays at 0, breaking the feedback loop.

## The Fix

### Immediate: Drop obs[12] entirely (commit `c6fd4d4`)

The `recent_damage` dimension was **redundant** with obs[6] (opponent HP normalized). In a 1v1 fight:

```
cumulative_damage_dealt ≈ opponent_max_hp × (1 - opponent_hp_normalized)
```

There is no other source of damage to the opponent, so tracking cumulative damage dealt adds no information beyond what's derivable from the opponent's current HP.

More importantly, obs[12] was the **only observation dimension that required episode-level state tracking**. Every other dimension (position, velocity, HP, stamina, stance, distance, arena width) is instantaneous — reconstructible from a single tick's snapshot. This made obs[12] inherently incompatible with the snapshot-based inference path.

**Changes made:**
- Removed `recent_damage` parameter from `build_observation()`, `build_observation_batch()`, `build_observation_from_snapshot()`
- Changed observation space from 13D to 12D in `gym_env.py` and `vmap_env_wrapper.py`
- Removed `recent_damage` from all callers (population trainer, replay recorder, etc.)
- Updated all tests (shape checks, parity tests)

### Prerequisite: Stochastic inference (commit `a88267f`)

- Switched `export_fighters.py` template from `deterministic=True` to `deterministic=False`
- Changed `test_curriculum_graduate.py` default from deterministic to stochastic

### Impact

All existing trained models are incompatible with the new 12D observation space and must be retrained. The next training run will produce models where every observation dimension can be faithfully reconstructed from a single-tick snapshot, eliminating the train/inference gap.

## Verification Plan

After retraining with the 12D observation space:
1. Run the same 50-seed comparison test: training env vs runtime arena WR should converge
2. Exported fighters tested via `atom_fight.py` should match training-time performance
3. Population fighters should maintain curriculum-level skills (once curriculum opponents are mixed into population evaluation)
