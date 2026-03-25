# Reward System Review

Deep analysis of the training reward system from March 2026, based on
signal_engine.py code review and run4 empirical data.

## Current Architecture

6 independent reward components computed per step, combined into a single
scalar, then normalized by VecNormalize (running mean/variance, clipped to
[-10, +10]).

### Component Summary

Empirical averages are **raw logged episode-sum component means** (pre-
normalization). PPO sees normalized/clipped values, not these magnitudes.

| Component | Per-Step Magnitude | Empirical Avg | % of Signal | Type |
|-----------|-------------------|---------------|-------------|------|
| Damage | ±10.0 per HP point | 272–728 | 60–85% | Per-tick |
| Terminal | 100–181 (win) / -100–-200 (loss) | 53–133 | 10–20% | End-of-episode |
| Inaction | -0.02 to -0.1 | -17 to -18 | ~3% | Per-tick |
| Stance | +0.05 to +0.10 | +4.0 to +4.7 | <1% | Per-tick |
| Proximity | +0.1 to +0.2 | +0.7 to +1.0 | <0.2% | Per-tick |
| Stamina | -0.05 to +0.02 | -5.5 to -6.3 | ~1% | Per-tick |

**Key finding:** Damage reward dominates (60–85% of raw signal). Proximity,
stamina, and stance rewards are 100–1000x smaller — likely too weak to
reliably shape behavior, even though VecNormalize operates on the combined
reward (not per-component). The secondary signals may be underweighted
relative to damage rather than literally invisible.

### Non-Reward Guardrails Already In Place

Reward tuning is no longer the only defense against bad graduates. The
current trainer has:
- Combat-quality graduation gate (`min_mean_damage_dealt`, `min_nonzero_damage_rate`)
- Deterministic sanity check (must deal damage under `deterministic=True`)
- Per-level stance observability (`extended_stance_rate` in level summaries)

Reward improvements now target **faster learning**, **better per-level
shaping**, and **cleaner generalization** — not basic "can the fighter
fight at all?"

---

## Proposed Improvements

### Small (Low Risk, Quick Wins)

**1. Fix first-step proximity = 0 bug.**
`last_distance` is initialized to `None`, so proximity reward is always 0
on the first step after reset. Should initialize to current distance.
One-line fix in `signal_engine.py`.

**2. Reduce timeout tie penalty from -200 to -50.**
The -200 penalty was a bootstrapping hack from early training when fighters
refused to engage and just stood still tying. It worked for that purpose,
but now that the curriculum produces fighters with 41% extended stance rate
by Level 1, it's no longer needed. With per-level reward scaling (#8), L1's
boosted inaction/proximity rewards handle the engagement forcing instead.
Reduce to -50 (same as slight loss) so ties aren't punished worse than
actual losses at higher levels.

**3. Increase time bonus divisor from 40 to 15.**
Currently winning at tick 1 gives only 6.25 bonus — negligible next to the
100 base win. Dividing by 15 gives max ~16.7, still modest but enough to
create a real incentive for efficient KOs.

### Medium (Moderate Risk, Meaningful Impact)

**4. Scale up proximity rewards 10x.**
Proximity is currently 0.2/0.1/0.1 per step — likely too weak to shape
behavior reliably next to damage at 10.0 per point. Scaling to 2.0/1.0/1.0
makes proximity a real secondary signal. The pursuit/recovery/engagement
logic is well-designed but currently drowned out.

**5. Scale up stance rewards 5x.**
Extended bonus 0.05 → 0.25, defending bonus 0.10 → 0.50. Currently stance
rewards total ~4.5 per episode vs 500+ for damage. The fighter already uses
all 3 stances (from the logit fix), but the reward doesn't reinforce
*situationally appropriate* stance choices.

**6. Add per-tick reward for extended stance in range.**
No direct reward for *being in extended stance while close enough to hit*.
Damage reward fires only when damage actually happens (distance + stance +
cooldown aligned). A small per-tick bonus for "extended stance within reach"
(distance < 0.82m) bridges the gap between intent and execution.

**Caution:** Must be gated carefully to avoid rewarding **camping in extended
stance** near the opponent. Consider requiring recent closing movement or a
short time horizon (e.g., only reward if distance decreased in the last
2-3 ticks). Otherwise it creates a new exploit: stand next to opponent in
extended and collect free reward without timing attacks.

**7. Add defending-while-taking-hits bonus.**
Blocking costs 1.0 stamina but gives no reward signal. A small bonus when
taking damage in defending stance (e.g., +0.5 per hit blocked) teaches that
blocking is the right response to incoming attacks. Currently the only
defending incentive is a -0.05 stamina penalty avoidance — too weak.

**Implementation note:** Requires new runtime instrumentation. The reward
engine (`signal_engine.py`) only receives damage/stamina/stance inputs.
Blocked-hit detection happens in `arena_1d_jax_jit.py:598` but is not
currently surfaced to the reward function. Would need a new field like
`blocked_hit_count` or `blocked_damage` in the env info dict.

### Large (Higher Risk, Architectural)

**8. Per-level reward scaling (replaces uniform #4/#5 scaling).**
Instead of scaling all rewards uniformly, apply per-level multipliers that
focus each level's reward signal on the skill cluster it's supposed to
teach. Base formulas stay untouched — just multiply the secondary components.
Damage and terminal always stay at 1.0x (core learning signal).

```python
LEVEL_REWARD_WEIGHTS = {
    # Level 1-2: loud proximity + inaction → force engagement
    "fundamentals":  {"proximity": 20.0, "inaction": 3.0, "stance": 5.0, "stamina": 1.0},
    "basic_skills":  {"proximity": 15.0, "inaction": 2.0, "stance": 5.0, "stamina": 2.0},
    # Level 3-4: boost stamina/stance → teach resource management
    "intermediate":  {"proximity": 10.0, "inaction": 1.5, "stance": 8.0, "stamina": 5.0},
    "advanced":      {"proximity": 5.0,  "inaction": 1.0, "stance": 5.0, "stamina": 3.0},
    # Level 5+: back off shaping → let damage/terminal dominate
    "adaptive":      {"proximity": 3.0,  "inaction": 1.0, "stance": 3.0, "stamina": 2.0},
    "expert":        {"proximity": 1.0,  "inaction": 1.0, "stance": 1.0, "stamina": 1.0},
    "gauntlet":      {"proximity": 1.0,  "inaction": 1.0, "stance": 1.0, "stamina": 1.0},
}
```

This approach:
- Replaces the -200 tie penalty: L1's 3x inaction + 20x proximity handles
  engagement forcing, so the tie penalty drops to -50 safely.
- Replaces uniform #4/#5 scaling: proximity and stance get boosted where
  they matter (early levels) and fade where they'd over-constrain (late).
- Keeps the reward engine itself simple: one multiplier lookup per component.
- Is observable: level_summaries already log per-component means, so we can
  see if the scaling is having the intended effect per level.

**Implementation:** Add a `difficulty` or `level_name` parameter to
`compute_step_reward_scalar` / `compute_step_rewards_batch`, look up
the weight dict, multiply each component before summing. Pass the current
level's difficulty from the curriculum trainer through the env.

**Scope note:** The reward weights table applies only to curriculum training.
Population training uses the same reward engine but should always use 1.0x
weights (no curriculum-level shaping). The weight lookup should default to
all-1.0 when no level name is provided, ensuring population training is
unaffected.

**9. Curriculum-aware reward normalization.**
VecNormalize running mean/variance persists across level transitions. When
graduating from L1 (easy, high rewards) to L2 (harder, lower rewards), the
normalization is calibrated to L1's distribution. L2 rewards look
artificially deflated until stats catch up.

Options (increasing scope):
- **Reward-stat reset only:** Reset running reward mean/variance on level
  transition, keep observation normalization intact. Lowest risk.
- **Full obs+reward reset:** Reset both. Higher risk — observation stats
  may need to persist for stability.
- **Per-level normalizer snapshots:** Save/restore normalizer state per
  level. Most robust but most complex.

Recommend starting with reward-stat reset only.

**10. Replace inaction penalty with engagement reward.**
Inaction penalty is negative-only (-0.02 to -0.1). It punishes not doing
anything but doesn't teach *what to do*. An engagement reward (positive for
being within striking range, higher for active combat) would be more
informative. Requires rethinking inaction/proximity interaction.

**11. Opponent-aware damage scaling.**
All damage rewarded equally regardless of opponent difficulty. In population
training with mixed-opponent batches, dealing 10 damage to a weak assigned
opponent has the same reward as 10 damage to a strong one. Consider scaling
by opponent strength (level difficulty or evaluation win rate) to prevent
overvaluation of damage against weaker assigned opponents.

---

## JAX/Python Opponent Parity — DONE

All items completed March 2026:
- 14 Python fighters rewritten to match JAX canonical behavior
- 6 pre-existing JAX functions rewritten for direction-based logic
- Strict ValueError on unknown JAX opponents (no silent fallback)
- 27 parity tests × 5 states each, 0.5 accel tolerance
- Resolved-opponent logging in `create_multi_opponent_func()`

---

## Recommended Priority

**Phase 0 — JAX parity: DONE**

**Phase A — reward fixes (next):**
- Small fixes #1 (first-step proximity), #2 (tie penalty → -50), #3 (time bonus)
- #8 (per-level reward scaling) — this subsumes #4 and #5 by applying
  level-appropriate multipliers instead of uniform scaling. It also properly
  replaces the -200 tie penalty with level-specific engagement forcing.

**Phase B — after validating Phase A on a training run:**
- #6 (extended-in-range bonus) and #7 (blocking bonus) if stance behavior
  still isn't situationally appropriate after scaling
- #9 (VecNormalize reward-stat reset on level transition) if level summaries
  show reward distribution artifacts at level boundaries
- #10 and #11 are longer-term architectural considerations

---

## Detailed Formulas

### Damage (per-tick)
```
damage_component = (damage_dealt - damage_taken) * 10.0
if damage_dealt > 0 and distance < arena_width * 0.3:
    damage_component += damage_dealt * 2.0
```

### Proximity (per-tick, requires last_distance != None)
```
pursue = opp_hp < 30% or opp_stamina < 20%
recover = not pursue and stamina < 20%
engage = not pursue and not recover and distance < 25% arena_width

if pursue and distance_decreased > 0.1: +0.2
if recover and distance_increased > 0.1: +0.1
if engage: +0.1 * (1 - distance / threshold)
```

### Stamina (per-tick)
```
if stamina_pct > opp_stamina_pct + 0.2: +0.02
if stamina_pct < 0.2 and stance != defending: -0.05
```

### Stance (per-tick)
```
if extended and opp_hp < 50%: +0.05
if defending and stamina < 30%: +0.10
```

### Inaction (per-tick, when no damage dealt or taken)
```
if distance < 20% arena: -0.1
elif distance < 40% arena: -0.05
else: -0.02
```

### Terminal (end-of-episode)
```
Win:  100 + time_bonus(0-6.25) + hp_bonus(±50) + efficiency(0-25)
Loss: -100 - (opp_hp_pct - fighter_hp_pct) * 50
Tie:  -25

Timeout clear win (>10% HP lead):  100 + hp_diff * 50
Timeout slight win (0-10%):        0
Timeout slight loss (-10%-0%):     -50
Timeout clear loss (<-10%):        -100 + hp_diff * 50
Timeout exact tie:                 -200
```
