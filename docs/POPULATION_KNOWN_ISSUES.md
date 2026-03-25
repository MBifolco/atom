# Population Training — Known Issues

Structural issues identified from the March 2026 training runs. These are
methodology concerns, not bugs — the code works as designed, but the design
limits evolution effectiveness.

## 1. Single-Opponent Training Pairs

Each fighter trains against exactly one opponent per generation. The matchmaking
in `population_training_loop.py:65-86` assigns opponents from ELO-balanced pairs,
but each fighter only sees one opponent's style during its training episodes.

**Impact:** Fighters specialize against their training partner instead of learning
general combat. Children trained against each other (e.g., two G1 children paired
together) learn nothing useful and enter evaluation unprepared.

**Future fix:** Round-robin or multi-opponent training within each generation.

## 2. Uniform Mass (70kg) — Intentional

`population_trainer.py:773` defaults to `mass_range=(70.0, 70.0)`. This is a
**deliberate choice**, not an oversight. Early experiments showed that heavier
fighters always won when mass varied — the combat dynamics didn't produce
interesting mass tradeoffs, just a "heavier is better" gradient. Rather than
try to fix the physics balance while training was still broken, the decision
was to hold mass constant and revisit after training is in a good place and
more complexity is warranted.

**Prerequisite for revisiting:** The combat system needs rebalancing so that
lighter fighters have genuine advantages (speed, stamina efficiency, etc.)
that can compensate for the damage/HP advantage of heavier fighters. This is
a game design problem, not a training problem.

**When to revisit:** After the stance action space fix is validated on full
Colab runs and fighters demonstrate genuine strategic behavior (stance
switching, stamina management, spacing).

## 3. All-Time ELO Bias — Fixed

~~The ELO tracker maintained cumulative all-time ratings. Incumbent fighters
accumulated wins across all generations while new children started at 1500,
structurally locking the top 4 rankings.~~

**Fixed (March 2026):** ELO ratings are now reset to 1500 before each
generation's evaluation round-robin (`elo_tracker.reset_ratings()`). Rankings
are determined solely by current-generation match results. All-time W/L/D
stats are preserved for observability but not used for selection decisions.

## 4. Uniform Parent Selection — Fixed

~~Parent selection used `random.choice(survivors)` — uniform random from the
top 50%. Any survivor was equally likely to be a parent regardless of fitness.~~

**Fixed (March 2026):** Parent selection now uses rank-weighted probability
(`PopulationEvolver.select_parent`). Weights are `[N, N-1, ..., 1]` for N
survivors — the top-ranked fighter gets N times the selection weight of the
bottom survivor. Every survivor still has a nonzero chance (exploration), but
stronger fighters contribute proportionally more offspring (exploitation).

---

## Open Questions (Not Bugs)

### 5. Evaluation Seed Determinism

Evaluation matches use a fixed seed, so the same pair of fighters always
produces the same outcome. In the March 2026 runs, win rate variance was
exactly 0.327 every generation — identical rankings reproduced each round.

With the per-generation ELO reset (issue #3), rankings can now shift as
fighters improve between generations. But individual matchup outcomes are
still deterministic: a fighter that wins seed 42 but would lose seed 43 is
rated the same as one that wins both.

**Future improvement:** Vary evaluation seeds across generations (e.g.,
`seed = base_seed + generation`) or run multiple seeded matches per pair.
This would make rankings more robust to seed-specific flukes. Not urgent —
the ELO reset already broke the frozen-ranking problem.

### 6. Training Volume per Generation

Each fighter gets ~250 episodes of training per generation. A mutated child
(parent weights + 10% noise) may need significantly more than 250 episodes
to recover from the perturbation, let alone improve beyond the parent.

**Instrumented (March 2026):** Per-fighter `reward_trajectory` is now logged
in `generation_summary.jsonl` with quartile breakdowns (Q1-Q4). If Q4 reward
is significantly higher than Q3, the fighter was still improving when training
stopped — the episode budget is too small. Look for this pattern in the next
full Colab run to decide whether to increase `episodes_per_generation`.

**How to read the trajectory:**
- Q1 → Q4 **rising steadily**: still learning, needs more budget
- Q1 → Q4 **plateaued by Q3**: budget is adequate
- Q1 → Q4 **declining**: overfitting or unstable training
