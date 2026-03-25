# Algorithmic Options for Atom Combat Training

An analysis of what we could change algorithmically, from easiest to most ambitious, with honest assessment of impact vs effort.

## Current Baseline

PPO (Proximal Policy Optimization) with:
- 64-64 MLP policy (8K parameters)
- Gaussian policy with stochastic sampling
- On-policy learning (experience discarded after each update)
- Curriculum → Population pipeline
- ~8M timesteps budget on A100

Current results: Graduates through L1-L6, plateaus at ~24% WR on Expert fighters.

---

## Tier 1: Drop-in Changes (hours of work, immediate impact)

### 1A. Larger Network (256-256 MLP)

**What**: Increase the policy and value network hidden layers from 64-64 to 256-256 (~130K parameters vs ~8K).

**Why it helps**: The expert fighters implement ~7 distinct behavioral modes each (distance-dependent × stamina-dependent × HP-dependent). A 64-64 network literally doesn't have enough capacity to represent this complexity. It's like trying to fit a 7-branch decision tree into a network that can only learn 2-3 branches.

**Pros**: One-line config change in PPO hyperparams. No code changes. Same training infrastructure.

**Cons**: ~2-3x slower per training step (more parameters to update). Slightly less stable early training (more parameters to initialize). May need learning rate adjustment.

**Expected impact**: Strongest single change available. Likely gets Expert WR from 24% to 40-50%.

**Implementation**: Change `policy_kwargs=dict(net_arch=[256, 256])` in the PPO constructor.

### 1B. Learning Rate Schedule

**What**: Anneal the learning rate from 3e-4 down to 3e-5 over training, rather than using a fixed 3e-5.

**Why it helps**: Early training needs large steps to escape bad initialization. Late training needs small steps for fine-tuning. A fixed low LR (which we use to avoid NaN) slows early learning.

**Pros**: Simple config change. Well-understood technique.

**Cons**: Need to tune the schedule. Wrong schedule can hurt.

**Expected impact**: 10-20% faster curriculum progression. Doesn't change the ceiling, just how fast we reach it.

### 1C. GAE Lambda Tuning

**What**: Adjust the Generalized Advantage Estimation lambda (currently using SB3 default of 0.95).

**Why it helps**: Lambda controls the bias-variance tradeoff in advantage estimation. Lower lambda (0.9) reduces variance but increases bias — better for short fights where credit assignment is clearer. Higher lambda (0.98) is better for longer strategic planning.

**Pros**: One parameter change.

**Cons**: Requires experimentation to find the sweet spot.

**Expected impact**: Small but compounds with other changes. 5-10% improvement.

---

## Tier 2: Algorithm Swap (days of work, significant impact)

### 2A. SAC (Soft Actor-Critic)

**What**: Replace PPO with SAC — an off-policy, entropy-regularized actor-critic method.

**Why it helps**:
- **Off-policy**: SAC stores all experience in a replay buffer and retrains on it repeatedly. PPO trains on each batch once then discards it. SAC extracts 10-100x more learning per environment step.
- **Entropy bonus**: SAC maximizes reward + entropy, which naturally encourages exploration. PPO's exploration is just Gaussian noise. SAC's exploration is structured — it tries to maintain a diverse behavioral repertoire.
- **Handles continuous actions well**: SAC was designed for continuous action spaces. PPO was designed for discrete actions and adapted to continuous.

**Pros**: SB3 has a SAC implementation. Our 4D continuous action space is a natural fit. The entropy bonus should help discover diverse fighting strategies.

**Cons**:
- SAC needs a replay buffer (memory overhead, ~1-4GB for 1M transitions).
- SAC can be less stable in early training than PPO.
- Our vmap wrapper would need adaptation — SAC's data collection pattern differs from PPO's rollout-based collection.
- SAC doesn't parallelize as cleanly as PPO (off-policy updates are inherently sequential).
- We briefly tried SAC early in the project and had stability issues, though those may have been from the old 2D action space.

**Expected impact**: With 256-256 network, SAC would likely reach 50-60% Expert WR. The replay buffer means it gets much more learning from the same number of environment steps. The entropy bonus should naturally produce more diverse fighting styles.

**Implementation effort**: Medium. SB3 supports SAC, but our curriculum trainer is structured around PPO's rollout-based training loop. The checkpoint/recovery system would need updates. The vmap wrapper's batched collection would need to feed into a replay buffer instead of a rollout buffer.

### 2B. TD3 (Twin Delayed DDPG)

**What**: Another off-policy method, but with a deterministic policy.

**Why it helps**: Off-policy benefits same as SAC. Deterministic policy means no stochastic sampling issues.

**Pros**: Simpler than SAC (no entropy tuning). SB3 supports it.

**Cons**:
- Deterministic policy needs explicit exploration noise (same stance-collapse risk we already fixed).
- Our 4D logit space relies on stochastic sampling for stance selection — deterministic TD3 would need a different action space design.
- Generally inferior to SAC for continuous control.

**Expected impact**: Similar to SAC but likely slightly worse. Not recommended over SAC.

### 2C. Recurrent Policy (PPO-LSTM)

**What**: Replace the MLP policy with an LSTM-based policy that maintains hidden state across ticks.

**Why it helps**: Current MLP policy is purely reactive — it sees one observation and outputs one action with no memory. An LSTM policy can:
- Track opponent patterns over time ("opponent attacks every 5 ticks")
- Remember recent events ("I just got hit, should dodge")
- Build implicit world models ("opponent is low stamina, they'll switch to neutral soon")

**Pros**: SB3 supports recurrent policies via `sb3-contrib`. Minimal architecture change.

**Cons**:
- LSTM policies are harder to train (vanishing gradients, longer training times).
- Doesn't work with our vmap batched environments (LSTM state must be maintained per-env).
- Significantly slower per step (~5-10x).
- Adds complexity to checkpointing/resuming (LSTM hidden state must be saved/restored).

**Expected impact**: Moderate. Would help against pattern-based opponents (oscillator, stance_switcher) but the current 14D observation already captures most relevant state. The marginal benefit over a larger MLP is debatable.

**Recommendation**: Try 256-256 MLP first. If that plateaus, consider LSTM.

---

## Tier 3: Training Paradigm Changes (weeks of work, transformative impact)

### 3A. Imitation Learning (Behavioral Cloning + Fine-tuning)

**What**: Pre-train the policy network to mimic the hand-crafted expert fighters, THEN fine-tune with RL.

**How it works**:
1. Run each expert fighter for 10K episodes, recording (observation, action) pairs
2. Train the policy network via supervised learning to predict the expert's action given the observation
3. Use this pre-trained network as the starting point for PPO/SAC training

**Why it helps**: Instead of discovering fighting strategies from scratch via trial and error, the policy starts already knowing how to jab, retreat, manage stamina, etc. RL fine-tuning then adapts and improves these strategies.

**Pros**:
- Dramatically faster initial training (skip the "learn to approach" phase)
- The policy starts with a diverse behavioral repertoire (one expert uses each strategy)
- Well-understood technique (used in AlphaStar, OpenAI Five, etc.)

**Cons**:
- Need to convert expert fighter decisions to the PPO action format (acceleration → normalized value, stance → logits). This is the inverse of what we do in the export template.
- Behavioral cloning alone produces brittle policies that fail on out-of-distribution states. Must fine-tune with RL.
- If expert demonstrations are suboptimal (and they are — they're heuristic), the policy inherits their weaknesses.
- The expert fighters use the snapshot protocol, not the 14D observation space. Need a translation layer to generate training data.

**Expected impact**: High. Would likely get to 60%+ Expert WR because the policy starts knowing what good fighting looks like. The RL fine-tuning phase could then discover improvements the hand-crafted experts miss.

**Implementation effort**: Medium-high. Need a data collection pipeline, supervised training loop, and careful initialization of the SB3 PPO model from pre-trained weights.

### 3B. League Training (AlphaStar-style)

**What**: Instead of a simple population of 8 clones, maintain a league of agents with different roles:
- **Main agents**: Trained to beat everyone in the league
- **Exploiters**: Trained specifically to beat the current main agents (finds weaknesses)
- **Frozen snapshots**: Historical versions kept as training opponents (prevents forgetting)

**Why it helps**: Our current population training produces a monoculture — 8 fighters that all play the same way. League training forces diversity by explicitly training exploiters that find and punish dominant strategies. This produces robust policies that don't have blind spots.

**Pros**:
- Produces genuinely diverse fighting styles
- Prevents catastrophic forgetting (frozen snapshots maintain historical competence)
- The gold standard for competitive self-play (AlphaStar, OpenAI Five, Cicero)

**Cons**:
- Significant infrastructure complexity (managing multiple training streams, snapshot storage, matchmaking)
- Needs much more compute (3-10x current training time)
- Overkill for the current stage of the project

**Expected impact**: Transformative for population training quality. Would produce fighters that can handle any style, not just their siblings. But diminishing returns if the base policy (PPO 64-64) can't learn complex strategies to begin with.

**Implementation effort**: High. Would need to redesign the population training pipeline.

### 3C. Model-Based RL (Dreamer / MuZero-style)

**What**: Learn a model of the environment (predict next state from current state + action), then use the model to plan ahead or generate synthetic training data.

**Why it helps**:
- **Planning**: Before committing to an action, simulate 10-20 future ticks to see which action leads to the best outcome. This is how chess engines work — look ahead, evaluate positions, pick the best path.
- **Data efficiency**: Generate millions of synthetic experiences from the learned model without running the real physics engine. Train the policy on synthetic + real data.

**Pros**: Dramatically better long-horizon reasoning. Would enable genuine strategic play (e.g., "if I retreat now, opponent will chase, I'll turn and counter at the wall").

**Cons**:
- Learning an accurate world model is its own hard problem. If the model is wrong, the policy learns to exploit model errors rather than real physics.
- MuZero-style planning requires a tree search at inference time, which is too slow for real-time fights (unless we use a learned policy to amortize the search).
- Dreamer is complex to implement and not available in SB3.
- Our physics engine is already fast (JAX JIT) — the bottleneck is policy learning, not data collection. Model-based RL helps most when data collection is expensive.

**Expected impact**: High ceiling but high risk. If the world model is accurate, could produce genuinely strategic fighters. If not, could be worse than PPO.

**Implementation effort**: Very high. Would need a custom implementation.

---

## Tier 4: Research-Level (months of work, uncertain payoff)

### 4A. Hierarchical RL (Options Framework)

**What**: Split the policy into two levels:
- **High-level policy**: Selects a "strategy" every N ticks (e.g., "approach aggressively", "maintain distance", "retreat and recover")
- **Low-level policy**: Executes the selected strategy at the tick level (actual acceleration + stance decisions)

**Why it helps**: The expert fighters are essentially hand-coded hierarchical policies. Boxer has ~3 strategies (jab range, recovery, pursuit) selected by stamina/distance conditions. Teaching a flat PPO policy to discover this hierarchy is much harder than explicitly structuring it.

**Cons**: Designing the strategy space is a form of manual engineering. Hard to get the abstraction level right. Complex to train (two policies, potentially different timescales).

### 4B. Curriculum as Reward Shaping (Potential-Based)

**What**: Instead of switching opponents at level boundaries, use the curriculum to shape the reward function. Each level adds a new reward component that gradually "teaches" a new skill, with potential-based shaping that provably preserves the optimal policy.

**Why it helps**: Avoids the sudden difficulty jumps between levels. The policy continuously learns new skills without forgetting old ones.

**Cons**: Theoretically elegant but practically hard to design the potential functions.

---

## Recommendation: What to Do Next

In priority order, for the next training run:

| Change | Effort | Impact | Do Now? |
|--------|--------|--------|---------|
| 256-256 network | 1 line | High | **Yes** |
| 16M timesteps | Config | Medium | **Yes** |
| Sparse rewards for L6+ | Config | Medium | **Yes** |
| Curriculum opponents in population | ~50 lines | High | **Yes** |
| Learning rate schedule | Config | Low-Medium | Yes |
| SAC instead of PPO | ~1 week | High | After next run |
| Imitation pre-training | ~2 weeks | Very High | After SAC |
| League training | ~3 weeks | Transformative | Future |

The first 4 changes are all achievable before the next training run and compound with each other. Together they should get Expert WR from 24% to 50%+. SAC and imitation learning are the next big jumps but require more investment.

The honest truth: with PPO + 256-256 + more training time + sparse rewards, we'll produce fighters that are **competent but not expert**. They'll beat test dummies and simplified experts consistently, hold their own against full experts sometimes, but not consistently outplay the hand-crafted strategies. Getting past that ceiling requires either SAC (better sample efficiency) or imitation learning (starting from a higher baseline).
