# ML Review: Observation Space, Training Approach, and Realistic Expectations

## Current Observation Space (14D)

```
[0]  you_position           — absolute position on arena (0 to 12.476)
[1]  you_velocity            — current velocity (-2.67 to 2.67)
[2]  you_hp                  — normalized 0-1
[3]  you_stamina             — normalized 0-1
[4]  distance_to_opponent    — absolute distance (0 to 12.476)
[5]  relative_velocity       — signed, positive = approaching
[6]  opponent_hp             — normalized 0-1
[7]  opponent_stamina        — normalized 0-1
[8]  arena_width             — constant 12.476 (redundant but harmless)
[9]  wall_distance_left      — derivable from position + arena_width
[10] wall_distance_right     — derivable from position + arena_width
[11] opponent_stance          — 0=neutral, 1=extended, 2=defending
[12] your_stance              — 0=neutral, 1=extended, 2=defending (NEW)
[13] tick_fraction            — 0.0=start, 1.0=end (NEW)
```

### Assessment: Is this sufficient?

**Yes, for the current game complexity.** The observation contains all information needed to play optimally:
- Spatial awareness: position, distance, walls, velocity
- Resource state: HP and stamina for both fighters
- Tactical state: both stances, time remaining
- All information is instantaneous (no episode-level accumulation)

**Minor redundancies** (obs[8], obs[9], obs[10] are derivable from obs[0]) are fine — they make the network's job easier. Neural networks are universal function approximators but learn faster when useful features are pre-computed.

**What we intentionally DON'T include:**
- Opponent absolute velocity/position — relative velocity + distance is more useful for combat decisions and more generalizable
- Hit cooldown timer — 5-tick cooldown is short enough that the policy can learn timing implicitly
- Historical state (last N observations) — would help with prediction but dramatically increases complexity; not needed at this stage

## Training Approach Analysis

### What we're doing right

1. **Curriculum learning** — Progressive difficulty is the right approach for RL in complex environments. Training directly against experts would give zero learning signal (0% WR = no gradient).

2. **Per-opponent mastery** — Prevents the policy from "gaming" aggregate WR by farming easy opponents while ignoring hard ones. This was a real problem we fixed.

3. **4D logit action space** — The stance-as-logits design avoids the deterministic inference collapse that killed early training runs. Clean solution to a hybrid discrete/continuous action space.

4. **Shared physics engine** — Training and runtime now use identical physics, observations, and action processing. The recent_damage gap we found and fixed was the last known parity issue.

5. **Population-based training** — Self-play after curriculum adds diversity and teaches the policy to handle adaptive opponents, which test dummies can't provide.

### What's limiting us

#### 1. PPO is the wrong algorithm for this task (but the pragmatic choice)

PPO (Proximal Policy Optimization) is an on-policy algorithm designed for stability and sample efficiency in continuous control. It's a good general-purpose choice, but for competitive fighting games, **off-policy methods** like SAC (Soft Actor-Critic) or **model-based methods** would likely perform better:

- **PPO is on-policy** — it discards all collected experience after each update. In a 512K-step rollout, PPO trains on those transitions once, then throws them away. Off-policy methods (SAC, TD3) reuse experience via replay buffers, getting 10-100x more learning per environment step.
- **PPO's exploration is undirected** — it relies on Gaussian noise in the policy, which produces random-looking behavior. For fighting, you need structured exploration (try different attack sequences, test spacing strategies). Curiosity-driven or count-based exploration would help.
- **PPO struggles with long-horizon credit assignment** — in a 250-tick fight, the reward for a good positioning decision at tick 50 is diluted by 200 subsequent ticks. Methods with better temporal credit assignment (e.g., transformer-based policies with attention over trajectory) would learn strategic play faster.

**However**, PPO is the right pragmatic choice because:
- SB3 has a battle-tested PPO implementation
- It's stable (no divergence issues beyond the NaN we already handle)
- It parallelizes trivially with our vmap setup
- SAC support in SB3 exists but is less mature for our use case

#### 2. Network architecture is undersized for the task

SB3's default MlpPolicy uses two hidden layers of 64 units each (64-64). This is ~8K parameters. For a 14D input and 4D output, this is adequate for simple reactive policies but insufficient for strategic reasoning.

The expert fighters implement multi-phase strategies (boxer has 3 stamina tiers × 2 distance conditions × pursuit mode = ~7 behavioral modes). Learning to replicate this requires either:
- **Larger networks** — 256-256 or even 512-256 would give the policy more capacity to represent complex state-dependent strategies
- **Recurrent policies** (LSTM) — would let the policy maintain internal state across ticks, enabling memory of recent events ("opponent just whiffed an attack, counter now")
- **Attention mechanisms** — overkill for 14D input, but would help if we expand to sequence observations

**Recommendation**: Switch to 256-256 MLP. This is a one-line change in the PPO configuration and would significantly increase the policy's capacity without meaningful training speed impact.

#### 3. Reward shaping may be over-engineered

The current reward function has 6 components with per-level weight scaling:
- damage_dealt, proximity, stamina, stance, inaction, terminal

Over-engineered reward shaping can cause **reward hacking** — the policy optimizes the reward signal rather than the true objective (winning). For example:
- Proximity reward encourages approaching, but the best strategy against a counter-puncher is to NOT approach
- Stance reward encourages stance switching, but sometimes staying in one stance is optimal
- Inaction penalty punishes standing still, but patience is a valid strategy

**A simpler reward would likely work better:**
```
reward = damage_dealt - damage_taken + win_bonus - loss_penalty
```

This sparse reward is harder to learn from initially (which is why the shaped reward exists for L1-L2) but produces policies that actually optimize for winning rather than for maximizing auxiliary signals. The curriculum can use shaped rewards for early levels and transition to sparse rewards for later levels.

#### 4. Training budget is insufficient for the Expert level

The numbers tell the story:

| Level | Episodes to graduate | Opponents | Difficulty |
|-------|---------------------|-----------|------------|
| L1 Fundamentals | 2,791 | 6 (stationary + simple) | Trivial |
| L2 Basic Skills | 373 | 6 (moving + stances) | Easy |
| L3 Intermediate | 928 | 5 (spacing + stamina) | Medium |
| L4 Advanced | 500 | 7 (complex patterns) | Medium |
| L5 Adaptive | 400 | 4 (reactive) | Medium |
| L6 Pre-Expert | 11,606 | 5 (simplified expert) | Hard |
| L7 Expert | 12,800+ (didn't graduate) | 5 (full expert) | Very Hard |

L6 took more episodes than L1-L5 **combined**. L7 didn't graduate at all. This isn't a bug — it reflects a genuine difficulty cliff. The expert fighters implement strategies that require fundamentally different capabilities than the curriculum teaches:

- **Patience** (counter_puncher) — the curriculum rewards approaching and attacking; the counter_puncher rewards waiting
- **Retreating under advantage** (out_fighter) — the curriculum never teaches when to NOT fight
- **Momentum building** (slugger) — requires understanding velocity × mass × distance interactions
- **Stamina cycling** (boxer) — requires multi-step planning across 3 stamina thresholds

PPO with a 64-64 network needs **millions of episodes** to learn these, not thousands. The 8M timestep budget gives ~30K episodes at L7 — probably 10x too few.

#### 5. Population training fights the wrong opponents

The population phase pits 8 PPO fighters against each other. Since they all descended from the same curriculum graduate, they all play similarly. This creates a **strategy monoculture** — the population evolves to beat one style of play (their own) rather than developing diverse strategies.

The style fingerprinting and diversity matchmaking we implemented helps, but fundamentally 8 identical-origin fighters can't generate enough strategic diversity. The population should also train against:
- Curriculum opponents (to prevent forgetting)
- The hand-crafted expert fighters (to maintain competence against known strategies)
- Randomly-initialized policies (to prevent overfitting to one meta)

## Realistic Expectations

### What's achievable with the current approach

**With more training time (16M+ timesteps, ~3 hours on A100):**
- L1-L6 graduation: Achievable, already demonstrated
- L7 Expert graduation at 70% WR: Possible with 256-256 network + longer training
- Beating individual expert fighters 30-50%: Achievable for some (counter_puncher, out_fighter)
- Consistent 50%+ WR against ALL expert fighters: Unlikely with current setup

**With architecture improvements (256-256 MLP, sparse rewards for L6+):**
- L7 graduation: Likely
- 40-60% WR against expert fighters in stochastic mode: Achievable
- Population fighters with distinct styles: Achievable with mixed opponent pools

### What's NOT achievable without bigger changes

**> 70% WR against hand-crafted experts** — The expert fighters exploit specific combat mechanics (jab range, stamina cycling, momentum) with hand-tuned thresholds. A PPO policy needs to discover these same strategies through trial and error. Without:
- Self-play with diverse opponents (not just clones)
- Larger networks (256-256 minimum)
- Much more training time (100M+ timesteps)
- Possibly curriculum stages specifically designed to teach patience, retreating, and timing

...the policy will plateau at "aggressive but undisciplined" — which beats test dummies and loses to experts.

**Human-competitive play** — The expert fighters ARE roughly human-level (simple but strategic). Matching them requires the RL agent to independently discover multi-phase strategies. This is fundamentally hard for PPO and would benefit from:
- Imitation learning (pre-train on expert fighter trajectories)
- Hierarchical RL (high-level strategy selection + low-level action execution)
- Model-based planning (learn a world model, plan ahead)

These are research-level additions, not implementation fixes.

## Recommended Next Steps (Priority Order)

### 1. Increase network size (easy, high impact)
Change PPO's policy network from 64-64 to 256-256. One-line config change. Gives 16x more parameters for learning complex strategies.

### 2. Increase training budget (easy, medium impact)
Bump curriculum timesteps from 8M to 16M. Gives L7 Expert twice as many episodes to learn. Cost: ~40 more minutes on A100.

### 3. Simplify rewards for later levels (medium, high impact)
Use shaped rewards (proximity, stance, etc.) for L1-L4 where the policy needs guidance. Switch to sparse rewards (damage + win/loss) for L5+ where the shaped rewards can mislead. The `LEVEL_REWARD_WEIGHTS` system already supports this — set all secondary weights to 0 for Expert/Gauntlet.

### 4. Mix curriculum opponents into population (medium, high impact)
Include 2-3 curriculum opponents in each population evaluation round to prevent catastrophic forgetting. The infrastructure exists — just expand the opponent pool for population evaluation.

### 5. Larger population with random initialization (hard, high impact)
Start population with 16+ fighters, some initialized randomly (not all from curriculum graduate). Creates genuine strategic diversity from the start.
