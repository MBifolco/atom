# How AI Fighter Training Works

**Goal:** Train a neural network (AI model) to control a fighter in Atom Combat matches.

---

## Overview: Teaching a Fighter to Fight

Think of training an AI fighter like teaching a dog tricks:

1. **The dog does random things** (sit, roll, bark)
2. **You reward good behaviors** (treats when it sits)
3. **Over time, the dog learns** what actions lead to treats
4. **Eventually, it consistently does** the right thing

Our AI fighters learn the same way:
- They start with **random actions**
- They get **rewards** for good moves (landing hits, winning)
- After **thousands of matches**, they learn patterns
- Eventually, they develop a **fighting strategy**

---

## The Training Loop

```mermaid
graph TB
    Start[Start Training] --> Init[Create AI Fighter<br/>Random brain]
    Init --> Match[Run Match vs Opponent]
    Match --> Observe[Fighter observes:<br/>position, HP, distance, etc]
    Observe --> Decide[AI decides action:<br/>acceleration, stance]
    Decide --> Execute[Arena executes action<br/>Physics simulation]
    Execute --> Reward[Calculate reward:<br/>Did it help?]
    Reward --> Learn[AI adjusts its brain<br/>slightly]
    Learn --> Check{Match<br/>over?}
    Check -->|No| Observe
    Check -->|Yes| Next{More<br/>matches<br/>needed?}
    Next -->|Yes| Match
    Next -->|No| Done[Training Complete<br/>Save AI fighter]

    style Init fill:#e1f5ff
    style Reward fill:#ffe1e1
    style Learn fill:#fff4e1
    style Done fill:#e1ffe1
```

---

## What the AI Fighter "Sees" (Observations)

Every tick (game step), the fighter receives a **snapshot** of the game state:

```mermaid
graph LR
    subgraph Fighter's View
        A[My Position<br/>5.2m]
        B[My Velocity<br/>+1.5 m/s]
        C[My HP<br/>85%]
        D[My Stamina<br/>60%]
        E[Opponent Distance<br/>2.1m]
        F[Opponent Velocity<br/>-0.5 m/s]
        G[Opponent HP<br/>70%]
        H[Opponent Stamina<br/>80%]
        I[Arena Width<br/>12m]
    end

    A --> Brain[AI Brain<br/>Neural Network]
    B --> Brain
    C --> Brain
    D --> Brain
    E --> Brain
    F --> Brain
    G --> Brain
    H --> Brain
    I --> Brain

    Brain --> Out1[Acceleration<br/>+3.2 m/s²]
    Brain --> Out2[Stance<br/>extended]

    style Brain fill:#ffe1f5
    style Out1 fill:#e1ffe1
    style Out2 fill:#e1ffe1
```

**The fighter only sees these 9 numbers** - it must learn to fight with just this information!

---

## How the AI "Brain" Works

The AI uses a **neural network** - think of it as a mathematical function that transforms observations into actions.

```mermaid
graph LR
    subgraph Input Layer
        I1[Position]
        I2[Velocity]
        I3[HP]
        I4[Stamina]
        I5[Distance]
        I6[Opp Vel]
        I7[Opp HP]
        I8[Opp Stamina]
        I9[Arena Width]
    end

    subgraph Hidden Layers
        H1[Neuron]
        H2[Neuron]
        H3[Neuron]
        H4[...]
        H5[Neuron]
        H6[Neuron]
    end

    subgraph Output Layer
        O1[Acceleration<br/>-1.0 to +1.0]
        O2[Stance<br/>0 to 3.99]
    end

    I1 --> H1
    I1 --> H2
    I2 --> H1
    I2 --> H2
    I3 --> H3
    I4 --> H3
    I5 --> H4
    I6 --> H4
    I7 --> H5
    I8 --> H5
    I9 --> H6

    H1 --> O1
    H2 --> O1
    H3 --> O1
    H4 --> O1
    H5 --> O2
    H6 --> O2

    style H1 fill:#ffe1f5
    style H2 fill:#ffe1f5
    style H3 fill:#ffe1f5
    style H4 fill:#ffe1f5
    style H5 fill:#ffe1f5
    style H6 fill:#ffe1f5
```

**Key concept:** The network has **weights** (numbers) that determine how it transforms inputs to outputs. Training = adjusting these weights to get better results.

---

## The Reward System: Teaching Right from Wrong

The AI learns by receiving **rewards** (positive numbers = good, negative = bad).

```mermaid
graph TB
    Action[AI performs action] --> Result{What happened?}

    Result -->|Won the match!| Win["+200 to +250<br/>BIG REWARD"]
    Result -->|Lost the match| Loss["-200<br/>BIG PENALTY"]
    Result -->|Timeout winning on HP| TimeWin["+50<br/>Small reward"]
    Result -->|Timeout losing on HP| TimeLoss["-100<br/>Penalty"]
    Result -->|Landed a hit| Hit["+5.0<br/>Good!"]
    Result -->|Got in range| Range["+0.5<br/>Nice"]
    Result -->|Took damage| Damage["Damage × -2<br/>Ouch"]
    Result -->|Dealt damage| DealDmg["Damage × +2<br/>Yes!"]
    Result -->|Did nothing| Idle["-0.2<br/>Be active!"]

    Win --> Learn[AI remembers:<br/>Those actions were GOOD]
    Loss --> Learn2[AI remembers:<br/>Those actions were BAD]
    Hit --> Learn3[AI remembers:<br/>This was helpful]

    style Win fill:#e1ffe1
    style Loss fill:#ffe1e1
    style Hit fill:#e1f5ff
    style Learn fill:#fff4e1
    style Learn2 fill:#fff4e1
    style Learn3 fill:#fff4e1
```

### Current Reward Function

**Episode-ending rewards:**
- ✅ **Win by KO:** +200 + time bonus (faster wins = better)
- ❌ **Loss by KO:** -200
- 😐 **Timeout winning:** +50
- 😞 **Timeout losing:** -100
- 🤷 **Timeout tied:** -25

**Per-tick rewards (during fight):**
- 👊 **Land a hit:** +5.0 (encourages aggression)
- 🎯 **Get in range:** +0.5 if distance < 2m (encourages engagement)
- 💥 **Damage dealt:** damage × 2.0
- 🩹 **Damage taken:** damage × -2.0
- 😴 **No action:** -0.2 (discourages passivity)

---

## The Learning Algorithm (PPO)

We use **PPO (Proximal Policy Optimization)** - a popular algorithm for teaching AI to make decisions.

### How PPO Works (Simplified)

```mermaid
graph TB
    subgraph "1. Experience Collection"
        A[Run 1000 ticks<br/>across 10 parallel matches]
        A --> B[Collect all:<br/>observations, actions, rewards]
    end

    subgraph "2. Learning Step"
        B --> C[Look at what worked<br/>and what didn't]
        C --> D[Adjust neural network<br/>to repeat good actions]
        D --> E[But don't change<br/>TOO much at once]
    end

    subgraph "3. Repeat"
        E --> F[Run more matches<br/>with updated AI]
        F --> G{Getting<br/>better?}
        G -->|Yes| F
        G -->|No| H[Stop training]
    end

    style D fill:#ffe1f5
    style E fill:#fff4e1
    style H fill:#e1ffe1
```

**Key PPO principles:**
1. **Learn from experience:** Try actions, see what happens
2. **Small updates:** Don't change too drastically (prevents forgetting)
3. **Multiple attempts:** Same situations get tried many times
4. **Parallel training:** Run 10 matches at once (faster learning)

---

## Training Stages: What Actually Happens

```mermaid
graph LR
    subgraph "Stage 1: Random<br/>(Episodes 0-100)"
        R1[Random flailing]
        R2[No strategy]
        R3[Loses badly]
    end

    subgraph "Stage 2: Exploration<br/>(Episodes 100-500)"
        E1[Tries different things]
        E2[Sometimes lands hits]
        E3[Still losing most]
    end

    subgraph "Stage 3: Pattern Recognition<br/>(Episodes 500-2000)"
        P1[Learns: moving forward = combat]
        P2[Learns: extended stance = hits]
        P3[Still figuring out timing]
    end

    subgraph "Stage 4: Strategy<br/>(Episodes 2000+)"
        S1[Consistent approach]
        S2[Manages stamina]
        S3[Starts winning?]
    end

    R1 --> E1
    E1 --> P1
    P1 --> S1

    style R1 fill:#ffe1e1
    style E1 fill:#fff4e1
    style P1 fill:#e1f5ff
    style S1 fill:#e1ffe1
```

---

## Current Problems and Why Training is Hard

### Problem 1: The Task is Difficult

```mermaid
graph TB
    Challenge[Learning to fight is HARD] --> C1[Must learn physics]
    Challenge --> C2[Must learn timing]
    Challenge --> C3[Must predict opponent]
    Challenge --> C4[Must manage resources]

    C1 --> Example1[When do collisions happen?]
    C2 --> Example2[When to strike vs defend?]
    C3 --> Example3[Where will opponent be?]
    C4 --> Example4[When to conserve stamina?]

    style Challenge fill:#ffe1e1
```

**Why this matters:** The AI starts knowing NOTHING. It must discover these concepts from scratch through trial and error.

### Problem 2: Sparse Rewards

```mermaid
graph LR
    A[Episode starts] --> B[300-1000 ticks<br/>of actions]
    B --> C[Episode ends]
    C --> D[Get one big reward<br/>+200 or -200]

    D --> E{Problem:<br/>Which of those<br/>1000 actions<br/>caused the win?}

    style E fill:#ffe1e1
```

**Current solution:** Dense rewards (per-tick bonuses for hits, range, damage) provide more feedback.

### Problem 3: Exploration vs Exploitation

```mermaid
graph TB
    Dilemma[AI's Dilemma] --> Exploit[Do what I know works<br/>Safe but limited]
    Dilemma --> Explore[Try new things<br/>Risky but might find better strategy]

    Exploit --> Result1[Consistent performance<br/>but plateaus]
    Explore --> Result2[Inconsistent<br/>but might break through]

    style Dilemma fill:#fff4e1
```

**What we see:** AI finds a "local minimum" strategy (like avoiding combat) and gets stuck.

### Problem 4: Opponent Difficulty

```mermaid
graph TB
    Start[Random AI] --> Easy{Can beat<br/>opponent?}
    Easy -->|Never| Stuck[No wins = no positive signal<br/>Can't learn]
    Easy -->|Sometimes| Learn[Gets feedback<br/>Can improve]
    Easy -->|Always| Bored[No challenge<br/>Doesn't push skills]

    style Stuck fill:#ffe1e1
    style Learn fill:#e1ffe1
    style Bored fill:#fff4e1
```

**Current approach:** Train against `training_dummy.py` (stationary, same weight) to give the AI a fighting chance.

---

## Solution: Curriculum Learning (Progressive Difficulty Training)

**The key insight:** Just like humans learn martial arts through progressive difficulty (white belt → black belt), AI fighters learn best when opponents get progressively harder.

### What is Curriculum Learning?

Instead of throwing a random AI against a master fighter (Tank) and hoping it learns, we create a **progression of opponents** from trivial to expert:

```mermaid
graph LR
    L1[Level 1<br/>Training Dummy<br/>Stationary] --> L2[Level 2<br/>Wanderer<br/>Random movement]
    L2 --> L3[Level 3<br/>Bumbler<br/>Poor execution]
    L3 --> L4[Level 4<br/>Novice<br/>Basic competent]
    L4 --> L5[Level 5<br/>Rusher<br/>Aggressive pressure]
    L5 --> L6[Level 6<br/>Tank<br/>Defensive expert]
    L6 --> L7[Level 7<br/>Balanced<br/>Adaptive tactician]
    L7 --> L8[Level 8<br/>Self-Play<br/>vs itself]

    style L1 fill:#e1ffe1
    style L2 fill:#e1ffe1
    style L3 fill:#fff4e1
    style L4 fill:#fff4e1
    style L5 fill:#ffe1e1
    style L6 fill:#ffe1e1
    style L7 fill:#ffe1e1
    style L8 fill:#e1f5ff
```

### The 7-Level Training Curriculum

**Level 1: Training Dummy** (fighters/training_opponents/training_dummy.py)
- **Strategy:** Stands still, does nothing
- **Goal:** Learn basic mechanics (100% win rate)
- **What AI learns:** Movement causes collisions, collisions do damage
- **Expected time:** 500-1000 episodes

**Level 2: Wanderer** (fighters/training_opponents/wanderer.py)
- **Strategy:** Random movement, no fighting logic
- **Goal:** Learn positioning matters (90%+ win rate)
- **What AI learns:** Must track moving target, predict position
- **Expected time:** 1000-2000 episodes

**Level 3: Bumbler** (fighters/training_opponents/bumbler.py)
- **Strategy:** Tries to fight but poor timing/execution
- **Goal:** Learn timing and precision (80%+ win rate)
- **What AI learns:** WHEN you do things matters, not just WHAT
- **Expected time:** 2000-3000 episodes

**Level 4: Novice** (fighters/training_opponents/novice.py)
- **Strategy:** Competent fundamentals, but predictable
- **Goal:** Develop tactics (70%+ win rate)
- **What AI learns:** Must have a strategy, not just reactions
- **Expected time:** 3000-5000 episodes

**Level 5: Rusher** (fighters/examples/rusher.py)
- **Strategy:** Aggressive pressure fighter
- **Goal:** Learn counter-aggression and defense (60%+ win rate)
- **What AI learns:** Must defend, can't just attack
- **Expected time:** 5000-8000 episodes

**Level 6: Tank** (fighters/examples/tank.py)
- **Strategy:** Defensive counter-puncher
- **Goal:** Break through defense (55%+ win rate)
- **What AI learns:** Must force openings, patience required
- **Expected time:** 8000-12000 episodes

**Level 7: Balanced** (fighters/examples/balanced.py)
- **Strategy:** Adaptive tactician (adjusts to situation)
- **Goal:** Adapt to changing situations (50%+ win rate)
- **What AI learns:** Must read opponent and adjust strategy
- **Expected time:** 10000-15000 episodes

**Level 8: Self-Play**
- **Strategy:** Fight against copies of itself
- **Goal:** Continuous improvement
- **What AI learns:** Arms race - must keep innovating
- **Expected time:** Ongoing

### How to Train Through the Curriculum

**Step 1: Train against Level 1**
```bash
cd training
python train_fighter.py \
  --opponent ../fighters/training_opponents/training_dummy.py \
  --output fighter_level1 \
  --episodes 1000 \
  --cores 8 \
  --create-wrapper
```

**Step 2: Test graduation criteria**
```bash
# Run 10 matches to check win rate
cd ..
for i in {1..10}; do
  python atom_fight.py fighter_level1.py fighters/training_opponents/training_dummy.py --seed $i
done
```

**Step 3: If passing (100% wins), move to Level 2**
```bash
cd training
python train_fighter.py \
  --opponent ../fighters/training_opponents/wanderer.py \
  --output fighter_level2 \
  --episodes 2000 \
  --cores 8 \
  --create-wrapper
```

**Step 4: Repeat until reaching desired level**

### Multi-Opponent Training (Alternative Approach)

Instead of sequential training, train against **multiple difficulty levels** at once:

```bash
cd training
python train_fighter.py \
  --opponents ../fighters/training_opponents/*.py \
  --output versatile_fighter \
  --episodes 10000 \
  --cores 10 \
  --create-wrapper
```

**Pros:**
- Fighter learns to handle variety
- More robust to different styles
- Less manual progression management

**Cons:**
- Slower initial progress (harder average opponent)
- Might plateau at "good enough" against easy opponents
- Less focused skill development

### Why Curriculum Learning Works

```mermaid
graph TB
    Start[Random AI] --> Easy[Easy opponent]
    Easy --> Win1[Gets some wins]
    Win1 --> Signal[Positive reward signal]
    Signal --> Learn1[Learns basics]
    Learn1 --> Medium[Harder opponent]
    Medium --> Win2[Fewer wins, but some]
    Win2 --> Learn2[Refines strategy]
    Learn2 --> Hard[Expert opponent]
    Hard --> Win3[Rare wins]
    Win3 --> Learn3[Masters skills]

    Start -.Without curriculum.-> Hard2[Expert opponent]
    Hard2 --> NoWin[Zero wins]
    NoWin --> NoSignal[No positive signal]
    NoSignal --> Stuck[Gets stuck<br/>Can't learn]

    style Win1 fill:#e1ffe1
    style Win2 fill:#e1ffe1
    style Win3 fill:#e1ffe1
    style Stuck fill:#ffe1e1
```

**Key principle:** The AI needs to win SOMETIMES to get positive reinforcement, but not ALWAYS or it won't be challenged to improve.

### Complete Curriculum Guide

See **[fighters/training_opponents/OPPONENT_PROGRESSION.md](../fighters/training_opponents/OPPONENT_PROGRESSION.md)** for the complete curriculum guide with:
- Detailed opponent descriptions
- Graduation criteria for each level
- Training commands
- Expected time estimates
- What each level teaches

---

## Training Configuration

### What You Can Control

| Parameter | What It Does | Trade-off |
|-----------|--------------|-----------|
| `--episodes` | How many matches to run | More = better learning, but slower |
| `--cores` | Parallel matches | More = faster training, uses more CPU |
| `--max-ticks` | Match length limit | Shorter = faster training, but may cut off learning |
| `--opponent` | Who to fight | Easier = faster progress, harder = better final skill |
| `--mass` | Fighter weight | Affects combat style to learn |
| `--patience` | When to stop if not improving | Higher = more thorough, lower = faster completion |

### Recommended Settings

**Quick test (5-10 min):**
```bash
python train_fighter.py \
  --opponent ../fighters/training_opponents/training_dummy.py \
  --output test_ai \
  --episodes 1000 \
  --max-ticks 300 \
  --cores 4
```

**Serious training (1-2 hours):**
```bash
python train_fighter.py \
  --opponent ../fighters/training_opponents/training_dummy.py \
  --output trained_ai \
  --episodes 10000 \
  --max-ticks 500 \
  --cores 10
```

**Multi-opponent training (3-4 hours):**
```bash
python train_fighter.py \
  --opponents ../fighters/training_opponents/training_dummy.py ../fighters/examples/rusher.py \
  --output versatile_ai \
  --episodes 20000 \
  --max-ticks 500 \
  --cores 10
```

---

## What Good Training Looks Like

```mermaid
graph LR
    subgraph "Progress Indicators"
        M1[Mean Reward Increasing] -.-> G1[Improving!]
        M2[Episode Length Decreasing] -.-> G1
        M3[Consistent upward trend] -.-> G1
    end

    subgraph "Warning Signs"
        W1[Reward goes up then down] -.-> B1[Unstable learning]
        W2[No change in 1000+ episodes] -.-> B2[Stuck in local minimum]
        W3[Reward = 0 for many episodes] -.-> B3[Not learning to engage]
    end

    style G1 fill:#e1ffe1
    style B1 fill:#ffe1e1
    style B2 fill:#ffe1e1
    style B3 fill:#ffe1e1
```

**Example of good training output:**
```
Step 1,000  | Mean Reward: -180.0 ⬆️
Step 5,000  | Mean Reward: -120.0 ⬆️
Step 10,000 | Mean Reward: -60.0 ⬆️
Step 15,000 | Mean Reward: -20.0 ⬆️
Step 20,000 | Mean Reward: +40.0 ⬆️  ← Starting to win!
Step 25,000 | Mean Reward: +120.0 ⬆️
```

---

## What Happens After Training

```mermaid
graph LR
    Train[Training Complete] --> Save1[fighter.zip<br/>Neural network model]
    Train --> Save2[fighter.onnx<br/>Portable format]
    Train --> Save3[fighter.py<br/>Runnable wrapper]

    Save3 --> Use1[Test it:<br/>atom_fight.py fighter.py opponent.py]
    Save3 --> Use2[Tournament:<br/>atom tournament fighters/*.py]
    Save3 --> Use3[Share it:<br/>Upload to registry]

    style Save3 fill:#e1ffe1
```

The trained fighter is now a **black box** - you can't see its logic, but you can:
- ✅ Run it in matches
- ✅ Analyze its behavior
- ✅ Use it as an opponent for training others
- ✅ Continue training it further
- ✅ Share it with others

---

## Future: Scaling to Complex Worlds

```mermaid
graph TB
    subgraph "Current: Simple (9 inputs)"
        C1[Position, Velocity<br/>HP, Stamina, etc]
    end

    subgraph "Future: Medium (50+ inputs)"
        F1[Multiple stance types<br/>Weapon choices<br/>Terrain effects<br/>Special abilities]
    end

    subgraph "Future: Complex (1000+ inputs)"
        F2[2D/3D movement<br/>Team coordination<br/>Power-ups<br/>Environmental hazards<br/>Multiple opponents]
    end

    C1 --> F1
    F1 --> F2

    subgraph "Training Evolution"
        T1[Current: Hours on laptop]
        T2[Medium: Days on workstation]
        T3[Complex: Weeks on GPU cluster]
    end

    C1 -.-> T1
    F1 -.-> T2
    F2 -.-> T3

    style F2 fill:#e1f5ff
    style T3 fill:#ffe1f5
```

**The path forward:**
1. ✅ Get basic training working (current)
2. Improve reward shaping and training efficiency
3. Add curriculum learning (train on progressively harder opponents)
4. Scale to more complex worlds
5. Eventually: Pro teams with dedicated ML engineers optimizing training pipelines

---

## Key Takeaways

1. **AI fighters learn by trial and error** - no explicit programming
2. **Rewards shape behavior** - what you reward is what you get
3. **Training is hard** - especially with sparse feedback
4. **Current challenge:** Getting AI to learn winning strategies, not just survival
5. **This is a foundation** - as worlds get complex, training will too
6. **Pro leagues will require serious ML expertise** - that's the point!

---

## Next Steps to Improve Training

1. **Better reward shaping** - find the right balance of incentives
2. **Curriculum learning** - start easy, gradually increase difficulty
3. **Self-play** - train against copies of itself
4. **Longer training** - current runs are too short
5. **Hyperparameter tuning** - optimize learning rate, batch size, etc.
6. **Better baselines** - create a hierarchy of opponents (very easy → hard)

The goal: An AI that consistently beats `training_dummy`, then graduates to beating `rusher`, then `tank`, then other trained AIs.
