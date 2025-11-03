
# 🏟️ Atom Combat — The Arena

## Purpose
The **Arena** is the **stage and the referee** of Atom Combat.  
It defines the physical world where fights occur and enforces the shared rules that every fighter must obey.  
It does not think, predict, or decide — it only **applies the laws of motion and combat** that make the match fair and deterministic.

In short:
> The Arena is where the world lives, and where every action has consequence.

---

## What the Arena Does

### 1. Hosts the World
The Arena defines:
- The **space** in which fighters exist (2D or 3D bounds).  
- **Obstacles** or special zones (walls, platforms, hazards).  
- **Environmental parameters** — gravity, friction, energy decay, etc.  

Each league division can have its own arena layouts or physics presets, but all arenas within that division follow the same underlying ruleset.

### 2. Applies Physics & Movement
The Arena is responsible for all motion and interactions:
- Moves fighters based on their chosen actions.  
- Applies momentum, inertia, drag, and balance.  
- Checks for collisions and constraints (e.g., floor, walls, joint limits).  
- Updates stamina, cooldowns, and timing.  

Every tick (moment of simulated time), the Arena **steps forward** according to the laws defined by the current protocol version.

### 3. Resolves Combat
When attacks and defenses overlap in time and space:
- The Arena determines if a **hit** occurs.  
- Calculates **damage**, **knockback**, **stun**, and **energy cost**.  
- Updates each fighter’s state accordingly (health, stamina, posture).  
- Emits combat events like `HIT`, `BLOCK`, `KO`, or `MISS`.

The logic is **transparent and reproducible** — given the same actions and seed, the result is always identical.

### 4. Enforces Rules & Boundaries
The Arena also acts as the referee:
- Validates that every fighter’s move is **legal** and within time limits.  
- Enforces **round timers** and **win conditions**.  
- Declares outcomes (knockout, timeout, draw).  
- Signals the **end of a round or match** to the orchestrator.

### 5. Emits the Official Record
At each tick, the Arena outputs:
- Positions, velocities, and states of all fighters.  
- Combat events and environment changes.  
- Time remaining and round status.  

These outputs form the **official replay log**, which all other systems (training, evaluation, rendering) use later.

---

## What the Arena Is *Not*
To keep the sport fair and deterministic:
- It does **not** include any fighter intelligence or randomness beyond seeded physics.  
- It does **not** decide tactics — only applies the consequences.  
- It does **not** change between fighters or teams; it’s identical for both sides.  

Think of it as **the laws of nature** in Atom Combat — consistent, predictable, and unbendable.

---

## Configurable Parameters
Each Arena has a small set of tunable, **protocol-approved** parameters:

| Category | Example Parameters | Purpose |
|-----------|--------------------|----------|
| Physics | gravity, drag, restitution, max speed | Core movement realism |
| Environment | size, walls, hazards, arena zones | Strategic variation |
| Round Rules | duration, stamina regen rate, win condition | Pacing & fairness |
| Effects | camera hints, sound triggers, VFX IDs | Spectator experience (not gameplay) |

These can change **between seasons** (as part of a league patch) but never during a match.

---

## Determinism & Fairness
The Arena runs deterministically:
- Every random number (e.g., slight variation in friction or damage roll) is seeded.  
- Given the same seed, fighter inputs, and protocol version, **the replay will always produce the same outcome**.  
- This ensures all fights can be audited, replayed, and verified.

---

## Example Lifecycle (simplified)
```
1. Initialize Arena (load layout, rules, seed)
2. Place fighters at starting positions
3. Loop (tick):
      - Apply fighter actions
      - Step physics
      - Resolve hits and effects
      - Log new state
4. End round if timer == 0 or fighter HP <= 0
5. Emit final match summary
```

---

## Why the Arena Matters
- It ensures **every match is played on equal ground**.  
- It makes the entire system **reproducible and fair**.  
- It separates **creative strategy** (the fighter’s mind) from **natural law** (the world).  

When people watch an Atom Combat fight, what they’re really watching is:
> Two ideas colliding under the same unbreakable physics.

---

**One-liner summary:**  
*The Arena is the neutral stage where fairness lives — a deterministic, physics-based world that turns fighter decisions into visible, verifiable outcomes.*
