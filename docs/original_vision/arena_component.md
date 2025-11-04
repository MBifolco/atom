
# 🏟️ Atom Combat — The Arena

## Purpose
The **Arena** is the **stage and the referee** of Atom Combat.  
It defines the physical world where fights occur and enforces the shared rules that every fighter must obey.  
It does not think, predict, or decide — it only **applies the laws of motion and combat** that make the match fair and deterministic.

In short:
> The Arena is where the world lives, and where every action has consequence.

---

## What the Arena Does (Full Vision)

### 1. Hosts the World
The Arena defines:
- The **space** in which fighters exist (2D or 3D bounds)
- **Obstacles** or special zones (walls, platforms, hazards)
- **Environmental parameters** — gravity, friction, energy decay, etc.

Each league division can have its own arena layouts or physics presets, but all arenas within that division follow the same underlying ruleset.

### 2. Applies Physics & Movement
The Arena is responsible for all motion and interactions:
- Applies forces to articulated body parts
- Simulates momentum, inertia, drag, and balance
- Checks for collisions and constraints (joint limits, ground contact)
- Updates stamina based on exertion

Every tick, the Arena **steps forward** according to the laws defined by the current protocol version.

### 3. Resolves Combat (Collision-Based)
When body parts collide:
- The Arena detects **geometric overlaps**
- Calculates **damage** based on mass, velocity, impact location
- Applies **knockback** and physics responses
- Updates each fighter's state (health, stamina, balance)
- Emits combat events like `COLLISION`, `DAMAGE`, `KO`

The logic is **transparent and reproducible** — given the same actions and seed, the result is always identical.

### 4. Enforces Rules & Boundaries
The Arena acts as the referee:
- Validates that actions are within physical limits
- Enforces **round timers** and **win conditions**
- Declares outcomes (knockout, timeout, draw)
- Signals the **end of a round or match** to the orchestrator

### 5. Emits the Official Record
At each tick, the Arena outputs:
- Positions, velocities, and states of all fighters
- Combat events and collisions
- Time remaining and round status

These outputs form the **official replay log**, which all other systems (training, evaluation, rendering) use later.

---

## POC: 1D Arena Implementation

The **1D POC Arena** simplifies physics while preserving core mechanics.

### POC Arena Responsibilities

#### 1. Manage 1D World State
- Track fighter positions (single float per fighter: 0 to arena_width)
- Track fighter velocities
- Enforce boundary constraints (0 ≤ position ≤ 10.0)

#### 2. Apply Physics Each Tick

**Velocity Update:**
```python
# Apply acceleration (from fighter action)
v_new = v_old + acceleration * dt

# Apply friction
v_new = v_new * (1 - friction_coefficient * dt)

# Clamp to max velocity
v_new = clamp(v_new, -max_velocity, max_velocity)
```

**Position Update:**
```python
p_new = p_old + v_new * dt

# Enforce walls (hard boundaries)
p_new = clamp(p_new, 0.0, arena_width)

# If hit wall, zero velocity
if p_new == 0.0 or p_new == arena_width:
    v_new = 0.0
```

#### 3. Compute Footprints (Body Geometry)

Each fighter occupies space based on position + stance:

```python
def compute_footprint(position, stance, world_spec):
    stance_config = world_spec.stances[stance]
    reach = stance_config.reach
    width = stance_config.width

    # Fighter occupies [left_edge, right_edge]
    left_edge = position - width / 2
    right_edge = position + reach

    return (left_edge, right_edge)
```

**Example:**
- Fighter at position 5.0 with `extended` stance (reach=0.6, width=0.2):
  - Occupies [4.9, 5.6]
- Fighter at position 7.0 with `neutral` stance (reach=0.2, width=0.3):
  - Occupies [6.85, 7.2]

#### 4. Detect Collisions

```python
def detect_collision(footprint_a, footprint_b):
    (a_left, a_right) = footprint_a
    (b_left, b_right) = footprint_b

    # Check if ranges overlap
    return not (a_right < b_left or b_right < a_left)
```

#### 5. Calculate & Apply Damage

When collision detected:

```python
def calculate_damage(fighter_a, fighter_b, world_spec):
    # Relative velocity (closing speed)
    relative_velocity = abs(fighter_a.velocity - fighter_b.velocity)

    # Mass ratio
    mass_ratio = fighter_a.mass / fighter_b.mass

    # Stance defense multiplier
    defense_mult = world_spec.stances[fighter_b.stance].defense

    # Damage formula
    damage = world_spec.damage.base_collision_damage \
             * (1 + relative_velocity * world_spec.damage.velocity_damage_scale) \
             * (mass_ratio ** world_spec.damage.mass_damage_scale) \
             / defense_mult

    return damage
```

**Symmetric application:** If both fighters' footprints overlap, BOTH take damage (potentially different amounts based on mass/stance).

#### 6. Update Stamina

```python
def update_stamina(fighter, action, dt, world_spec):
    # Cost of acceleration
    accel_cost = abs(action.acceleration) * world_spec.stamina.acceleration_cost * dt

    # Stance drain
    stance_drain = world_spec.stances[action.stance].drain

    # Base regen
    regen = world_spec.stamina.base_regen

    # Neutral stance bonus
    if action.stance == "neutral":
        regen *= world_spec.stamina.neutral_stance_bonus

    # Apply delta
    stamina_delta = -accel_cost - stance_drain + regen
    fighter.stamina = clamp(fighter.stamina + stamina_delta, 0, fighter.max_stamina)

    # If stamina hits zero, force safe state
    if fighter.stamina == 0:
        fighter.stance = "neutral"
        fighter.velocity *= 0.5  # Lose momentum
```

#### 7. Emit Events

```python
events = []

if collision_detected:
    events.append({
        "type": "COLLISION",
        "tick": current_tick,
        "fighters": ["fighter_a", "fighter_b"],
        "damage": {"fighter_a": damage_to_a, "fighter_b": damage_to_b},
        "relative_velocity": rel_velocity
    })

if fighter.hp <= 0:
    events.append({
        "type": "KO",
        "tick": current_tick,
        "fighter": fighter.id
    })
```

### POC Tick Execution Order

```
1. Receive actions from both fighters
2. Validate actions (clamp acceleration, check stance validity)
3. Update velocities (apply acceleration + friction)
4. Update positions (apply velocity + boundary constraints)
5. Compute footprints for both fighters
6. Detect collisions
7. Calculate and apply damage (if collision)
8. Update stamina for both fighters
9. Check win conditions (HP ≤ 0 or time expired)
10. Emit events
11. Log state to telemetry
```

**Critical:** Steps 3-7 are **atomic and simultaneous** for both fighters — no turn order advantage.

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

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | Full Vision (2D/3D) |
|--------|----------|---------------------|
| **Space** | 1D line (position is float) | 2D/3D arena with obstacles |
| **Physics** | velocity, friction, boundaries | + gravity, torque, balance, joint constraints |
| **Body Geometry** | Footprint = position + stance | Articulated body parts with shapes |
| **Collision** | 1D interval overlap | Full geometric collision detection (SAT, GJK) |
| **Damage** | Collision → damage via formula | Impact point, force direction, body part vulnerability |
| **Simultaneity** | Both fighters act atomically per tick | Same principle, more complex state updates |

**What stays constant:**
- Deterministic physics simulation
- Collision-based combat (not move-based)
- Symmetric, simultaneous resolution
- Arena as neutral source of truth

---

## Why the Arena Matters
- It ensures **every match is played on equal ground**
- It makes the entire system **reproducible and fair**
- It separates **creative strategy** (the fighter's mind) from **natural law** (the world)

When people watch an Atom Combat fight, what they're really watching is:
> Two ideas colliding under the same unbreakable physics.

---

**One-liner summary:**
*The Arena is the neutral stage where fairness lives — a deterministic, physics-based world that turns fighter decisions into visible, verifiable outcomes — starting with 1D collisions in the POC, evolving toward full 3D physics.*
