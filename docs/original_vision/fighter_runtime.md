
# 🧠 Atom Combat — Fighter Runtime (Component)

## Purpose
The **Fighter Runtime** is the "mind in the ring."
It's the logic module that receives snapshots from the Arena and decides what actions to take each tick within a strict time limit.

It doesn't learn, predict, or change during a match — it simply acts, quickly and deterministically.

> The Fighter Runtime is where instinct lives — fast, bounded decision-making under pressure.

---

## Full Vision: Articulated Body Control

In the full vision, the runtime controls individual body parts:
- Decide forces/positions for each joint and limb
- Balance, stance, and weight distribution
- Complex multi-part coordination (punching while stepping)

---

## POC: Acceleration + Stance Control (1D)

For the **1D POC**, the runtime is much simpler:

### Responsibilities
- Read and interpret the snapshot provided by the Orchestrator
- Decide acceleration and stance for this tick
- Return action before the time budget expires
- Never alter game state directly — only act through the protocol

### Behavior Contract
| Input | Output | Constraint |
|--------|----------|-------------|
| Snapshot (sensor-filtered world view) | Action: acceleration + stance | Must return within 50ms |

---

## POC Fighter Interface

Every POC fighter implements this simple function signature:

```python
def decide(snapshot: dict) -> dict:
    """
    Make a decision based on current snapshot.

    Args:
        snapshot: Current state visible to this fighter (filtered by sensors)
            {
                "tick": int,
                "you": {"position": float, "velocity": float, "hp": float, "stamina": float, "stance": str},
                "opponent": {"distance": float, "velocity": float, "stance_hint": str},
                "arena": {"width": float, ...}
            }

    Returns:
        action: {"acceleration": float, "stance": str}
            - acceleration: m/s² in range [-5.0, 5.0]
            - stance: one of ["neutral", "extended", "retracted", "defending"]
    """
    pass
```

---

## Example POC Fighters

### 1. Aggressive Fighter
Always rushes forward and extends when close:

```python
def decide(snapshot):
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    # If close, extend to hit
    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}

    # If have stamina, rush forward
    elif stamina > 2.0:
        return {"acceleration": 5.0, "stance": "neutral"}

    # Low stamina - conserve energy
    else:
        return {"acceleration": 0.0, "stance": "neutral"}
```

**Strategy:** Simple pressure - always moving forward, extends when in range.

### 2. Defensive Counter-Fighter
Waits for opponent, defends when close:

```python
def decide(snapshot):
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    my_hp = snapshot["you"]["hp"]

    # If opponent rushing toward me (negative velocity = approaching)
    if opp_velocity < -0.5 and distance < 2.0:
        # Defend if low HP, counter-extend if healthy
        if my_hp < 50:
            return {"acceleration": 0.0, "stance": "defending"}
        else:
            return {"acceleration": 2.0, "stance": "extended"}

    # If far away, slowly advance
    elif distance > 5.0:
        return {"acceleration": 2.0, "stance": "neutral"}

    # Mid-range - maintain position
    else:
        return {"acceleration": 0.0, "stance": "neutral"}
```

**Strategy:** Reactive - responds to opponent behavior, uses defending stance when threatened.

### 3. Stamina Manager
Balances aggression with energy conservation:

```python
def decide(snapshot):
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]
    velocity = snapshot["you"]["velocity"]

    # High stamina - aggressive
    if stamina > 7.0:
        if distance < 1.5:
            return {"acceleration": 3.0, "stance": "extended"}
        else:
            return {"acceleration": 4.0, "stance": "neutral"}

    # Medium stamina - cautious
    elif stamina > 3.0:
        if distance < 1.0:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 1.0, "stance": "neutral"}

    # Low stamina - recover
    else:
        # Retreat if too close
        if distance < 2.0:
            return {"acceleration": -2.0, "stance": "retracted"}
        # Stand still in neutral to regen faster
        else:
            return {"acceleration": 0.0, "stance": "neutral"}
```

**Strategy:** Energy-aware - shifts tactics based on stamina level.

### 4. Position-Based Fighter
Uses arena positioning strategically:

```python
def decide(snapshot):
    my_position = snapshot["you"]["position"]
    arena_width = snapshot["arena"]["width"]
    distance_to_right = snapshot["arena"]["your_distance_to_right_wall"]
    distance = snapshot["opponent"]["distance"]

    # Avoid being cornered
    if distance_to_right < 1.0:
        # Trapped on right side - must fight
        return {"acceleration": -3.0, "stance": "extended"}

    elif my_position < 1.0:
        # Trapped on left side - must fight
        return {"acceleration": 3.0, "stance": "extended"}

    # Center control - maintain middle ground
    elif abs(my_position - arena_width/2) > 2.0:
        # Too far from center, move back
        target_accel = 2.0 if my_position < arena_width/2 else -2.0
        return {"acceleration": target_accel, "stance": "neutral"}

    # Good position - engage opponent
    else:
        if distance < 2.0:
            return {"acceleration": 2.0, "stance": "extended"}
        else:
            return {"acceleration": 3.0, "stance": "neutral"}
```

**Strategy:** Spatial awareness - avoids walls, maintains center control.

---

## Determinism
- Every decision must be reproducible given the same snapshot
- If using randomness, must use the global match seed (passed via snapshot or initialization)
- Same snapshot → same action (critical for replay verification)

---

## Compliance

A Fighter Runtime must:
- Implement the division's Combat Protocol version (for POC: `proto_1d_poc_v1`)
- Respect tick timing (return within 50ms for POC)
- Output action format exactly as defined by protocol
- Handle edge cases gracefully (missing fields, unexpected values)

### Invalid Actions
If runtime returns invalid data:
- Missing fields → default to `{"acceleration": 0.0, "stance": "neutral"}`
- Out of range acceleration → clamped to `[-5.0, 5.0]`
- Invalid stance → forced to `"neutral"`

---

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | Full Vision (2D/3D) |
|--------|----------|---------------------|
| **Input** | Simple snapshot with distance/velocity | Visual fields, tactile, proprioception |
| **Output** | acceleration + stance | Forces/positions for each body part |
| **Complexity** | ~10-50 lines of logic | Potentially neural networks, RL policies |
| **Decision Time** | 50ms | Variable by division |

**What stays constant:**
- Snapshot → Decision → Action contract
- Time-bounded execution
- Deterministic behavior
- No state modification (only return actions)

---

**One-liner summary:**
*The Fighter Runtime is the fighter's decision core — starting with simple acceleration + stance logic in the POC, evolving toward complex multi-body control strategies.*
