
# 🥋 Atom Combat — Fighter Specification (Component)

## Purpose
The **Fighter Specification** (or **Fighter Spec**) defines **what a fighter is** inside the Atom Combat world —
its physical body, abilities, and limits.

It's the "character sheet" that determines what a fighter *can* do, but not *how* it decides to do it.
Every fighter must conform to the same overall schema so matches remain fair and interpretable.

> The Fighter Spec defines the body and rules of a combatant — the mind is separate.

---

## Full Vision: Physics-Based Body

In the full vision, a Fighter Spec defines:
- **Articulated body structure**: joints, limbs, torso with mass distribution
- **Physical constraints**: joint angles, extension limits, balance requirements
- **Material properties**: density, elasticity, damage resistance per body part
- **Sensor configuration**: what the fighter can perceive and how accurately

Combat emerges from how fighters position and move their body parts within physics constraints.

---

## What the Fighter Spec Represents (Full Vision)
The Fighter Spec captures key aspects of a fighter's physical form:

| Aspect | Description |
|---------|--------------|
| **Body Structure** | Articulated parts, joints, mass distribution, dimensions |
| **Physical Properties** | Density, elasticity, damage resistance per part |
| **Sensors** | What the fighter can perceive, sensor quality and range |
| **Stats** | Base health, stamina, recovery modifiers |
| **Constraints** | Movement limits, joint angles, balance requirements |

Each fighter in a given division follows the same schema and physics rules — only parameter values differ.

---

## POC: 1D Simplified Fighter Spec

For the **1D POC**, we simplify while preserving core trade-offs:

### What the POC Fighter Spec Defines

| Aspect | Description |
|---------|--------------|
| **Core Identity** | ID, name, version |
| **Physical Body** | Mass (affects damage dealt/taken, stamina costs) |
| **Stats** | HP, stamina pool |
| **Sensors** | How precisely the fighter perceives opponent |

### POC Fighter Spec Structure

#### 1. Core Identity
```json
{
  "id": "aggressive_bot_v1",
  "name": "Aggressive Bot",
  "version": "1.0",
  "creator": "system",
  "tags": ["poc", "aggressive", "lightweight"]
}
```

#### 2. Physical Body (1D)
```json
{
  "mass": 60.0  // kg - affects damage and stamina cost
}
```

**Trade-offs:**
- Higher mass → more damage dealt, more damage resistance, higher stamina cost for acceleration
- Lower mass → less damage, moves more efficiently, easier to push around

#### 3. Stats
```json
{
  "max_hp": 100,
  "max_stamina": 10.0
}
```

**Note:** Stamina regen and stance properties come from World Spec (universal rules)

#### 4. Sensors (Perception Quality)
```json
{
  "sensors": {
    "distance_precision": 0.5,     // Can detect opponent within ±0.5m
    "velocity_precision": 0.3,     // Can detect opponent velocity within ±0.3 m/s
    "stance_detection": "fuzzy"    // "exact", "fuzzy", or "none"
  }
}
```

**How sensors work:**
- **distance_precision**: Opponent distance is bucketed to this resolution
  - Precision 0.5: see 7.3m as ~7.5m
  - Precision 0.1: see 7.3m as ~7.3m
- **velocity_precision**: Same for opponent velocity
- **stance_detection**:
  - `"exact"`: See opponent's actual stance
  - `"fuzzy"`: See general category (attacking/defending/neutral)
  - `"none"`: No stance information

**Trade-offs for future divisions:**
- Better sensors might cost mass, stamina drain, or HP
- Creates strategic choice: precision vs. other advantages

### POC Fighter Artifact (Runtime Package)

At match time, the **Fighter Artifact** packages everything needed to run the fighter:

```json
{
  "spec": { /* Fighter Spec as above */ },
  "runtime": {
    "type": "python_function",
    "entrypoint": "decide",
    "code_hash": "sha256:abc123..."
  },
  "metadata": {
    "created": "2025-11-03T12:00:00Z",
    "certified": true,
    "protocol_version": "proto_v1",
    "world_spec": "world_1d_poc_v1"
  }
}
```

**For POC:** Runtime is simply a Python function with signature:
```python
def decide(snapshot: dict) -> dict:
    """
    Args:
        snapshot: Current state visible to this fighter
    Returns:
        {"acceleration": float, "stance": str}
    """
    pass
```

---

## Example POC Fighter Specs

### Lightweight Aggressive Fighter
```json
{
  "id": "aggressive_bot",
  "name": "Rusher",
  "version": "1.0",
  "mass": 50.0,
  "stats": {
    "max_hp": 80,
    "max_stamina": 12.0
  },
  "sensors": {
    "distance_precision": 0.5,
    "velocity_precision": 0.5,
    "stance_detection": "fuzzy"
  }
}
```
**Strategy:** Low mass = efficient movement, high stamina = can maintain pressure

### Heavyweight Tank Fighter
```json
{
  "id": "tank_bot",
  "name": "Tank",
  "version": "1.0",
  "mass": 90.0,
  "stats": {
    "max_hp": 120,
    "max_stamina": 8.0
  },
  "sensors": {
    "distance_precision": 1.0,
    "velocity_precision": 1.0,
    "stance_detection": "none"
  }
}
```
**Strategy:** High mass = devastating hits, high HP = durable, but slow and imprecise

---

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | Full Vision (2D/3D) |
|--------|----------|---------------------|
| **Body** | Single mass value | Articulated parts with individual masses |
| **Stats** | HP, stamina | + body part health, joint strain, fatigue |
| **Sensors** | Distance/velocity precision | + visual cones, audio range, reaction time |
| **Control** | Choose stance (4 options) | Control individual body part positions/forces |
| **Collision** | Single footprint | Multiple body part geometries |

**What stays constant:**
- Spec defines physical capabilities, not strategy
- Trade-offs in design (no strictly superior fighter)
- Sensors define information quality
- Artifact packages spec + runtime + metadata

---

**One-liner summary:**
*The Fighter Spec defines the physical body and sensors — starting with mass and perception in the POC, evolving toward fully articulated, multi-sensory combatants.*
