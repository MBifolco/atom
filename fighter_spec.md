
# 🥋 Atom Combat — Fighter Specification (Component)

## Purpose
The **Fighter Specification** (or **Fighter Spec**) defines **what a fighter is** inside the Atom Combat world —  
its physical body, abilities, and limits.  

It’s the “character sheet” that determines what a fighter *can* do, but not *how* it decides to do it.  
Every fighter must conform to the same overall schema so matches remain fair and interpretable.

> The Fighter Spec defines the body and rules of a combatant — the mind is separate.

---

## What the Fighter Spec Represents
The Fighter Spec captures three key aspects:

| Aspect | Description |
|---------|--------------|
| **Body** | The structure, mass, and movement limits of the fighter (how it exists in the world). |
| **Abilities** | The list of legal actions and what each one costs or causes. |
| **Stats** | Base health, stamina, recovery, and modifiers that govern pacing and risk. |

Each fighter in a given division follows the same schema and physics rules — only parameter values differ.

---

## Fighter Spec Breakdown

### 1. Core Identity
- `id` — unique identifier for the fighter type  
- `version` — which schema version this spec follows  
- `name` — display name for replays and rosters  
- `tags` — keywords like `"light"`, `"counter"`, `"prototype"`

### 2. Physical Model (“The Body”)
| Field | Example | Description |
|--------|----------|-------------|
| `mass` | 70.0 | Total mass for physics calculations. |
| `height` | 1.8 | Visual/physical height (used for reach). |
| `collision_shape` | `"capsule"` | Shape type for hit detection. |
| `reach` | 0.6 | Base attack reach multiplier. |
| `speed` | 1.0 | Movement speed factor. |

### 3. Stats
| Stat | Description |
|-------|--------------|
| `max_hp` | Maximum health points. |
| `stamina_max` | Maximum stamina (energy for actions). |
| `regen_rate` | How quickly stamina recovers per tick. |
| `attack_power` | Base multiplier for damage. |
| `defense_power` | Reduces incoming damage. |

### 4. Moveset
Each move includes:
| Field | Description |
|--------|--------------|
| `id` | Move identifier (e.g., `light_attack`) |
| `cost` | Stamina cost per use |
| `cooldown` | Ticks before reuse |
| `damage` | Base damage amount |
| `range` | Effective reach |
| `startup`, `active`, `recovery` | Timing windows in ticks |

Example:
```json
{
  "id": "light_attack",
  "cost": 0.5,
  "damage": 5,
  "range": 1.0,
  "startup": 2,
  "active": 3,
  "recovery": 5,
  "cooldown": 8
}
```

### 5. Fighter Artifact (runtime form)
At match time, each fighter’s **mind** (decision model) is paired with its **spec** to create a **Fighter Artifact** —  
the self-contained package used in the arena.

A Fighter Artifact includes:
- The **Fighter Spec**
- The **decision logic**
- Metadata: creator, version, checksum, and validation info

This artifact is immutable during a fight.

---

## Example Minimal Fighter Spec
```json
{
  "id": "stickfighter_v1",
  "name": "Stick Fighter",
  "version": "1.0",
  "mass": 70.0,
  "height": 1.8,
  "stats": {
    "max_hp": 100,
    "stamina_max": 10,
    "regen_rate": 0.1
  },
  "moveset": [
    {"id":"idle","cost":0,"cooldown":0},
    {"id":"step_forward","cost":0.2,"cooldown":1},
    {"id":"light_attack","cost":0.5,"damage":5,"cooldown":5},
    {"id":"heavy_attack","cost":1.5,"damage":12,"cooldown":10},
    {"id":"block","cost":0.3,"cooldown":3}
  ]
}
```

---

**One-liner summary:**  
*The Fighter Spec defines the body, stats, and legal actions of a combatant — the physical blueprint that every “mind” must inhabit during battle.*
