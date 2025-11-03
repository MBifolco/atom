
# 🌍 Atom Combat — World Specification (Component)

## Purpose
The **World Spec** defines the *laws of physics and environmental constraints* that all fighters must obey.  
It ensures that designing or training a fighter is always a game of trade-offs — every advantage comes with a cost.  

> The World Spec is the invisible rulebook that keeps the sport honest.

---

## What the World Spec Does

| Function | Description |
|-----------|--------------|
| **Defines the physical environment** | Gravity, friction, time step, arena size, air resistance, surface properties. |
| **Sets physical constraints** | Maximum size, weight, and volume a fighter can occupy. |
| **Controls motion & energy** | Acceleration, speed limits, inertia, momentum conservation. |
| **Shapes combat feel** | How hard hits land, how knockback works, how stamina or fatigue behave. |
| **Enforces fairness** | Keeps all fighters within measurable, reproducible bounds of motion and energy. |

---

## Design Philosophy
The World Spec makes sure that:
- **Bigger** fighters hit harder but move slower.  
- **Smaller** fighters are faster but have less reach and HP.  
- **Heavier** fighters resist knockback but burn more stamina when moving.  
- **Light** fighters can reposition quickly but are easier to stagger.

It transforms fighter design into a *creative optimization problem* instead of a tech arms race.

---

## Core Parameters

| Category | Example Fields | Description |
|-----------|----------------|--------------|
| **Geometry** | `arena_width`, `arena_height`, `arena_depth` | Defines spatial limits; larger arenas reward mobility. |
| **Gravity & Physics** | `gravity`, `friction`, `air_drag`, `bounce_coeff` | Governs how movement and knockback behave. |
| **Time & Simulation** | `tick_rate`, `max_ticks`, `dt` | Sets pacing; defines how often fighters act per second. |
| **Energy Model** | `stamina_drain_rate`, `regen_rate`, `move_cost_multiplier` | Controls energy economy and fatigue. |
| **Damage Model** | `damage_scaling`, `knockback_scale`, `block_efficiency` | Defines how hits translate into consequences. |
| **Body Constraints** | `max_mass`, `max_height`, `max_width_percent` | Caps how large or heavy a fighter can be relative to the arena. |
| **Arena Features** | `hazards`, `zones`, `walls` | Optional features that add complexity (per division). |

---

## Example World Spec (Simplified JSON)
```json
{
  "id": "world_v1",
  "version": "1.0",
  "arena": {
    "width": 10.0,
    "height": 5.0,
    "gravity": 9.8,
    "friction": 0.9,
    "air_drag": 0.1
  },
  "timing": {
    "tick_rate": 15,
    "round_duration": 60
  },
  "energy": {
    "stamina_drain_rate": 0.05,
    "stamina_regen_rate": 0.02,
    "move_cost_multiplier": 1.0
  },
  "damage": {
    "base_scale": 1.0,
    "knockback_scale": 0.5,
    "block_efficiency": 0.7
  },
  "body_constraints": {
    "max_mass": 120,
    "min_mass": 40,
    "max_height": 2.5,
    "max_width_percent": 0.1
  }
}
```

---

**One-liner summary:**  
*The World Spec defines the laws of physics and balance — ensuring every fighter must play by the same physical reality, turning design itself into a strategic game of trade-offs.*
