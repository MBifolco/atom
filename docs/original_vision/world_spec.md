
# 🌍 Atom Combat — World Specification (Component)

## Purpose
The **World Spec** defines the *laws of physics and environmental constraints* that all fighters must obey.
It ensures that designing or training a fighter is always a game of trade-offs — every advantage comes with a cost.

> The World Spec is the invisible rulebook that keeps the sport honest.

---

## Philosophy: Hybrid Physics Model (Full Vision)

Atom Combat uses a **hybrid physics model** where:
- **Movement is continuous**: position, velocity, and acceleration follow real physics (forces, friction, momentum)
- **Body control is articulated**: fighters control body parts (arms, legs, torso) with physics constraints (joints, mass distribution)
- **Combat emerges from collision**: damage occurs when body parts collide, scaled by mass, velocity, and impact location
- **No artificial "moves"**: attacks are emergent from body positioning and physics, not predefined actions

This creates a system where:
- Strategy comes from body control and positioning, not memorizing move lists
- Fights are realistic and physics-based
- Complexity emerges from simple, universal rules
- Fighter design involves real mechanical trade-offs

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

## Core Parameters (Full Vision - 2D/3D)

| Category | Example Fields | Description |
|-----------|----------------|--------------|
| **Geometry** | `arena_width`, `arena_height`, `arena_depth` | Defines spatial limits; larger arenas reward mobility. |
| **Gravity & Physics** | `gravity`, `friction`, `air_drag`, `bounce_coeff` | Governs how movement and knockback behave. |
| **Time & Simulation** | `tick_rate`, `max_ticks`, `dt` | Sets pacing; defines how often fighters act per second. |
| **Energy Model** | `stamina_drain_rate`, `regen_rate`, `acceleration_cost` | Controls energy economy and fatigue. |
| **Damage Model** | `base_collision_damage`, `velocity_scale`, `mass_scale` | Defines how collisions translate into damage. |
| **Body Constraints** | `max_mass`, `max_height`, `max_width_percent` | Caps how large or heavy a fighter can be relative to the arena. |
| **Body Parts** | `joint_constraints`, `articulation_limits`, `part_masses` | Defines how body parts can move and interact. |
| **Arena Features** | `hazards`, `zones`, `walls` | Optional features that add complexity (per division). |

---

## POC: 1D Simplified Implementation

The **1D POC** simplifies the full vision while preserving core principles:

### Simplifications for POC:
1. **1D space instead of 2D/3D**: Fighters move along a line (position is a single number)
2. **Discrete body states instead of articulated parts**: Fighters choose "stances" that define their geometry
3. **Simplified collision**: Overlap detection based on position + stance reach
4. **No gravity/jumping**: Just horizontal movement and collisions

### What's Preserved:
- ✓ Continuous physics (position, velocity, acceleration, friction)
- ✓ Collision-based combat (no predefined "attack" moves)
- ✓ Mass/velocity affect damage
- ✓ Stamina economy with costs and regen
- ✓ Trade-offs in fighter design

### POC Stances (Simplified Body Control)

Instead of controlling individual body parts, fighters choose discrete stances:

| Stance | Purpose | Reach | Width | Stamina Drain | Defense |
|--------|---------|-------|-------|---------------|---------|
| `neutral` | Balanced, energy-efficient | 0.2m | 0.3m | 0.0 | 1.0x |
| `extended` | Reaching forward to hit | 0.6m | 0.2m | 0.05/tick | 0.8x (exposed) |
| `retracted` | Pulled back, compact | 0.1m | 0.2m | 0.02/tick | 1.0x |
| `defending` | Armored, wide stance | 0.3m | 0.4m | 0.03/tick | 1.5x (protected) |

**How stances scale to full vision:**
- POC: Choose one of 4 discrete stances
- v2 (2D): Add Y-axis, stances become more granular (high/mid/low)
- v3 (Full): Replace stances with continuous body part control, stances emerge from configuration

### POC Parameters (1D)

```json
{
  "id": "world_1d_poc_v1",
  "version": "1.0.0",
  "arena": {
    "width": 10.0,
    "boundary_type": "hard_wall"
  },
  "physics": {
    "friction_coefficient": 0.8,
    "max_acceleration": 5.0,
    "max_velocity": 3.0,
    "dt": 0.067
  },
  "timing": {
    "tick_rate": 15,
    "round_duration_ticks": 600,
    "decision_time_budget_ms": 50
  },
  "stamina": {
    "acceleration_cost": 0.1,
    "base_regen": 0.15,
    "neutral_stance_bonus": 1.5
  },
  "stances": {
    "neutral": {"reach": 0.2, "width": 0.3, "drain": 0.0, "defense": 1.0},
    "extended": {"reach": 0.6, "width": 0.2, "drain": 0.05, "defense": 0.8},
    "retracted": {"reach": 0.1, "width": 0.2, "drain": 0.02, "defense": 1.0},
    "defending": {"reach": 0.3, "width": 0.4, "drain": 0.03, "defense": 1.5}
  },
  "damage": {
    "base_collision_damage": 10.0,
    "velocity_damage_scale": 2.0,
    "mass_damage_scale": 0.5
  },
  "body_constraints": {
    "max_mass": 100,
    "min_mass": 40,
    "max_hp": 100,
    "max_stamina": 10.0
  }
}
```

### POC Motion & Collision (Per Tick)

1. **Collect Actions**: Both fighters submit `{acceleration: float, stance: enum}`
2. **Update Velocities**:
   ```
   v_new = (v_old + acceleration * dt) * (1 - friction * dt)
   v_new = clamp(v_new, -max_velocity, max_velocity)
   ```
3. **Update Positions**:
   ```
   p_new = p_old + v_new * dt
   p_new = clamp(p_new, 0, arena_width)
   ```
4. **Compute Footprints** (fighter occupies space):
   ```
   Fighter at position p with stance s:
   - Occupies: [p - width/2, p + reach]
   ```
5. **Detect Collisions**: If footprints overlap → collision event
6. **Calculate Damage**:
   ```
   damage = base_damage
            * (1 + |relative_velocity| * velocity_scale)
            * (attacker_mass / defender_mass) ^ mass_scale
            / defender_stance_defense
   ```
7. **Update Stamina**:
   ```
   stamina_delta = -|acceleration| * accel_cost * dt
                   - stance_drain
                   + base_regen * stance_regen_multiplier
   ```

---

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | v2 (2D) | Full Vision (2D/3D) |
|--------|----------|---------|---------------------|
| **Space** | 1D line | 2D arena (X, Y) | 3D arena with vertical space |
| **Body** | 4 discrete stances | 12+ stances (high/mid/low × 4) | Articulated body parts with joints |
| **Movement** | Horizontal acceleration | X/Y acceleration + rotation | Full 3D movement + balance |
| **Combat** | Overlap → damage | Hit zones, angles matter | Body part collisions, impact physics |
| **Damage** | Velocity + mass | + angle, + part hit | + torque, joint damage, KO mechanics |

**What stays constant:**
- Deterministic physics simulation
- Collision-based combat (no "move lists")
- Mass/velocity trade-offs
- Stamina economy
- Fair, reproducible rules

---

**One-liner summary:**
*The World Spec defines the laws of physics and balance — starting with 1D simplified stances in the POC, evolving toward fully articulated physics-based combat.*
