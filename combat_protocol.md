
# 🤝 Atom Combat — Combat Protocol (Component)

## Purpose
The **Combat Protocol** is the shared **language** between a fighter and the arena during a sanctioned match.
It defines **what fighters can sense**, **what they are allowed to do**, and **how quickly** they must respond.
This contract is identical for everyone in a given division, which is what makes the sport fair.

> One world, one rulebook, one interface — no matter how a fighter was built or trained.

---

## Full Vision: Body Control Protocol

In the full vision, the protocol defines:
- **Body part control inputs**: forces/positions for each articulated part
- **Rich sensory snapshot**: visual fields, tactile feedback, balance, proprioception
- **Physics constraints**: what forces can be applied, joint limits
- **Timing**: how often fighters receive updates and must respond

No predefined "moves" — just continuous control of body parts within physics limits.

---

## What the Protocol Guarantees
- **Fairness:** Every fighter receives the same kind of limited snapshot and the same list of legal moves.
- **Reproducibility:** Given the same seeds and the same move choices, the fight plays out exactly the same.
- **Timing discipline:** Fighters must return one legal move **within the per-tick time limit**.
- **Compatibility:** Different training methods/models can compete as long as they speak this protocol.

---

## Responsibilities (Overview)
1. **Sense:** Define the snapshot of the world a fighter is allowed to "see" each tick
2. **Act:** Define the control inputs fighters can provide
3. **Pace:** Define the decision rhythm (ticks per second) and the response time budget
4. **Record:** Provide a stable, replayable record of snapshots and actions
5. **Version:** Evolve carefully — divisions pin to a specific protocol version

---

## POC: 1D Control Protocol

The **1D POC protocol** simplifies control while maintaining fairness and physics-based combat.

### Control Inputs (Fighter → Arena)

Each tick, fighters submit an **action**:
```json
{
  "acceleration": 2.5,    // m/s² (clamped to [-max_accel, +max_accel])
  "stance": "extended"    // one of: "neutral", "extended", "retracted", "defending"
}
```

**No discrete "moves"** — fighters control continuous acceleration + discrete body state.

### Snapshot (Arena → Fighter)

Each tick, fighters receive a **snapshot** filtered through their sensors:

```json
{
  "protocol_version": "proto_1d_poc_v1",
  "tick": 42,
  "time_remaining_ticks": 558,

  "you": {
    "position": 4.2,           // exact (always know your own position)
    "velocity": 1.5,           // exact
    "hp": 85.0,                // exact
    "stamina": 6.3,            // exact
    "stance": "extended"       // exact
  },

  "opponent": {
    "distance": 3.0,           // bucketed by distance_precision from fighter spec
    "velocity": -0.6,          // bucketed by velocity_precision
    "stance_hint": "defending", // based on stance_detection quality
    "hp_visible": false        // opponent HP not visible in POC
  },

  "arena": {
    "width": 10.0,
    "your_distance_to_left_wall": 4.2,
    "your_distance_to_right_wall": 5.8
  }
}
```

**Sensor Quality Impact** (from Fighter Spec):
- `distance_precision: 0.5` → opponent at 6.7m shown as 6.5m or 7.0m
- `stance_detection: "fuzzy"` → `extended`/`retracted` shown as `"attacking"`, `defending` shown as `"defending"`, `neutral` as `"neutral"`
- `stance_detection: "none"` → no stance information at all

### Timing Rules (POC)
- **Tick rate:** 15 ticks per second (one tick = 67ms simulation time)
- **Response budget:** 50ms wall-clock time to return action
- **Late response:** Treated as `{"acceleration": 0, "stance": "neutral"}` (forced idle)
- **Invalid action:** Rejected, replaced with safe default (0 accel, previous stance)

### Constraints & Validation

The Arena validates that actions are legal:

| Constraint | Rule | If Violated |
|-----------|------|-------------|
| **Acceleration range** | Must be in `[-max_accel, +max_accel]` | Clamped to limit |
| **Stance value** | Must be one of 4 valid stances | Forced to "neutral" |
| **Stamina** | Must have stamina ≥ 0 | If stamina hits 0, forced to neutral stance, accel = 0 |
| **Response time** | Must respond within 50ms | Use default safe action |

---

## Example POC Tick Exchange

### Arena → Fighter A
```json
{
  "protocol_version": "proto_1d_poc_v1",
  "tick": 10,
  "time_remaining_ticks": 590,
  "you": {
    "position": 3.5,
    "velocity": 0.8,
    "hp": 100.0,
    "stamina": 9.2,
    "stance": "neutral"
  },
  "opponent": {
    "distance": 4.5,        // real: 4.3, bucketed to 4.5
    "velocity": -0.5,       // real: -0.4, bucketed to -0.5
    "stance_hint": "neutral"
  },
  "arena": {
    "width": 10.0,
    "your_distance_to_left_wall": 3.5,
    "your_distance_to_right_wall": 6.5
  }
}
```

### Fighter A → Arena
```json
{
  "tick": 10,
  "acceleration": 3.0,
  "stance": "extended"
}
```

### After Physics Step
Arena applies both fighters' actions simultaneously, then generates tick 11 snapshots.

---

## Replay & Audit
- Every tick's snapshot and chosen action are written to the **official replay**
- Anyone can re-run the fight from the replay under the same protocol version to get identical results
- Violations (late responses, invalid actions) are recorded in telemetry

---

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | Full Vision (2D/3D) |
|--------|----------|---------------------|
| **Control Inputs** | acceleration (1D) + stance (4 options) | Forces/positions for each body part |
| **Snapshot** | Position, velocity, distance to opponent | Visual fields, tactile, proprioception, balance |
| **Opponent Info** | Distance, velocity, stance hint | Visual observation (what you can "see") |
| **Timing** | 15 Hz, 50ms budget | Variable by division, potentially faster |
| **Validation** | Simple range checks | Joint limits, force limits, balance constraints |

**What stays constant:**
- Deterministic, replayable protocol
- Sensor-based perception (limited information)
- Time-bounded decision making
- Fair and identical for all fighters in a division

---

## Boundaries (what the Protocol is **not**)
- **Not training:** No learning or tuning happens here — only match-time decisions
- **Not visuals:** Cameras and effects are separate; they never change results
- **Not model-specific:** The protocol doesn't care if a fighter is rules-based, RL, or LLM — only that it speaks the contract

---

## Versioning & Divisions
- Each **division** pins to a **protocol version** (e.g., `proto_1d_poc_v1`)
- New versions may add richer snapshots or controls for higher divisions
- Older divisions remain stable to keep the entry ladder open
- Changes between versions are documented with migration guides

---

## Success Criteria
- Clear, public **control schema** and **snapshot fields** per division
- Enforced **time budget** and **input validation** every tick
- Fully **replayable** results with compact logging
- Simple to implement by any team, in any language

---

**One-liner summary:**
*The Combat Protocol is the shared, fair contract between fighters and arena — starting with simple acceleration + stance in the POC, evolving toward full body control while maintaining determinism and fairness.*
