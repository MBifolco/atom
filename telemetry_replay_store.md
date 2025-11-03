
# 🧾 Atom Combat — Telemetry & Replay Store (Component)

## Purpose
The **Telemetry & Replay Store** is the official record keeper of Atom Combat.  
It captures every tick, move, and event that happens during a match — creating an immutable timeline of truth.

> If it’s not in the replay, it didn’t happen.

---

## Responsibilities
- Collect state data from the Arena and Orchestrator each tick.  
- Log fighter actions, events, positions, and outcomes.  
- Serialize replays in a deterministic, replayable format (e.g., JSON or binary).  
- Store metadata for indexing and retrieval (match ID, fighters, protocol version).  
- Expose APIs for replay reading, evaluation, and rendering.  

---

## Data Captured per Tick
| Field | Description |
|--------|--------------|
| `tick` | Simulation step number |
| `fighter_actions` | Moves chosen by each fighter |
| `positions` | Simplified coordinates of fighters |
| `events` | Hits, blocks, knockouts, etc. |
| `state_summary` | HP, stamina, and timers |

---

## Replay File Structure
```json
{
  "match_id": "abc123",
  "protocol": "proto_v1",
  "seed": 42,
  "fighters": ["FighterA", "FighterB"],
  "ticks": [
    {
      "tick": 1,
      "actions": {"A":"step_forward","B":"idle"},
      "events": [],
      "states": {"A":{"hp":100},"B":{"hp":100}}
    },
    ...
  ],
  "result": {"winner":"B","reason":"timeout"}
}
```

---

## Guarantees
- **Deterministic:** Every replay can be re-simulated exactly.  
- **Immutable:** Replays are never altered after creation.  
- **Auditable:** Full transparency for judges, players, and spectators.

---

**One-liner summary:**  
*Telemetry and Replay Store preserve the entire fight’s history — the single source of truth for what actually happened.*
