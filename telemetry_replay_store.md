
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

## Replay File Structure (POC Example)
```json
{
  "match_id": "poc_001",
  "protocol": "proto_1d_poc_v1",
  "world_spec": "world_1d_poc_v1",
  "seed": 42,
  "fighters": {
    "A": {"id": "aggressive_bot", "spec": {...}},
    "B": {"id": "defensive_bot", "spec": {...}}
  },
  "ticks": [
    {
      "tick": 1,
      "actions": {
        "A": {"acceleration": 5.0, "stance": "neutral"},
        "B": {"acceleration": 0.0, "stance": "neutral"}
      },
      "events": [],
      "states": {
        "A": {"position": 2.17, "velocity": 0.34, "hp": 100, "stamina": 9.97, "stance": "neutral"},
        "B": {"position": 8.0, "velocity": 0.0, "hp": 100, "stamina": 10.0, "stance": "neutral"}
      }
    },
    {
      "tick": 2,
      "actions": {
        "A": {"acceleration": 5.0, "stance": "neutral"},
        "B": {"acceleration": 2.0, "stance": "neutral"}
      },
      "events": [],
      "states": {
        "A": {"position": 2.57, "velocity": 0.61, "hp": 100, "stamina": 9.94},
        "B": {"position": 7.93, "velocity": 0.13, "hp": 100, "stamina": 9.99}
      }
    },
    {
      "tick": 15,
      "actions": {
        "A": {"acceleration": 3.0, "stance": "extended"},
        "B": {"acceleration": 0.0, "stance": "defending"}
      },
      "events": [
        {
          "type": "COLLISION",
          "fighters": ["A", "B"],
          "damage": {"A": 8.2, "B": 12.5},
          "relative_velocity": 1.8
        }
      ],
      "states": {
        "A": {"position": 5.8, "velocity": 1.2, "hp": 91.8, "stamina": 8.1, "stance": "extended"},
        "B": {"position": 6.1, "velocity": -0.6, "hp": 87.5, "stamina": 9.7, "stance": "defending"}
      }
    }
  ],
  "result": {
    "winner": "A",
    "reason": "KO",
    "tick": 387,
    "final_hp": {"A": 45.2, "B": 0.0}
  }
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
