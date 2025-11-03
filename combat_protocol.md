
# 🤝 Atom Combat — Combat Protocol (Component)

## Purpose
The **Combat Protocol** is the shared **language** between a fighter and the arena during a sanctioned match.  
It defines **what fighters can sense**, **what they are allowed to do**, and **how quickly** they must respond.  
This contract is identical for everyone in a given division, which is what makes the sport fair.

> One world, one rulebook, one interface — no matter how a fighter was built or trained.

---

## What the Protocol Guarantees
- **Fairness:** Every fighter receives the same kind of limited snapshot and the same list of legal moves.
- **Reproducibility:** Given the same seeds and the same move choices, the fight plays out exactly the same.
- **Timing discipline:** Fighters must return one legal move **within the per-tick time limit**.
- **Compatibility:** Different training methods/models can compete as long as they speak this protocol.

---

## Responsibilities (at a glance)
1. **Sense:** Define the snapshot of the world a fighter is allowed to “see” each tick (simple, limited, slightly fuzzy).
2. **Act:** Define the allowed move list and when a move is currently legal/illegal.
3. **Pace:** Define the decision rhythm (ticks per second) and the response time budget.
4. **Record:** Provide a stable, replayable record of snapshots, moves, and outcomes.
5. **Version:** Evolve carefully — divisions pin to a specific protocol version.

---

## Inputs & Outputs (simple view)

**From Arena → Fighter (every tick):**
- A **snapshot** (limited view): distance buckets, health/stamina buckets, timers, simple flags (e.g., “opponent likely in recovery”).
- A **legality mask:** which moves are currently allowed (cooldowns, stamina, position).
- A **tick marker:** which moment this snapshot refers to.

**From Fighter → Arena (every tick):**
- **One move** chosen from the legal list, returned before the time limit.

---

## What’s in the Snapshot (examples)
*(Exact fields are set by the division. These are illustrative.)*

- **Positioning:** rough distance band between fighters (e.g., 0–10), corner/center flag.
- **Vitals:** your health band, your stamina band; opponent health band (coarse).
- **Timing:** time left in round; your cooldown bands for key moves.
- **State hints:** “opponent likely recovering,” “you are guarding,” etc.
- **Environment:** simple zone or hazard flags (if the arena has them).

> The snapshot is **intentionally limited and discretized** so perfect play is impossible and strategy matters.

---

## The Move List (examples)
*(Division-defined, public, and identical for all fighters.)*

- `idle` — do nothing this tick  
- `step_forward`, `step_back`  
- `dash` (costs stamina; short cooldown)  
- `block` (reduces damage; has commitment)  
- `light_attack` (fast, low damage)  
- `heavy_attack` (slow, high damage)

Each move has public **costs** (stamina), **timing** (startup/active/recovery), **reach**, **damage**, and **cooldown**.  
The protocol marks a move **illegal** when its conditions aren’t met (e.g., on cooldown, not enough stamina). Fighters must pick from the legal set.

---

## Timing Rules
- **Tick rate:** e.g., 15 ticks per second (division-specific).
- **Response budget:** e.g., ≤ 20 ms to return a move (division-specific).
- **Late or invalid moves:** become `idle` (or are dropped) and count as a violation in the log.

---

## Example Tick (human-readable JSON)
```json
// Arena → Fighter
{
  "version": "proto_v1",
  "tick": 42,
  "snapshot": {
    "dist_band": 6,
    "you":   {"hp_band": 18, "stam_band": 7, "cooldowns": {"light":2,"heavy":5,"dash":1}},
    "opp":   {"hp_band": 15, "stance_hint": "recovering"},
    "timer_band": 23,
    "zone":  "center"
  },
  "legal_moves": ["idle","step_forward","step_back","block","light_attack"]
}

// Fighter → Arena
{ "tick": 42, "move": "light_attack" }
```

---

## Replay & Audit
- Every tick’s snapshot, legality set, and chosen move are written to the **official replay**.
- Anyone can re-run the fight from the replay under the same protocol version to get the exact same outcome.
- Violations (late responses, illegal attempts) are recorded.

---

## Boundaries (what the Protocol is **not**)
- **Not training:** no learning or tuning happens here — only match-time decisions.
- **Not visuals:** cameras and effects are separate; they never change results.
- **Not model-specific:** the protocol doesn’t care if a fighter is rules-based or uses advanced AI — only that it speaks the contract.

---

## Versioning & Divisions
- Each **division** pins to a **protocol version** (e.g., `proto_v1`).
- New versions may add richer snapshots or moves for higher divisions while preserving fairness within each division.
- Older divisions remain stable to keep the entry ladder open.

---

## Success Criteria (for this component)
- Clear, public **move list** and **snapshot fields** per division.
- Enforced **time budget** and **move legality** every tick.
- Fully **replayable** results with a compact log.
- Simple to implement by any team, in any language.

---

**One-liner summary:**  
*The Combat Protocol is the shared, fair, and replayable conversation between a fighter and the arena — a tiny contract that lets any kind of “mind” compete on equal terms.*
