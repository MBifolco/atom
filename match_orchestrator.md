
# 🎛️ Atom Combat — Match Orchestrator (Component)

## Purpose
The **Match Orchestrator** is the director of every fight.  
It coordinates the flow between the Arena and both fighters, advancing time tick by tick, enforcing fairness and pacing.

> The Orchestrator is the heartbeat of the match.

---

## Responsibilities
- Initialize the match (load fighters, Arena, World Spec, Protocol).  
- Manage the tick loop (e.g., 15 ticks per second).  
- Send snapshots to fighters.  
- Receive and validate moves.  
- Advance the Arena’s physics state.  
- Record all actions into the Telemetry system.  
- Detect end conditions and declare results.

---

## Core Loop
1. Generate snapshot for each fighter.  
2. Wait for each to respond within time budget.  
3. Apply moves to the Arena.  
4. Log results and events.  
5. Repeat until win condition or time expires.  

---

## Determinism & Reproducibility
- Each match runs from a seed and config file (Match Spec).  
- The Orchestrator’s sequence defines the official truth of the fight.  
- Replays can re-run the exact same tick sequence with no variation.

---

## Example Lifecycle
```text
load fighters → init arena → tick → collect moves → apply physics → record → repeat → end
```

---

**One-liner summary:**  
*The Match Orchestrator runs the tick-by-tick heartbeat of Atom Combat, ensuring fairness, order, and perfect reproducibility.*
