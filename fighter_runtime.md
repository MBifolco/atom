
# 🧠 Atom Combat — Fighter Runtime (Component)

## Purpose
The **Fighter Runtime** is the “mind in the ring.”  
It’s the logic module that receives snapshots from the Arena and chooses one legal move each tick within a strict time limit.  

It doesn’t learn, predict, or change during a match — it simply acts, quickly and deterministically.

> The Fighter Runtime is where instinct lives — fast, bounded decision-making under pressure.

---

## Responsibilities
- Read and interpret the snapshot provided by the Arena.  
- Select one legal move from the current move list.  
- Return that move before the time budget expires.  
- Handle invalid or late responses gracefully (convert to `idle`).  
- Never alter game state directly — only act through legal moves.

---

## Behavior Contract
| Input | Output | Constraint |
|--------|----------|-------------|
| Snapshot (limited world view) | One move (from legal list) | Must return within allotted time (e.g., ≤ 20ms). |

---

## Example Runtime Loop (pseudo-code)
```python
def decide(snapshot, legal_moves, time_budget_ms):
    if snapshot["dist_band"] <= 2 and "heavy_attack" in legal_moves:
        return "heavy_attack"
    elif snapshot["dist_band"] <= 4 and "light_attack" in legal_moves:
        return "light_attack"
    elif "step_forward" in legal_moves:
        return "step_forward"
    else:
        return "idle"
```

---

## Determinism
- Every decision must be reproducible given the same snapshot and seed.  
- Randomness (if any) must use the fight’s global seed so replays remain consistent.  

---

## Compliance
A Fighter Runtime must:
- Implement the division’s Combat Protocol version.  
- Respect tick timing and legality masks.  
- Output a move format exactly as defined by the protocol.

---

**One-liner summary:**  
*The Fighter Runtime is the fighter’s decision core — a deterministic brain that acts within the rules, not above them.*
