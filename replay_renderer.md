
# 🎥 Atom Combat — Replay Renderer (Component)

## Purpose
The **Replay Renderer** converts raw replay logs into cinematic, watchable experiences.  
It never affects gameplay outcomes — it simply replays what really happened.

> The Renderer turns truth into spectacle.

---

## Responsibilities
- Load replay logs from Telemetry Store.  
- Simulate motion and timing in sync with Arena rules.  
- Produce visual or text-based playback (video, web, terminal).  
- Overlay events: hits, KOs, stamina bars, and commentary.  

---

## Rendering Modes
| Mode | Description |
|-------|--------------|
| **Text Summary** | Minimal terminal-based replay (for POC). |
| **2D Visualizer** | Simple top-down or side-view representation. |
| **Cinematic Renderer** | Full 3D replay with cameras and effects (future). |

---

## Output Example (text - POC format)
```
═══════════════════════════════════════════════════
    ATOM COMBAT - Match poc_001
    aggressive_bot (A) vs defensive_bot (B)
═══════════════════════════════════════════════════

[T:1  | 0.07s] A [██------] accelerates forward (5.0 m/s²)
                B [------██] holds position
                Distance: 5.8m

[T:5  | 0.33s] A [███-----] rushes forward (v=1.2 m/s)
                B [-----███] advances slowly (2.0 m/s²)
                Distance: 3.2m

[T:12 | 0.80s] A [████----] extends stance, closing in!
                B [----████] takes defensive stance
                Distance: 1.5m

[T:15 | 1.00s] 💥 COLLISION!
                A [████====] crashes into B (rel. vel: 1.8 m/s)
                Damage: A -8.2 HP, B -12.5 HP (defending reduced impact)
                A: 91.8 HP | B: 87.5 HP

[T:45 | 3.00s] A [███-----] stamina recovering (neutral stance)
                B [-----███] counter-extends!
                Distance: 2.1m

...

[T:387 | 25.8s] 💀 KNOCKOUT!
                B's HP reaches 0
                Winner: aggressive_bot (A)
                Final: A 45.2 HP vs B 0.0 HP
═══════════════════════════════════════════════════
```

---

**One-liner summary:**  
*The Replay Renderer brings the fight to life — transforming deterministic logs into a human experience without changing the truth.*
