
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

## Output Example (text)
```
[00.0] A steps forward
[00.1] B charges heavy attack
[00.3] A hit by heavy! (-10 HP)
[01.0] Timeout → Winner: B
```

---

**One-liner summary:**  
*The Replay Renderer brings the fight to life — transforming deterministic logs into a human experience without changing the truth.*
