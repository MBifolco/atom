
# ⚖️ Atom Combat — Evaluator (Component)

## Purpose
The **Evaluator** is the impartial judge that turns raw replays into meaningful results.  
It decides winners, scores, and style metrics that drive league rankings.

> The Evaluator translates performance into recognition.

---

## Responsibilities
- Analyze replays and extract key statistics.  
- Apply scoring logic based on division rules (HP left, aggression, defense, etc.).  
- Validate legality of moves and detect rule violations.  
- Output standardized score reports.  

---

## Score Report Example
```json
{
  "match_id": "abc123",
  "winner": "FighterB",
  "method": "timeout",
  "hp_remaining": {"A":50,"B":70},
  "metrics": {
    "aggression": {"A":0.65,"B":0.72},
    "defense": {"A":0.4,"B":0.55},
    "illegal_moves": 0
  }
}
```

---

## Metrics Categories
| Type | Description |
|-------|--------------|
| **Performance** | Win/loss, KO, time remaining. |
| **Style** | Aggression, defense, action diversity. |
| **Compliance** | Violations, lateness, illegal moves. |

---

## Output
Evaluators generate results for:
- **Match results:** official outcomes.  
- **League rankings:** aggregated stats per season.  
- **Training insights:** data for fighter refinement.

---

**One-liner summary:**  
*The Evaluator turns the replay’s raw data into fair, interpretable results — quantifying both victory and style.*
