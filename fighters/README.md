# Atom Combat Fighters

Collection of AI fighters for Atom Combat.

---

## Directory Structure

```
fighters/
├── examples/              # Well-crafted example fighters
│   ├── rusher.py         # Aggressive pressure fighter
│   ├── tank.py           # Defensive counter-puncher
│   └── balanced.py       # Adaptive tactician
│
└── training_opponents/    # Curriculum for training AI
    ├── training_dummy.py # Level 1: Stationary target
    ├── wanderer.py       # Level 2: Random movement
    ├── bumbler.py        # Level 3: Poor execution
    └── novice.py         # Level 4: Basic competence
```

---

## Example Fighters

### Rusher
**Strategy:** Aggressive pressure
- Constantly advances
- Strikes when close
- Backs away from walls
- Retreats when HP critical

### Tank
**Strategy:** Defensive counter-puncher
- Maintains optimal distance (2-4m)
- Defends when opponent charges
- Counter-attacks on openings
- Strategic positioning

### Balanced
**Strategy:** Adaptive tactician
- Aggressive when winning
- Defensive when losing
- Smart stamina management
- Adapts to situation

---

## Training Opponents

See [training_opponents/OPPONENT_PROGRESSION.md](training_opponents/OPPONENT_PROGRESSION.md) for the complete curriculum guide.

**Quick summary:**
1. **training_dummy** - Learn basic mechanics
2. **wanderer** - Learn positioning
3. **bumbler** - Learn timing
4. **novice** - Learn tactics
5. **rusher** - Learn counter-aggression
6. **tank** - Learn breaking defense
7. **balanced** - Learn adaptation

---

## Creating Your Own Fighter

Create a Python file with a `decide` function:

```python
def decide(snapshot):
    """
    Args:
        snapshot: {
            "tick": int,
            "you": {"position": float, "velocity": float, "hp": float, "stamina": float, "stance": str},
            "opponent": {"distance": float, "velocity": float, "hp": float, "stamina": float},
            "arena": {"width": float}
        }

    Returns:
        {"acceleration": float, "stance": str}
    """
    # Your strategy here
    pass
```

**See example fighters for inspiration!**

---

## Testing Fighters

```bash
# Test against an example
python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py --html replay.html

# Test your fighter
python atom_fight.py my_fighter.py fighters/examples/balanced.py --watch

# Test against training opponents
python atom_fight.py my_fighter.py fighters/training_opponents/novice.py
```
