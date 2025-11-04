# ⚔️ Atom Combat

**AI-controlled fighters battle in a physics-based arena.**

A competitive platform where you train AI fighters, not control them directly. Each fighter makes split-second decisions based on what it perceives, creating a new kind of sport where intelligence meets combat.

---

## 🚀 Quick Start

### Run a Fight

```bash
# Basic fight
python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py

# With HTML replay
python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py --html replay.html

# Watch in terminal
python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py --watch

# Custom configuration
python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py \
    --mass-a 65 --mass-b 80 \
    --html replay.html \
    --save telemetry.json.gz
```

### Create Your Own Fighter

Create a Python file with a `decide` function:

```python
# my_fighter.py
def decide(snapshot):
    """
    Make a decision based on current game state.

    Args:
        snapshot: Current state visible to your fighter
            {
                "tick": int,
                "you": {"position": float, "velocity": float, "hp": float, "stamina": float, "stance": str},
                "opponent": {"distance": float, "velocity": float, "hp": float, "stamina": float},
                "arena": {"width": float}
            }

    Returns:
        {"acceleration": float, "stance": str}
            - acceleration: -4.5 to +4.5 m/s²
            - stance: "neutral", "extended", "retracted", or "defending"
    """
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    # Your strategy here
    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}
    elif stamina > 3.0:
        return {"acceleration": 4.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}
```

Then run it:
```bash
python atom_fight.py my_fighter.py fighters/examples/tank.py --html my_first_fight.html
```

---

## 🤖 Train AI Fighters

Want to create AI fighters that learn through reinforcement learning? Check out the `training/` directory!

**📖 [How Training Works](docs/HOW_TRAINING_WORKS.md)** - Learn about AI training with simple explanations and diagrams

```bash
cd training
pip install -r requirements.txt
python train_fighter.py --opponent ../fighters/examples/tank.py --output my_ai --episodes 5000 --create-wrapper
```

**What you get:**
- PPO-based reinforcement learning
- Multi-core parallel training
- Auto-stopping when performance plateaus
- ONNX export for portability
- Standalone wrapper files

**See [training/README.md](training/README.md) for setup and commands.**

---

## 📖 Example Fighters

See **[fighters/README.md](fighters/README.md)** for complete fighter guide and testing commands.

**Rusher** (`fighters/examples/rusher.py`) - Aggressive pressure fighter
- Constantly advances
- Strikes when close
- Backs away from walls
- Retreats when HP critical

**Tank** (`fighters/examples/tank.py`) - Defensive counter-puncher
- Maintains optimal distance (2-4m)
- Defends when charged
- Counter-attacks on openings
- Strategic positioning

**Balanced** (`fighters/examples/balanced.py`) - Adaptive tactician
- Aggressive when winning
- Defensive when losing
- Smart stamina management
- Adapts to situation

---

## 🎮 CLI Options

### Fighter Configuration
```bash
--name-a NAME           # Name for fighter A (default: filename)
--name-b NAME           # Name for fighter B (default: filename)
--mass-a MASS           # Mass in kg (default: 70)
--mass-b MASS           # Mass in kg (default: 75)
--pos-a POS             # Starting position (default: 2.0)
--pos-b POS             # Starting position (default: 10.0)
```

### Match Settings
```bash
--max-ticks N           # Maximum ticks before timeout (default: 1000)
--seed N                # Random seed for reproducibility (default: 42)
```

### Output Options
```bash
--watch                 # Show ASCII animation in terminal
--html FILE             # Generate HTML replay file
--save FILE             # Save telemetry (.json or .json.gz)
--speed SPEED           # Playback speed for --watch (default: 5.0)
```

---

## 🏗️ Project Structure

```
atom/
├── atom_fight.py              # CLI fight runner (START HERE!)
├── fighters/                   # Fighter collection
│   ├── examples/              # Well-crafted example fighters
│   │   ├── rusher.py         # Aggressive pressure fighter
│   │   ├── tank.py           # Defensive counter-puncher
│   │   └── balanced.py       # Adaptive tactician
│   ├── training_opponents/    # Curriculum for training AI
│   │   ├── training_dummy.py # Level 1: Stationary target
│   │   ├── wanderer.py       # Level 2: Random movement
│   │   ├── bumbler.py        # Level 3: Poor execution
│   │   └── novice.py         # Level 4: Basic competence
│   └── README.md             # Fighter guide
├── training/                   # AI fighter training (RL)
│   ├── train_fighter.py       # Training CLI
│   ├── requirements.txt       # Training dependencies
│   ├── README.md              # Training guide
│   └── src/                   # Training infrastructure
│       ├── gym_env.py         # Gymnasium wrapper
│       ├── trainer.py         # PPO trainer
│       └── onnx_fighter.py    # ONNX export/inference
├── docs/                       # Documentation
│   ├── original_vision/       # Design philosophy & specs
│   ├── HOW_TRAINING_WORKS.md # AI training guide (simple!)
│   ├── VISION_GAP_ANALYSIS.md # Built vs planned features
│   └── IMPROVEMENTS.md        # Stamina & AI improvements
├── src/                        # Core components
│   ├── arena/                 # Physics engine
│   ├── protocol/              # Combat contract
│   ├── orchestrator/          # Match coordinator
│   ├── telemetry/             # Replay storage
│   ├── evaluator/             # Spectacle scoring
│   ├── renderer/              # ASCII + HTML5 visualization
│   └── ai/                    # Tactical AI examples
├── poc/                        # Original proof-of-concept
└── tests/                      # Component tests
```

---

## 🎯 Game Mechanics

### Physics
- **1D movement** along a line arena
- **Continuous physics**: velocity, acceleration, friction
- **Collision-based combat**: no predefined attacks, damage from physics
- **Mass matters**: heavier = more damage, higher stamina cost

### Stamina Economy
- **Movement costs stamina** (acceleration * mass)
- **Stances cost stamina** (extended/defending drain energy)
- **Neutral stance regenerates** faster
- **Exhaustion penalty**: forced to neutral at 0 stamina

### Stances
- **Neutral** ● - Balanced, best regen, moderate reach
- **Extended** ▶ - Long reach, high drain, low defense
- **Retracted** ◀ - Short reach, low drain, good defense
- **Defending** ■ - Moderate reach, high drain, best defense

### Mass Tradeoffs
- **Light fighters** (50-65kg): Efficient, high stamina, less damage
- **Medium fighters** (65-80kg): Balanced
- **Heavy fighters** (80-95kg): Powerful hits, less stamina, slower

---

## 📊 Spectacle Scoring

Matches are evaluated on entertainment value (not just who wins):

**Metrics:**
1. **Duration** - Ideal length (100-400 ticks)
2. **Close Finish** - How tight was the ending
3. **Stamina Drama** - Exhaustion moments
4. **Comeback Potential** - HP lead changes
5. **Positional Exchange** - Movement variety
6. **Pacing Variety** - Speed variance
7. **Collision Drama** - Impactful exchanges

**Ratings:**
- 0.8+ = EXCELLENT ⭐⭐⭐⭐⭐
- 0.6+ = GOOD ⭐⭐⭐⭐
- 0.4+ = FAIR ⭐⭐⭐
- <0.4 = POOR ⭐⭐

---

## 🧪 Testing & Development

```bash
# Run component tests
python test_arena_component.py
python test_orchestrator.py
python test_evaluator.py
python test_improved_combat.py

# Generate test replays
python generate_html_replay.py

# Run parameter search (advanced)
python poc/param_search.py
```

---

## 📚 Documentation

- **[Fighter Guide](fighters/README.md)** - Fighter collection & testing commands
- **[Training Guide](training/README.md)** - Train AI fighters with reinforcement learning
- **[How Training Works](docs/HOW_TRAINING_WORKS.md)** - AI training explained simply with diagrams
- **[Vision Documents](docs/original_vision/)** - Original design philosophy
- **[Vision Gap Analysis](docs/VISION_GAP_ANALYSIS.md)** - What's built vs planned
- **[Improvements](docs/IMPROVEMENTS.md)** - Stamina & AI improvements

---

## 🎨 Features

✅ **Physics-based combat** - Continuous motion with discrete stances
✅ **Stamina economy** - Real resource management
✅ **AI training system** - Reinforcement learning with PPO
✅ **Config-driven** - No hardcoded constants
✅ **Replay system** - Save/load compressed matches
✅ **Spectacle evaluation** - 7-metric quality scoring
✅ **HTML5 animation** - Beautiful standalone replays
✅ **Simple CLI** - Run fights with one command!

---

## 🚧 Roadmap

See [VISION_GAP_ANALYSIS.md](VISION_GAP_ANALYSIS.md) for full gap analysis.

**Next Priorities:**
- [ ] Fighter spec & artifact system (metadata, versioning)
- [ ] Sensor precision (distance/velocity bucketing)
- [ ] Registry (fighter catalog, certification)
- [ ] Governance (time limits, sandboxing)
- [ ] Web UI (tournament organization)

---

## 🤝 Contributing

This is a vision-driven project. See `docs/original_vision/concept.md` for the big picture.

**Philosophy:**
- AI fights AI, humans train and coach
- Common protocol ensures fairness
- Entertainment value matters (spectacle scoring)
- Physics-based, not move-based combat
- Designed to scale from hobby to pro leagues

---

## 📄 License

MIT (for now - TBD based on project direction)

---

**Built with:**
- Python 3.x
- Pure component architecture
- Zero ML dependencies for core system
- Optional RL training (Stable-Baselines3, PyTorch, ONNX)
- Config-driven physics
- Deterministic replay system

---

**Quick Links:**
- 🎮 Run a fight: `python atom_fight.py fighters/examples/rusher.py fighters/examples/tank.py --html replay.html`
- 📖 Create a fighter: Copy `fighters/examples/rusher.py` and modify the `decide` function
- 🎬 Watch replays: Open generated `.html` files in browser
- 📊 View scores: Matches automatically show spectacle ratings

**Welcome to Atom Combat!** ⚔️
