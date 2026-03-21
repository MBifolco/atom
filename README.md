# ⚔️ Atom Combat

**AI-controlled fighters battle in a physics-based arena.**

A competitive platform where you train AI fighters, not control them directly. Each fighter makes split-second decisions based on what it perceives, creating a new kind of sport where intelligence meets combat.

---

## 🚀 Quick Start

### Run a Fight

```bash
# Basic fight
python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py

# With HTML replay
python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --html replay.html

# Watch in terminal
python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --watch

# Custom configuration
python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py \
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
            - stance: "neutral", "extended", or "defending"
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
python atom_fight.py my_fighter.py fighters/examples/boxer.py --html my_first_fight.html
```

---

## 🎯 Complete Workflow: From Zero to Fighting

### Step 1: Train Your Fighter

Train an AI fighter using the progressive training system:

```bash
# Full training pipeline (curriculum + population evolution)
python train_progressive.py --mode complete --cores 8

# This takes 6-12 hours and produces:
# - Curriculum graduate (skilled in fundamentals through expert level)
# - Population of evolved fighters with diverse strategies
# - Auto-exported fighters in fighters/AIs/
```

### Step 2: Build Fighter Registry

Create a registry of all available fighters:

```bash
# Scan all fighters and create registry.json
python scripts/training/build_registry.py

# This indexes:
# - Example fighters (fighters/examples/)
# - Test dummies (fighters/test_dummies/)
# - Your trained AIs (fighters/AIs/)
```

### Step 3: Test Your Fighter (CLI)

Quick command-line testing:

```bash
# Fight your AI against an example fighter
python atom_fight.py \
    fighters/AIs/your_fighter_name/fighter.py \
    fighters/examples/boxer.py \
    --html my_fight.html

# Open my_fight.html in browser to watch replay
```

### Step 4: Use the Web App

Launch the web interface for easier fighter selection and match visualization:

```bash
# Install web dependencies (one-time)
cd web
pip install -r requirements.txt
cd ..

# Start the web server
uvicorn apps.web.app:app --reload  # preferred
# Legacy compatibility path: uvicorn web.app:app --reload

# Open http://localhost:8000 in your browser
```

In the web app you can:
- Browse all fighters with stats and descriptions
- Select any two fighters to battle
- Configure match parameters (mass, seed, etc.)
- Watch live animated replays
- Export standalone HTML replays
- View spectacle scores and match statistics

**See [web/README.md](web/README.md) for complete web app documentation.**

---

## 🤖 Train AI Fighters

Create AI fighters using our **Progressive Training System** that combines curriculum learning with population-based evolution!

**📖 [Progressive Training Guide](docs/PROGRESSIVE_TRAINING.md)** - Complete training system documentation
**☁️ [Colab Setup Guide](docs/COLAB_SETUP.md)** - Persistent Drive cache + GPU training workflow
**✅ [Colab Validation Checklist](docs/COLAB_VALIDATION_CHECKLIST.md)** - Milestone gate checks for Phases 1/3/5
**📓 [Colab Notebook](notebooks/Atom_Training_Colab.ipynb)** - Run setup/train/resume directly in Colab

```bash
# Quick test run (~5 minutes)
python train_progressive.py --mode quick

# Full training pipeline (CPU)
python train_progressive.py --mode complete

# GPU-accelerated training (77x faster!)
python train_progressive.py --mode complete --use-vmap
```

**What you get:**
- **Curriculum Learning**: 5 progressive difficulty levels with 29 test dummies
- **Hardcore Graduation**: Maintain 80-88% win rates throughout training
- **Population Training**: Self-play evolution for diverse strategies (8 fighters in parallel)
- **GPU Acceleration**: 77x speedup with JAX vmap (250 parallel environments)
- **PPO/SAC algorithms**: Industry-standard reinforcement learning
- **Automatic export**: Fighters ready to use with `atom_fight.py`
- **Complete logs**: Track training progress in real-time

**Key Features:**
- ✅ Systematic skill progression (fundamentals → expert)
- ✅ Dual-requirement graduation (recent + overall win rates)
- ✅ 29 specialized test dummies across 5 curriculum levels
- ✅ Population diversity through evolution
- ✅ Battle-ready champions exported automatically
- ✅ GPU support (AMD ROCm + NVIDIA CUDA)
- ✅ Multicore parallel training support
- ✅ Detailed progress logging and reward tracking

---

## 📖 Example Fighters

See **[fighters/README.md](fighters/README.md)** for complete fighter guide and testing commands.

**Boxer** (`fighters/examples/boxer.py`) - Pressure-focused baseline
- Prioritizes close-range exchanges
- Uses stance transitions to stay active

**Counter Puncher** (`fighters/examples/counter_puncher.py`) - Reactive punisher
- Waits for openings and punishes overextension
- Focuses on timing over constant aggression

**Swarmer** (`fighters/examples/swarmer.py`) - High-tempo pressure style
- Maintains forward momentum
- Tries to overwhelm with sustained engagement

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
├── atom_fight.py               # Compatibility wrapper for the fight CLI
├── train_progressive.py        # Compatibility wrapper for progressive training
├── colab_bootstrap.sh          # Compatibility wrapper for the Colab bootstrap
├── apps/                       # User-facing application entrypoints
│   ├── cli/                   # Fight CLI implementation
│   ├── training/            # Training CLI implementation
│   └── web/                 # FastAPI web app implementation
├── fighters/                   # Fighter collection
│   ├── examples/              # Hardcoded example fighters
│   │   ├── boxer.py          # Pressure-focused baseline
│   │   ├── counter_puncher.py# Reactive punisher
│   │   ├── slugger.py        # Power-focused style
│   │   └── ...               # Additional built-in styles
│   ├── test_dummies/          # Training curriculum opponents
│   │   └── atomic/           # Atomic behavior fighters used by curriculum levels
│   ├── AIs/                   # Trained AI fighters (auto-exported)
│   └── README.md             # Fighter guide
├── src/                        # Core packages and platform modules
│   ├── atom/                  # Transitional Atom package namespace
│   │   ├── runtime/          # Runtime core namespace
│   │   ├── training/         # Training namespace
│   │   ├── registry/         # Platform registry namespace
│   │   └── coaching/         # Coaching namespace
│   ├── arena/                 # Legacy compatibility wrappers for runtime arena
│   ├── protocol/              # Legacy compatibility wrappers for the combat contract
│   ├── orchestrator/          # Legacy compatibility wrappers for match orchestration
│   ├── telemetry/             # Legacy compatibility wrappers for replay storage
│   ├── evaluator/             # Legacy compatibility wrappers for spectacle scoring
│   ├── renderer/              # Legacy compatibility wrappers for ASCII + HTML5 visualization
│   ├── registry/              # Legacy compatibility wrappers for fighter registry
│   ├── coaching/              # Legacy compatibility wrappers for coaching utilities
│   └── training/              # Training infrastructure and compatibility wrappers
│       ├── gym_env.py         # Gymnasium environment wrapper
│       ├── trainers/          # Training algorithms
│       │   ├── curriculum_trainer.py  # Curriculum learning
│       │   └── population/    # Population-based training
│       └── onnx_fighter.py    # ONNX export/inference
├── docs/                       # Documentation
│   ├── PROGRESSIVE_TRAINING.md  # Training system guide
│   ├── POPULATION_TRAINING.md   # Population evolution details
│   ├── original_vision/         # Design philosophy & specs
│   └── VISION_GAP_ANALYSIS.md   # Historical built-vs-planned snapshot
├── scripts/                    # Utility scripts
│   ├── analysis/              # Training/run analysis helpers
│   ├── training/              # Registry, baseline, resume helpers
│   ├── montage/               # Replay rendering and montage generation
│   ├── colab/                 # Colab bootstrap implementation
│   └── ops/                   # Operational helpers
├── archived/                   # Historical diagnostics and legacy assets
├── outputs/                    # Training outputs (logs, models, fighters)
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
# Run the full automated test suite
make test

# Fast unit-only run
make test-unit

# Integration slice
make test-integration
```

---

## 📚 Documentation

- **[Progressive Training Guide](docs/PROGRESSIVE_TRAINING.md)** - Complete training system documentation
- **[NaN Investigation](docs/NAN_ERROR_INVESTIGATION.md)** - Reward scaling and NaN mitigation details
- **[Repo Reorganization Plan](docs/REPO_REORGANIZATION_PLAN.md)** - Proposed layout for runtime, training, platform, and app boundaries
- **[Namespace Migration Policy](docs/NAMESPACE_MIGRATION_POLICY.md)** - Rules for when legacy wrappers stay, move, or retire
- **[Fighter Guide](fighters/README.md)** - Fighter collection & testing commands
- **[Test Dummies](fighters/test_dummies/README.md)** - Training curriculum opponents
- **[Vision Documents](docs/original_vision/)** - Original design philosophy
- **[Vision Gap Analysis](docs/VISION_GAP_ANALYSIS.md)** - Historical gap snapshot (Nov 2025)

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

See [docs/future_implementation/](docs/future_implementation/) for active roadmap docs.

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
- 🎮 Run a fight: `python atom_fight.py fighters/examples/boxer.py fighters/examples/slugger.py --html replay.html`
- 📖 Create a fighter: Copy `fighters/examples/boxer.py` and modify the `decide` function
- 🎬 Watch replays: Open generated `.html` files in browser
- 📊 View scores: Matches automatically show spectacle ratings

**Welcome to Atom Combat!** ⚔️
