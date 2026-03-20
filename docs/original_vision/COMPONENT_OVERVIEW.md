# Atom Combat - Component Architecture

> Legacy note: This file is a historical architecture snapshot.
> For the current high-level vision in this folder, start with `components.md`.

## Overview

The Atom Combat system has been refactored from the POC into a modular component architecture with config-driven design.

## Components

### 1. Protocol (`src/protocol/`)
**Purpose**: Defines the contract between fighters and the arena

**Key Files**:
- `combat_protocol.py` - Snapshot, Action dataclasses, ProtocolValidator, generate_snapshot()

**Responsibilities**:
- Define what fighters can perceive (Snapshot)
- Define what actions fighters can take (Action)
- Validate actions against protocol rules
- Generate snapshots for fighters

**Usage**:
```python
from src.protocol.combat_protocol import generate_snapshot, Action

snapshot = generate_snapshot(fighter_a, fighter_b, tick, arena_width)
action = Action(acceleration=4.0, stance="extended")
```

---

### 2. Arena (`src/arena/`)
**Purpose**: Source of truth for physics, collisions, and damage

**Key Files**:
- `world_config.py` - WorldConfig, StanceConfig (config dataclasses)
- `fighter.py` - FighterState with world-calculated stats
- `arena_1d.py` - Arena1D physics engine

**Responsibilities**:
- Manage world configuration (physics constants, damage formulas, stances)
- Calculate fighter stats from mass using world formulas
- Execute physics simulation (velocity, position, collisions)
- Calculate collision damage
- Update stamina based on actions

**Key Design**: Config-driven - no hardcoded constants!

**Usage**:
```python
from src.arena import WorldConfig, FighterState, Arena1D

config = WorldConfig()  # Load spectacle-optimized defaults
fighter_a = FighterState.create("Blitz", mass=70.0, position=2.0, world_config=config)
arena = Arena1D(fighter_a, fighter_b, config=config)
```

---

### 3. Match Orchestrator (`src/orchestrator/`)
**Purpose**: Coordinates the tick loop between fighters and arena

**Key Files**:
- `match_orchestrator.py` - MatchOrchestrator, MatchResult

**Responsibilities**:
- Run tick loop
- Generate snapshots for each fighter
- Call fighter decision functions
- Validate and clamp actions
- Pass actions to arena
- Record telemetry
- Handle timeouts and finish conditions
- Return match results

**Usage**:
```python
from src.orchestrator import MatchOrchestrator

orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)
result = orchestrator.run_match(
    fighter_a_spec={"name": "Blitz", "mass": 70.0, "position": 2.0},
    fighter_b_spec={"name": "Tank", "mass": 80.0, "position": 10.0},
    decision_func_a=aggressive_ai,
    decision_func_b=defensive_ai
)
```

---

### 4. Telemetry & Replay Store (`src/telemetry/`)
**Purpose**: Save and load match recordings

**Key Files**:
- `replay_store.py` - ReplayStore, save_replay(), load_replay()

**Responsibilities**:
- Save match telemetry to disk (JSON or compressed)
- Load replays from disk
- List available replays
- Extract replay metadata

**Features**:
- Gzip compression for smaller file sizes
- Metadata indexing
- Auto-generated filenames with timestamps

**Usage**:
```python
from src.telemetry import ReplayStore

store = ReplayStore(replay_dir="replays")
filepath = store.save(result.telemetry, result, metadata={"description": "Epic fight"})
replay_data = store.load("replay_20251103_164008_Blitz_vs_Tank.json.gz")
```

---

### 5. Evaluator (`src/evaluator/`)
**Purpose**: Analyze match quality and spectacle

**Key Files**:
- `spectacle_evaluator.py` - SpectacleEvaluator, SpectacleScore

**Responsibilities**:
- Calculate 7 spectacle metrics:
  1. **Duration** - Ideal match length (100-400 ticks)
  2. **Close Finish** - How tight was the ending
  3. **Stamina Drama** - Exhaustion moments (10-30% time at <30% stamina)
  4. **Comeback Potential** - HP lead changes
  5. **Positional Exchange** - Movement variety (5-20% position swaps)
  6. **Pacing Variety** - Speed variance
  7. **Collision Drama** - Impactful exchanges (8-25 collisions)
- Aggregate into overall quality score

**Philosophy**: WHO wins doesn't matter - only QUALITY of the fight matters

**Usage**:
```python
from src.evaluator import SpectacleEvaluator

evaluator = SpectacleEvaluator()
score = evaluator.evaluate(result.telemetry, result)
print(f"Overall spectacle: {score.overall:.3f}")
```

---

### 6. Replay Renderer (`src/renderer/`)
**Purpose**: Visualize matches from telemetry

**Key Files**:
- `ascii_renderer.py` - AsciiRenderer

**Responsibilities**:
- Render individual ticks as ASCII art
- Display arena with fighter positions and stances
- Show HP/stamina bars
- Highlight collisions
- Render match summary with spectacle score
- Play full replays with configurable speed

**Features**:
- Configurable playback speed
- Skip ticks for highlights
- Stance visualization (●=neutral, ▶=extended, ◀=retracted, ■=defending)
- Collision indicators (💥)

**Usage**:
```python
from src.renderer import AsciiRenderer

renderer = AsciiRenderer(arena_width=config.arena_width)
renderer.play_replay(
    result.telemetry,
    result,
    spectacle_score=score,
    playback_speed=5.0,
    skip_ticks=5
)
```

---

## Full Pipeline Example

See `test_full_pipeline.py` for a complete example demonstrating all components working together:

1. Initialize WorldConfig with spectacle-optimized parameters
2. Create fighters with world-calculated stats
3. Run match with MatchOrchestrator
4. Evaluate spectacle with SpectacleEvaluator
5. Save replay with ReplayStore
6. Render visualization with AsciiRenderer

---

## Key Design Principles

### 1. **Config-Driven Architecture**
- No hardcoded constants in Arena
- WorldConfig can be loaded from JSON/YAML
- Parameter searches can use real components (not monkey-patching)

### 2. **World Calculates Stats**
- Mass is the ONLY fighter spec
- HP and stamina are derived by world formulas
- Prevents "perfect fighter" with max everything

### 3. **Spectacle Over Balance**
- Matches evaluated on entertainment value
- Heavy fighters SHOULD win (physics)
- But every fight should be exciting

### 4. **Separation of Concerns**
- Protocol: Contract
- Arena: Physics
- Orchestrator: Coordination
- Telemetry: Recording
- Evaluator: Scoring
- Renderer: Visualization

### 5. **Replay-Driven Development**
- Full telemetry recorded by default
- Matches can be saved, loaded, analyzed
- Enables offline analysis and debugging

---

## Migration from POC

The POC (`poc/poc_minimal.py`) has been refactored into components:

| POC Element | Component |
|-------------|-----------|
| Global constants | `WorldConfig` |
| `FighterState` | `arena/fighter.py` |
| `Arena1D` | `arena/arena_1d.py` |
| Match loop | `MatchOrchestrator` |
| `calculate_match_quality()` | `SpectacleEvaluator` |
| `render_arena()` | `AsciiRenderer` |
| Snapshot generation | `protocol/combat_protocol.py` |

The POC still exists for reference and quick experiments, but production code should use the component architecture.

---

## Testing

Each component has a dedicated test file:
- `test_arena_component.py` - Arena with WorldConfig
- `test_orchestrator.py` - Match coordination
- `test_telemetry.py` - Save/load replays
- `test_evaluator.py` - Spectacle scoring
- `test_full_pipeline.py` - End-to-end integration

---

## Next Steps

1. Update `param_search.py` to use WorldConfig instead of monkey-patching
2. Build tournament system using components
3. Add more AI archetypes
4. Implement sensor filtering (bucketing, precision limits)
5. Add web-based renderer (HTML5/canvas)
