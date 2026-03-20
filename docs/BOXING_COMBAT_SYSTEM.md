# Boxing-Style Combat System

## Overview
Atom Combat now features a discrete hit boxing-style combat system with physics-based damage calculation, hit cooldowns, and strategic stamina management.

## Core Mechanics

### 3-Stance System
The combat system uses three distinct stances, each with unique properties:

| Stance | Reach | Defense | Stamina Effect | Purpose |
|--------|-------|---------|----------------|---------|
| **Neutral** | 0.28 | 1.06x | No drain, 3.5x regen bonus | Movement & recovery |
| **Extended** | 0.82 | 0.89x | -0.08/tick | Attacking (only stance that can deal damage) |
| **Defending** | 0.38 | 1.50x | +0.10/tick (regenerates!) | Blocking & stamina recovery |

### Discrete Hit System
Unlike continuous damage systems, hits are discrete events with:
- **Hit Cooldown**: 5 ticks minimum between hits
- **Impact Threshold**: 0.5 minimum force to register
- **Physics-Based Damage**: Damage scales with impact force (velocity × mass)

### Damage Calculation
```python
# Impact force determines hit strength dynamically
impact_force = relative_velocity × reduced_mass
damage = base_damage × (1 + impact_force/10) × stamina_multiplier / defense_multiplier
```

### Stamina Mechanics
- **Landing a hit**: -2.0 stamina
- **Blocking a hit**: -1.0 stamina
- **Defending stance**: +0.10 stamina/tick
- **Exhaustion**: Velocity reduced by 50% at 0 stamina

### Recoil System
After landing a hit, fighters experience 30% velocity reduction, creating natural separation and preventing continuous collision damage.

## Fighter Mass System

Fighter mass (40-91 kg) determines key stats:
- **HP**: Increases with mass (48-125 HP)
- **Stamina**: Decreases with mass (12.4-5.8 stamina)
- **Acceleration**: Lighter fighters accelerate faster

## Boxing Archetypes

Five distinct fighter styles are provided as examples:

### Boxer (`fighters/examples/boxer.py`)
- Quick jabs and footwork
- Excellent stamina management
- Hit-and-move tactics

### Slugger (`fighters/examples/slugger.py`)
- Heavy hits with momentum
- Aggressive forward pressure
- Trades defense for offense

### Counter-Puncher (`fighters/examples/counter_puncher.py`)
- Defensive style
- Waits for opponent mistakes
- Exploits openings

### Swarmer (`fighters/examples/swarmer.py`)
- Constant pressure
- High punch volume
- Minimal defense

### Out-Fighter (`fighters/examples/out_fighter.py`)
- Distance control
- Hit-and-move tactics
- Points over power

## Configuration Parameters

Located in `src/arena/world_config.py`:

```python
# Discrete hit system
hit_cooldown_ticks: int = 5          # Minimum ticks between hits
hit_impact_threshold: float = 0.5    # Minimum impact force
hit_recoil_multiplier: float = 0.3   # Velocity reduction on hit
hit_stamina_cost: float = 2.0        # Stamina cost when landing hit
block_stamina_cost: float = 1.0      # Stamina cost when blocking
```

## Implementation Details

### JAX JIT Compilation
The entire physics engine is JIT-compiled using JAX for GPU acceleration:
- Integer stance representation (no strings in JIT code)
- Pure functional design (no side effects)
- Vectorizable for parallel environments

### Key Files
- `src/arena/arena_1d_jax_jit.py` - Core physics engine
- `src/arena/world_config.py` - Configuration and parameters
- `src/arena/fighter.py` - Fighter state with hit tracking
- `src/training/gym_env.py` - Gymnasium environment (3-stance action space)

### Testing
Comprehensive test suite in `tests/`:
- `test_discrete_hits.py` - Hit mechanics testing
- `test_world_config.py` - Configuration validation
- `test_jax_compatibility.py` - JAX/JIT verification
- `test_integration.py` - Full combat scenarios

## Training Considerations

### Action Space
The Gym environment uses a continuous action space:
- `action[0]`: Acceleration (-1.0 to 1.0, scaled to max_acceleration)
- `action[1]`: Stance selector (0.0 to 2.99, cast to 0/1/2)

### Observation Space
9-dimensional observation vector:
- Fighter position, velocity, HP%, stamina%
- Opponent distance, relative velocity, HP%, stamina%
- Arena width

### Reward Structure
- Damage dealt/taken (with HP differential bonuses)
- Stamina efficiency rewards
- Stance-appropriate bonuses
- Anti-avoidance penalties

## Migration from Previous System

### Breaking Changes
1. **Removed "retracted" stance** - Now only 3 stances
2. **Discrete hits** - No continuous collision damage
3. **New stamina costs** - On hit events, not just movement
4. **Defending regenerates** - No longer drains stamina

### No Backward Compatibility
This is a clean break from the previous system. All trained models will need to be retrained with the new combat mechanics.

## Quick Start

```python
from src.atom.runtime.arena import WorldConfig, FighterState, Arena1DJAXJit

# Create fighters
config = WorldConfig()
boxer = FighterState.create("Boxer", 65.0, 5.0, config)
brawler = FighterState.create("Brawler", 75.0, 7.0, config)

# Create arena
arena = Arena1DJAXJit(boxer, brawler, config)

# Run combat
action_boxer = {"acceleration": 0.5, "stance": "extended"}
action_brawler = {"acceleration": -0.5, "stance": "defending"}
events = arena.step(action_boxer, action_brawler)
```