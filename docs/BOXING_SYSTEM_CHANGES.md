# Boxing Combat System - Implementation Changes

## Summary
Complete overhaul of the combat system from continuous collision damage to discrete hit boxing mechanics with physics-based damage calculation.

## Major Changes

### 1. Stance System Simplified (3 stances instead of 4)
- **Removed**: "retracted" stance (unnecessary complexity)
- **Kept**: neutral, extended, defending
- **New mechanic**: Defending stance GAINS stamina (+0.10/tick)

### 2. Discrete Hit System
- **Old**: Continuous damage every tick during collision
- **New**: Discrete hits with 5-tick cooldown
- Impact force = velocity × mass (physics-based)
- Minimum threshold (0.5 force) to register hit

### 3. Stamina Changes
- **Landing hit**: -2.0 stamina
- **Blocking hit**: -1.0 stamina
- **Defending**: +0.10 stamina/tick (regenerates!)
- **Recoil**: 30% velocity reduction after hits

### 4. Fighter Archetypes
**Removed all old fighters**: tank, rusher, grappler, etc.

**Created 5 boxing styles**:
- `boxer.py` - Footwork and stamina management
- `slugger.py` - Heavy hits, aggressive
- `counter_puncher.py` - Defensive, waits for openings
- `swarmer.py` - Constant pressure
- `out_fighter.py` - Distance control

### 5. Code Simplification
- **Removed**: `arena_1d.py`, `arena_1d_jax.py` (old implementations)
- **Kept only**: `arena_1d_jax_jit.py` (JAX JIT version)
- No backward compatibility - clean break

## Files Changed

### Core Physics
- `src/arena/world_config.py` - Added hit system parameters
- `src/arena/fighter.py` - Added `last_hit_tick` tracking
- `src/arena/arena_1d_jax_jit.py` - Complete discrete hit implementation
- `src/arena/__init__.py` - Updated to export Arena1DJAXJit

### Training
- `src/training/gym_env.py` - 3-stance action space (0-2.99)
- Removed references to old arena implementations

### Fighters
- Deleted: All fighters in `fighters/AIs/`
- Deleted: Old examples (tank, rusher, etc.)
- Created: 5 new boxing archetypes

### Testing
- Created comprehensive test suite:
  - `tests/test_discrete_hits.py`
  - `tests/test_world_config.py`
  - `tests/test_jax_compatibility.py`
  - `tests/test_integration.py`

### Documentation
- Created: `docs/BOXING_COMBAT_SYSTEM.md`
- Updated: `README.md` with 3-stance system
- Removed: Outdated JAX phase documentation

## Configuration

New parameters in `WorldConfig`:
```python
hit_cooldown_ticks = 5
hit_impact_threshold = 0.5
hit_recoil_multiplier = 0.3
hit_stamina_cost = 2.0
block_stamina_cost = 1.0
```

## Testing Results
✅ Combat functional - damage dealt correctly
✅ Stamina mechanics working (defending regenerates)
✅ All tests passing
✅ JAX JIT compilation working

## Migration Notes
⚠️ **Breaking Changes** - No backward compatibility
- All existing trained models must be retrained
- Action space changed (4 stances → 3 stances)
- Reward dynamics completely different
- Combat mechanics fundamentally changed

## Next Steps
1. Train new models with boxing mechanics
2. Tune physics parameters based on fight quality
3. Create tournament with new fighters