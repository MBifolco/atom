# Fighter Movement Bug Fixes

## Bugs Fixed

### 1. Orchestrator Using Stale Fighter States
**Symptom**: Fighters weren't moving in battles. Final HP always at starting values (93.7), no collisions.

**Root Cause**: `match_orchestrator.py` was generating snapshots using the original `fighter_a` and `fighter_b` variables created before the tick loop, not the updated states from `arena.fighter_a` and `arena.fighter_b`.

**Fix** (match_orchestrator.py:104-110):
```python
# OLD - used stale initial states
snapshot_a = generate_snapshot(fighter_a, fighter_b, tick, self.config.arena_width)

# NEW - use current arena state
current_fighter_a = arena.fighter_a
current_fighter_b = arena.fighter_b
snapshot_a = generate_snapshot(current_fighter_a, current_fighter_b, tick, self.config.arena_width)
```

Also fixed telemetry recording (line 143) and final HP reporting (lines 159, 183) to use current arena state.

### 2. Missing Direction Field in Snapshot
**Symptom**: Fighters all moved in same direction, didn't approach each other.

**Root Cause**: Snapshot only provided `distance` to opponent, not which direction (left or right). Fighters couldn't determine which way to move in 1D space.

**Fix** (combat_protocol.py:111-123):
```python
# Calculate direction to opponent
if my_fighter.position < opp_fighter.position:
    direction = 1.0  # Opponent to the right
elif my_fighter.position > opp_fighter.position:
    direction = -1.0  # Opponent to the left
else:
    direction = 0.0  # Same position

# Add to snapshot
"opponent": {
    "distance": float(distance),
    "direction": float(direction),  # NEW FIELD
    ...
}
```

Updated all 5 fighter archetypes to use `direction` field (multiply acceleration by direction to move toward/away from opponent).

### 3. JAX Stance Integer Conversion
**Symptom**: Not immediately visible, but could cause issues.

**Root Cause**: Arena returns `FighterStateJAX` with integer stance (0, 1, 2), but fighters expect string stance ("neutral", "extended", "defending").

**Fix** (combat_protocol.py:104-107):
```python
# Convert stance from int to string if JAX fighter
if isinstance(my_fighter, FighterStateJAX):
    my_stance = stance_to_str(int(my_fighter.stance))
```

### 4. Missing Hit Events
**Symptom**: "Collisions: 0" even when damage was being dealt.

**Root Cause**: JIT-compiled code returned empty events list. Hit events not generated from state changes.

**Fix** (arena_1d_jax_jit.py:239-265):
```python
# Generate events from state changes
if new_hp_a < old_hp_a:
    events.append({
        "type": "HIT",
        "attacker": self.name_b,
        "defender": self.name_a,
        "damage": float(old_hp_a - new_hp_a),
        "tick": int(self.state.tick)
    })
```

Changed atom_fight.py to count "HIT" events instead of "COLLISION" events.

## Tests Added

Created `tests/test_orchestrator_state.py` with tests for:
- ✅ Snapshot uses current arena state, not stale initial state
- ✅ Direction field present in snapshot
- ✅ Fighters approach each other using direction
- ✅ Orchestrator updates state each tick

Created `tests/test_fighter_behavior.py` with tests for:
- ✅ All fighters produce valid actions
- ✅ Fighters change position over time
- ✅ Aggressive fighters approach each other
- ✅ State format matches protocol

## Why These Bugs Weren't Caught

**Analysis** (following new PERMANENT_CONTEXT rule):

1. **No integration tests** checking end-to-end state flow from arena → orchestrator → fighter
2. **No contract tests** verifying snapshot format matches fighter expectations
3. **Tests mocked state directly** instead of using `generate_snapshot()`
4. **No tests for state updates** over multiple ticks

## Additional Tests Needed

Based on this analysis, added/updated:
- Integration tests using real orchestrator flow
- State format validation tests
- Multi-tick state progression tests
- Event generation tests

## Results

**Before**: Fighters frozen, 0 collisions, starting HP
**After**: Full combat, knockouts, 8 hits in 94 ticks, 0.0 vs 8.5 HP

✅ All major bugs fixed and tested!