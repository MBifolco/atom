# Test Dummy Architecture

## Design Philosophy

The test dummy system follows a hierarchical architecture designed to provide comprehensive, repeatable testing of fighter behaviors and game mechanics.

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                    TEST SUBJECTS                       │
│              (Fighters being tested)                   │
└────────────────────────┬───────────────────────────────┘
                         │ Tests against
                         ▼
┌────────────────────────────────────────────────────────┐
│                   SCENARIO TIER                        │
│            Complete strategic scenarios                │
│   • Tournament situations                              │
│   • Endgame scenarios                                  │
│   • Complex multi-phase battles                        │
└────────────────────────┬───────────────────────────────┘
                         │ Built from
                         ▼
┌────────────────────────────────────────────────────────┐
│                  BEHAVIORAL TIER                       │
│           Combinations of atomic behaviors             │
│   • Perfect defender (defense + spacing + stamina)     │
│   • Burst attacker (stamina cycles + aggression)       │
│   • Adaptive fighter (pattern recognition + counters)  │
└────────────────────────┬───────────────────────────────┘
                         │ Composed of
                         ▼
┌────────────────────────────────────────────────────────┐
│                    ATOMIC TIER                         │
│              Single isolated mechanics                 │
│   • Stationary stances (neutral, extended, defending)  │
│   • Movement patterns (shuttle, flee, approach)        │
│   • Distance keepers (1m, 3m, 5m)                     │
│   • Reactive behaviors (mirror, counter, charge)       │
└────────────────────────────────────────────────────────┘
```

## Tier Descriptions

### Atomic Tier (Foundation)
**Purpose**: Test individual game mechanics in complete isolation

**Characteristics**:
- Single, predictable behavior
- No conditional logic (or minimal)
- Deterministic outcomes
- Fast execution

**Categories**:
1. **Stationary**: Test stance properties without movement
2. **Movement**: Test physics and collision detection
3. **Stamina**: Test resource management mechanics
4. **Distance**: Test range control systems
5. **Reactive**: Test basic response patterns

**Example**:
```python
# stationary_defending.py - Tests defending stance damage reduction
def decide(snapshot):
    return {"acceleration": 0.0, "stance": "defending"}
```

### Behavioral Tier (Composition)
**Purpose**: Test combinations of mechanics and tactical patterns

**Characteristics**:
- Multiple coordinated behaviors
- Conditional decision making
- State management
- Strategy execution

**Categories**:
1. **Specialists**: Perfected single strategies
2. **Hybrids**: Balanced multi-strategy fighters
3. **Adaptors**: Dynamic strategy switching
4. **Exploiters**: Target specific weaknesses

**Example**:
```python
# perfect_kiter.py - Tests hit-and-run tactics
def decide(snapshot):
    # Combines: distance management + stance timing + movement prediction
    if distance < optimal_min:
        retreat_while_attacking()
    elif distance > optimal_max:
        approach_carefully()
    else:
        maintain_and_poke()
```

### Scenario Tier (Integration) - *Future*
**Purpose**: Test complete game situations and meta-strategies

**Planned Categories**:
1. **Tournament**: Multi-round adaptation
2. **Endgame**: Low HP situations
3. **Comeback**: Deficit recovery
4. **Pressure**: Advantage pressing

## Design Patterns

### 1. Builder Pattern
Creates test dummies programmatically:

```python
TestDummyBuilder("name")
    .with_stance("defending")
    .maintain_distance(3.0)
    .build()
```

**Benefits**:
- Consistent interface
- Reusable components
- Easy composition
- Self-documenting

### 2. Template Pattern
Pre-configured dummy types:

```python
StationaryTemplate.create("Defender", stance="defending")
ShuttleTemplate.create("Mover", speed=3.0)
```

**Benefits**:
- Quick creation
- Standard behaviors
- Reduced boilerplate

### 3. Utility Functions
Shared behavioral components:

```python
# In utils.py
def maintain_distance(snapshot, target, tolerance):
    # Reusable distance management logic
```

**Benefits**:
- DRY principle
- Consistent behavior
- Easy updates
- Bug fixes propagate

## Key Design Decisions

### 1. Deterministic Behavior
**Decision**: Test dummies use no randomness

**Rationale**:
- Reproducible results
- Easier debugging
- Reliable regression detection
- Clear cause-effect relationships

### 2. Hierarchical Organization
**Decision**: Three-tier architecture

**Rationale**:
- Progressive complexity
- Isolated testing
- Component reuse
- Clear mental model

### 3. Descriptive Naming
**Decision**: `behavior_variant.py` convention

**Rationale**:
- Self-documenting
- Easy discovery
- Clear purpose
- Sortable/groupable

### 4. Stateless Design
**Decision**: Dummies don't maintain internal state

**Rationale**:
- Simpler implementation
- Predictable behavior
- Easier testing
- No hidden dependencies

## Testing Matrix Design

### Coverage Strategy

```
Fighter → Test Against → Measure
   ↓           ↓            ↓
[Tank]  → [Stationary] → [Can close distance?]
        → [Fleeing]    → [Can catch runners?]
        → [Wall]       → [Handles walls?]
        → [Burst]      → [Survives spikes?]
```

### Metrics Hierarchy

**Primary Metrics** (Pass/Fail):
- Win rate
- HP differential
- Match completion

**Secondary Metrics** (Performance):
- Collision count
- Average distance
- Stance distribution
- Wall time

**Tertiary Metrics** (Behavior):
- Position variance
- Stamina efficiency
- Reaction time
- Pattern consistency

## Implementation Guidelines

### Creating New Atomic Dummies

1. **Single Responsibility**: Test ONE thing
2. **Clear Purpose**: Descriptive docstring
3. **Predictable**: Same input → same output
4. **Minimal Logic**: Fewest conditions possible

### Creating New Behavioral Fighters

1. **Composed Behaviors**: Combine atomic patterns
2. **Strategic Goal**: Clear tactical objective
3. **Realistic Constraints**: Respect game mechanics
4. **Observable Patterns**: Identifiable strategy

### Adding Test Categories

1. **Identify Gap**: What isn't being tested?
2. **Design Coverage**: What variations needed?
3. **Build Incrementally**: Start with atomic
4. **Document Purpose**: Why this category?

## Maintenance Strategy

### Regular Updates
- Review after game mechanic changes
- Update thresholds for balance changes
- Add tests for new features
- Deprecate obsolete tests

### Performance Optimization
- Keep atomic tests fast (<100 ticks)
- Behavioral tests moderate (<500 ticks)
- Scenario tests complete (<1000 ticks)
- Parallel execution where possible

### Documentation
- Update README for new dummies
- Document behavior changes
- Maintain architecture diagram
- Keep examples current

## Future Evolution

### Planned Enhancements

1. **Dynamic Difficulty Adjustment**
   - Dummies that scale with opponent
   - Progressive challenge levels
   - Adaptive testing thresholds

2. **Combinatorial Testing**
   - Auto-generate test combinations
   - Property-based testing
   - Fuzzing strategies

3. **Machine Learning Integration**
   - Test against trained models
   - Behavioral clustering
   - Anomaly detection

4. **Visual Analytics**
   - Movement heatmaps
   - Stance timeline visualization
   - Collision point plotting
   - Performance trending

---

*Architecture Version: 1.0.0*
*Last Updated: November 2024*