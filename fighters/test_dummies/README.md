# Test Dummy Fighter System

A comprehensive testing framework for Atom Combat using predictable test dummy fighters.

## Overview

The test dummy system provides deterministic, predictable fighters for testing game mechanics, fighter behaviors, and training AI. Test dummies are organized into three tiers:

1. **Atomic Dummies** - Test single mechanics in isolation
2. **Behavioral Fighters** - Test specific skills and tactics
3. **Scenario Fighters** - Validate complete strategies

## Quick Start

### Using the Builder Framework

```python
from fighters.test_dummies.builder import TestDummyBuilder

# Create a simple stationary dummy
dummy = TestDummyBuilder("My Test Dummy")
    .with_stance("defending")
    .stationary()
    .build()

# Save to file
dummy.save("fighters/test_dummies/atomic/my_dummy.py")
```

### Using Templates

```python
from fighters.test_dummies.builder import StationaryTemplate, ShuttleTemplate

# Quick stationary dummy
dummy1 = StationaryTemplate.create("Defender", stance="defending")

# Quick shuttle dummy
dummy2 = ShuttleTemplate.create("Mover", speed=3.0)
```

## Atomic Test Dummies

Located in `/fighters/test_dummies/atomic/`

### Stationary Stance Tests

Test different stances without movement:

- **stationary_neutral.py** - Stand still, neutral stance (baseline)
- **stationary_extended.py** - Stand still, extended stance (max reach)
- **stationary_defending.py** - Stand still, defending stance (1.63x defense)
- **stationary_retracted.py** - Stand still, retracted stance (minimal profile)

**Usage Example:**
```bash
# Test how your fighter handles a stationary defending opponent
python atom_fight.py my_fighter.py fighters/test_dummies/atomic/stationary_defending.py
```

### Movement Pattern Tests

Test physics and movement handling:

- **shuttle_slow.py** - Back/forth at speed 1.0 (3m ↔ 9m)
- **shuttle_medium.py** - Back/forth at speed 2.5 (3m ↔ 9m)
- **shuttle_fast.py** - Back/forth at speed 4.0 (3m ↔ 9m)
- **circle_left.py** - Always move left, bounce off wall
- **circle_right.py** - Always move right, bounce off wall
- **approach_slow.py** - Always approach at speed 1.5
- **approach_fast.py** - Always approach at speed 4.0
- **flee_always.py** - Always flee at speed 3.0
- **wall_hugger_left.py** - Stay at left wall
- **wall_hugger_right.py** - Stay at right wall

**Usage Example:**
```bash
# Test if your fighter can catch a fleeing opponent
python atom_fight.py my_fighter.py fighters/test_dummies/atomic/flee_always.py
```

## Creating Custom Test Dummies

### Method 1: Using the Builder (Recommended)

```python
from fighters.test_dummies.builder import TestDummyBuilder

dummy = TestDummyBuilder("Distance Keeper")
    .with_description("Maintains 4m distance")
    .with_stance("neutral")
    .maintain_distance(target_distance=4.0, tolerance=0.5)
    .save("fighters/test_dummies/atomic/distance_4m.py")
```

### Method 2: Manual Creation

```python
"""
My Custom Test Dummy

Description of what this tests.
"""

def decide(snapshot):
    # Your deterministic logic here
    return {"acceleration": 0.0, "stance": "neutral"}
```

### Method 3: Using Utilities

```python
from fighters.test_dummies.utils import (
    approach_opponent,
    maintain_distance,
    is_near_wall
)

def decide(snapshot):
    # Use utility functions for common patterns
    if is_near_wall(snapshot):
        acceleration = 0.0
    else:
        acceleration = maintain_distance(snapshot, target_distance=3.0)

    return {"acceleration": acceleration, "stance": "neutral"}
```

## Testing Your Fighter

### Basic Testing

Test against individual dummies:

```bash
# Test stance damage
python atom_fight.py my_fighter.py fighters/test_dummies/atomic/stationary_extended.py

# Test pursuit capability
python atom_fight.py my_fighter.py fighters/test_dummies/atomic/flee_always.py

# Test wall fighting
python atom_fight.py my_fighter.py fighters/test_dummies/atomic/wall_hugger_left.py
```

### Systematic Testing

Run the test suite:

```python
# test_my_fighter.py
from pathlib import Path
import subprocess

def test_against_dummies(fighter_path):
    dummies = [
        "stationary_neutral",
        "stationary_defending",
        "shuttle_medium",
        "approach_fast",
        "flee_always"
    ]

    for dummy in dummies:
        dummy_path = f"fighters/test_dummies/atomic/{dummy}.py"
        result = subprocess.run([
            "python", "atom_fight.py",
            fighter_path, dummy_path,
            "--seed", "42"
        ], capture_output=True, text=True)

        # Parse and check results
        print(f"{dummy}: {parse_result(result.stdout)}")
```

### Expected Behaviors

When testing against dummies, look for:

1. **Against stationary dummies:**
   - Can your fighter close distance?
   - Does it attack effectively?
   - Does it handle different stances?

2. **Against movement dummies:**
   - Can it catch fleeing opponents?
   - Can it maintain optimal distance?
   - Does it handle predictable patterns?

3. **Against wall huggers:**
   - Can it fight at walls without getting stuck?
   - Does it avoid wall damage?

## Best Practices

### For Testing

1. **Start with atomic dummies** - Test basic mechanics first
2. **Use consistent seeds** - Ensure reproducibility with `--seed`
3. **Test edge cases** - Wall huggers, flee_always expose issues
4. **Compare results** - Track performance across versions

### For Development

1. **Test incrementally** - Verify each behavior works in isolation
2. **Use appropriate dummies** - Match dummy to what you're testing
3. **Create custom dummies** - Build specific test cases as needed
4. **Document behaviors** - Explain what each dummy tests

## Utility Functions

Located in `/fighters/test_dummies/utils.py`

### Movement Helpers
- `approach_opponent(snapshot, speed)` - Calculate approach acceleration
- `flee_from_opponent(snapshot, speed)` - Calculate flee acceleration
- `maintain_distance(snapshot, target, tolerance)` - Keep specific distance
- `shuttle_movement(snapshot, left, right, speed)` - Oscillate between bounds
- `circle_movement(snapshot, direction, speed)` - Circular movement

### State Helpers
- `get_distance(snapshot)` - Get opponent distance
- `get_my_hp_percent(snapshot)` - Get HP percentage
- `get_my_stamina_percent(snapshot)` - Get stamina percentage
- `is_near_wall(snapshot, margin)` - Check wall proximity

### Stance Helpers
- `stance_by_distance(snapshot)` - Choose stance based on distance
- `stance_by_stamina(snapshot)` - Choose stance based on stamina
- `stance_by_hp_difference(snapshot)` - Choose stance based on HP
- `cycle_stances(tick, cycle_length)` - Cycle through stances

## Troubleshooting

### Common Issues

**Dummy not behaving as expected:**
- Check arena width assumptions (default ~12m)
- Verify stance names are correct
- Ensure acceleration within limits (-5.0 to 5.0)

**Test failures:**
- Use `--save` flag to capture telemetry
- Check starting positions (default 2m and 10m)
- Verify physics haven't changed

**Performance issues:**
- Keep dummy logic simple
- Avoid complex calculations
- Use utilities for common patterns

## Future Additions

Planned test dummies:

### Stamina Tests
- `stamina_waster.py` - Always extended (max drain)
- `stamina_cycler.py` - Attack at 90%, recover to 20%

### Distance Tests
- `distance_keeper_1m.py` - Maintain 1m distance
- `distance_keeper_3m.py` - Maintain 3m distance
- `distance_keeper_5m.py` - Maintain 5m distance

### Reactive Tests
- `mirror_movement.py` - Copy opponent movement
- `counter_movement.py` - Opposite of opponent
- `charge_on_approach.py` - Charge when opponent < 4m

## Contributing

To add new test dummies:

1. Create the dummy file in appropriate directory
2. Follow naming convention: `behavior_variant.py`
3. Add clear docstring explaining purpose
4. Update this README with the new dummy
5. Add tests to verify behavior

## See Also

- [TESTING_SYSTEM.md](../../docs/TESTING_SYSTEM.md) - Complete testing guide
- [builder.py](builder.py) - TestDummyBuilder source
- [utils.py](utils.py) - Utility functions source