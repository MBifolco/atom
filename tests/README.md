# Atom Combat Test Suite

Comprehensive unit and integration tests for the Atom Combat game engine.

## Running Tests

Run all tests:
```bash
python3 -m pytest tests/ -v
```

Run specific test file:
```bash
python3 -m pytest tests/test_world_dynamics.py -v
```

Run specific test class:
```bash
python3 -m pytest tests/test_world_dynamics.py::TestStaminaMechanics -v
```

Run specific test:
```bash
python3 -m pytest tests/test_world_dynamics.py::TestStaminaMechanics::test_stamina_drain_from_acceleration -v
```

## Test Coverage

### test_world_dynamics.py (26 tests)

Comprehensive tests ensuring world dynamics yield expected results.

#### TestStaminaMechanics (7 tests)
- ✓ Stamina drain from acceleration (scaled by mass)
- ✓ Heavier fighters consume more stamina for same acceleration
- ✓ Extended stance drains stamina
- ✓ Neutral stance regenerates stamina faster (bonus multiplier)
- ✓ Cannot attack (extended stance) at 0 stamina
- ✓ Can defend (defensive stances) at 0 stamina
- ✓ Stamina caps at maximum value

#### TestDamageCalculation (6 tests)
- ✓ Equal fighters deal equal damage at full stamina
- ✓ Heavier fighters have mass advantage in damage
- ✓ Defending stance provides damage reduction
- ✓ Extended vs non-extended creates massive damage asymmetry (attack advantage)
- ✓ Stamina affects damage output (low stamina = reduced damage)
- ✓ Zero stamina still deals minimum damage (25% baseline)

#### TestPhysics (7 tests)
- ✓ Positive acceleration increases velocity
- ✓ Negative acceleration decreases velocity
- ✓ Friction reduces velocity over time
- ✓ Velocity updates position correctly
- ✓ Wall collisions stop movement and reset velocity
- ✓ Velocity is clamped to max_velocity
- ✓ Acceleration is clamped to max_acceleration

#### TestCombatScenarios (3 tests)
- ✓ Pure aggression exhausts stamina (validates stamina management is critical)
- ✓ Match ends when fighter reaches 0 HP
- ✓ Mutual KO results in draw

#### TestStaminaIntegration (3 tests)
- ✓ Spamming extended stance exhausts stamina
- ✓ Neutral stance recovers stamina effectively
- ✓ Stamina management matters in long fights

## Key Insights from Tests

### Stamina Management is Critical
The tests reveal that pure aggression (constantly attacking + accelerating) actually LOSES to passive stance management due to stamina exhaustion. This confirms the game properly incentivizes strategic stamina management.

### Attack Advantage Asymmetry
When only one fighter uses extended stance (attacking), they gain massive advantage (~100x damage multiplier). This creates high-risk, high-reward offensive play.

### Mass Effects
Heavier fighters:
- Deal more damage (mass ratio in damage formula)
- Consume more stamina for same acceleration
- Have more HP (calculated from mass by world config)

### Stamina-Damage Scaling
- 100% stamina = 100% damage output
- 0% stamina = 25% damage output
- Linear scaling in between

## Adding New Tests

When adding new tests, follow the existing patterns:

1. **Test Class Organization**: Group related tests into classes
2. **setup_method()**: Create fresh test fixtures for each test
3. **Descriptive Names**: Use clear, descriptive test names that explain what's being tested
4. **Assertions**: Include helpful failure messages with actual vs expected values
5. **Comments**: Explain WHY behavior is expected, not just WHAT is expected

Example:
```python
def test_new_mechanic(self):
    """Test that new mechanic behaves as expected."""
    # Setup
    fighter = FighterState.create("Test", mass=70.0, position=5.0, world_config=self.config)

    # Execute
    result = some_function(fighter)

    # Assert
    assert result == expected_value, \
        f"Mechanic should produce {expected_value}, got {result}"
```

## CI/CD Integration

These tests should be run:
- Before every commit (pre-commit hook)
- On every pull request
- Before every deployment

All tests must pass before merging code changes.
