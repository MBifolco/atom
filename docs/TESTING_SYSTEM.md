# Atom Combat Testing System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Why We Built This](#why-we-built-this)
3. [System Architecture](#system-architecture)
4. [Components](#components)
5. [Usage Guide](#usage-guide)
6. [Creating Test Dummies](#creating-test-dummies)
7. [Running Tests](#running-tests)
8. [Understanding Reports](#understanding-reports)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The Atom Combat Testing System is a comprehensive framework for validating fighter behavior, detecting regressions, and ensuring consistent performance across the fighter ecosystem.

## Why We Built This

### The Problem

During development, we discovered critical issues with our hardcoded fighters:

1. **Behavioral Inconsistencies**: Fighters weren't living up to their names
   - Tank was retreating instead of standing ground
   - Rusher wasn't using aggressive stances
   - Berserker was getting stuck at walls for 984/1000 ticks

2. **Wall Grinding Crisis**: Fighters were getting trapped against arena walls
   - Root cause: Fighter B starting at position 10.0m (only 2m from 12m wall)
   - Fighters reached impossible positions (12.4m in 12m arena)
   - Wall detection was too late (1.0-1.5m), needed 2.0m early warning

3. **Testing Challenges**:
   - No repeatable way to validate fixes
   - Difficult to detect regressions
   - Manual testing was time-consuming and error-prone
   - No systematic way to test specific mechanics in isolation

### The Solution

We built a three-tier testing architecture:

```
┌─────────────────────────────────────────────┐
│            SCENARIO FIGHTERS                 │
│         (Complete strategies)                │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           BEHAVIORAL FIGHTERS               │
│        (Skill combinations)                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            ATOMIC DUMMIES                   │
│        (Single mechanics)                   │
└─────────────────────────────────────────────┘
```

## System Architecture

### Design Principles

1. **Predictability**: Test dummies have deterministic behavior
2. **Isolation**: Each dummy tests specific mechanics
3. **Composability**: Complex behaviors built from simple ones
4. **Measurability**: Quantifiable performance metrics
5. **Repeatability**: Consistent results with seed control

### Directory Structure

```
fighters/
├── test_dummies/
│   ├── atomic/           # Single-mechanic tests
│   │   ├── stationary_*.py
│   │   ├── shuttle_*.py
│   │   ├── distance_keeper_*.py
│   │   └── ...
│   ├── behavioral/       # Complex behavior tests
│   │   ├── perfect_defender.py
│   │   ├── burst_attacker.py
│   │   └── ...
│   ├── scenario/         # Full strategy tests (future)
│   ├── builder.py        # Test dummy factory
│   └── utils.py          # Utility functions
│
├── examples/             # Original fighters (test subjects)
│
outputs/
├── test_matrix_*/        # Test run results
│   ├── SUMMARY_REPORT.md
│   ├── DETAILED_REPORT.md
│   └── test_results.csv
└── baseline/             # Performance baseline

test_matrix_runner.py     # Automated testing
regression_detector.py    # Regression detection
test_atomic_dummies.py    # Dummy validation
```

## Components

### 1. Atomic Test Dummies (23 total)

**Stationary Tests** (4):
- Test stance mechanics without movement
- `stationary_neutral`, `stationary_extended`, `stationary_defending`, `stationary_retracted`

**Movement Tests** (10):
- Test physics and collision detection
- `shuttle_slow/medium/fast`, `approach_slow/fast`, `flee_always`
- `circle_left/right`, `wall_hugger_left/right`

**Stamina Tests** (3):
- Test stamina management and exploitation
- `stamina_waster`, `stamina_cycler`, `stamina_efficient`

**Distance Tests** (3):
- Test range control and spacing
- `distance_keeper_1m`, `distance_keeper_3m`, `distance_keeper_5m`

**Reactive Tests** (3):
- Test response to opponent behavior
- `mirror_movement`, `counter_movement`, `charge_on_approach`

### 2. Behavioral Fighters (6 total)

**Combat Specialists**:
- `perfect_defender`: Maximum defense optimization
- `burst_attacker`: Burst damage windows
- `perfect_kiter`: Hit-and-run tactics

**Resource Specialists**:
- `stamina_optimizer`: Perfect stamina management
- `wall_fighter`: Wall-trap strategies
- `adaptive_fighter`: Dynamic strategy adjustment

### 3. Test Infrastructure

**TestDummyBuilder**: Factory for creating test dummies
```python
from fighters.test_dummies.builder import TestDummyBuilder

dummy = TestDummyBuilder("My Test")
    .with_stance("defending")
    .maintain_distance(3.0)
    .build()
```

**Test Matrix Runner**: Automated comprehensive testing
- Runs fighters against all dummies
- Multiple seed validation
- Performance metrics collection
- Report generation

**Regression Detector**: Performance tracking
- Baseline comparison
- Configurable thresholds
- Severity classification
- Actionable reports

## Usage Guide

### Quick Start

1. **Test a single fighter against a dummy**:
```bash
python atom_fight.py fighters/examples/tank.py fighters/test_dummies/atomic/stationary_defending.py
```

2. **Run comprehensive tests**:
```bash
python test_matrix_runner.py fighters/examples/tank.py
```

3. **Test all example fighters**:
```bash
python test_matrix_runner.py
```

4. **Check for regressions**:
```bash
python regression_detector.py
```

### Creating Custom Test Dummies

#### Method 1: Using the Builder (Recommended)

```python
from fighters.test_dummies.builder import TestDummyBuilder

# Create a custom dummy
dummy = TestDummyBuilder("Aggressive Poker")
    .with_description("Maintains 2m distance and pokes with extended stance")
    .maintain_distance(target_distance=2.0, tolerance=0.3)
    .with_stance("extended")
    .build()

# Save to file
dummy.save("fighters/test_dummies/atomic/aggressive_poker.py")
```

#### Method 2: Using Templates

```python
from fighters.test_dummies.builder import ShuttleTemplate

# Quick shuttle dummy
dummy = ShuttleTemplate.create(
    name="Fast Shuttle",
    left_bound=2.0,
    right_bound=10.0,
    speed=4.0
)
dummy.save("fighters/test_dummies/atomic/fast_shuttle.py")
```

#### Method 3: Manual Creation

```python
"""
My Custom Test Dummy

Clear description of what this tests.
"""

def decide(snapshot):
    # Your deterministic logic
    return {"acceleration": 0.0, "stance": "neutral"}
```

### Running Test Matrices

#### Basic Test Run
```bash
# Test specific fighters
python test_matrix_runner.py fighters/my_fighter.py fighters/another_fighter.py

# Test all example fighters (default)
python test_matrix_runner.py
```

#### Output Structure
```
outputs/test_matrix_20241105_143022/
├── SUMMARY_REPORT.md        # Quick overview with pass/fail
├── DETAILED_REPORT.md       # Full behavioral analysis
├── test_results.csv         # Raw data for analysis
└── match_*.json             # Individual match telemetry
```

### Regression Detection

#### Setting Up Baseline
```bash
# Run initial tests
python test_matrix_runner.py

# Set as baseline
python regression_detector.py --update-baseline outputs/test_matrix_20241105_143022
```

#### Detecting Regressions
```bash
# Compare latest results with baseline
python regression_detector.py

# Compare specific results
python regression_detector.py outputs/test_matrix_20241105_150000
```

#### Understanding Severity Levels
- **HIGH**: Critical issues (wall grinding, >20% win rate drop)
- **MEDIUM**: Performance degradation (HP differential drop)
- **LOW**: Minor changes (collision count variations)

## Understanding Reports

### Summary Report

Shows high-level performance for each fighter:

```markdown
## Tank

### ATOMIC Tests

- **Average Win Rate**: 78.5%
- **Tests Passed**: 12/14

| Test Dummy | Win Rate | HP Diff | Collisions | Avg Distance |
|------------|----------|---------|------------|--------------|
| stationary_neutral | 100.0% | +85.2 | 245 | 1.8m |
| flee_always | 45.0% | -12.3 | 89 | 4.2m |
```

Key Metrics:
- **Win Rate**: Percentage of matches won
- **HP Diff**: Average HP differential at match end
- **Collisions**: Number of successful attacks
- **Avg Distance**: Average distance maintained

### Detailed Report

Provides deep behavioral analysis:

```markdown
#### stationary_defending

**Performance:**
- Win Rate: 92.3%
- HP Differential: +76.4
- Match Duration: 487 ticks
- Total Collisions: 198

**Behavior:**
- Average Distance: 1.45m
- Wall Time: 2.1%
- Position Variance: 0.34

**Stance Distribution:**
- Extended: 45.2%
- Defending: 38.1%
- Neutral: 15.3%
- Retracted: 1.4%
```

### Regression Report

Identifies performance changes:

```markdown
## 🚨 CRITICAL REGRESSIONS

### tank_atomic_wall_hugger_left
- **Metric**: wall_time
- **Baseline**: 15.2
- **Current**: 84.7
- **Change**: +69.5

## Recommendations

### Immediate Actions Required:
1. **Wall Grinding Detected** - Review movement and wall detection logic
```

## Best Practices

### For Testing

1. **Start with atomic tests**: Validate basic mechanics first
2. **Use consistent seeds**: Ensure reproducibility with `--seed 42`
3. **Test edge cases**: Wall huggers and flee_always expose issues
4. **Regular regression checks**: Run after every significant change
5. **Update baselines intentionally**: Only when improvements are expected

### For Development

1. **Test before committing**: Run matrix tests on changed fighters
2. **Create specific dummies**: Build tests for new mechanics
3. **Document dummy purpose**: Clear descriptions in docstrings
4. **Monitor wall time**: Critical metric for movement bugs
5. **Check collision counts**: Validate combat engagement

### Performance Expectations

**Good Performance Indicators**:
- Win rate > 70% against atomic dummies
- Wall time < 10% for non-wall-hugger tests
- Consistent collision counts for aggressive fighters
- HP differential positive against passive dummies

**Warning Signs**:
- Win rate < 50% against stationary dummies
- Wall time > 20% (indicates getting stuck)
- Zero collisions against stationary targets
- High position variance for defensive fighters

## Troubleshooting

### Common Issues and Solutions

**Problem**: Fighter fails against stationary dummies
- **Cause**: Poor distance management or stance selection
- **Solution**: Review approach logic and attack conditions

**Problem**: High wall time percentage
- **Cause**: Late wall detection or insufficient escape velocity
- **Solution**: Increase wall detection distance to 2.0m, escape velocity to 5.0

**Problem**: Zero collisions in aggressive matchups
- **Cause**: Incorrect distance calculations or stance timing
- **Solution**: Log distance values, verify extended stance usage

**Problem**: Regression detector shows no baseline
- **Cause**: First run or baseline not set
- **Solution**: Run `--update-baseline` with good results

**Problem**: Tests timeout
- **Cause**: Infinite loops or deadlocks in fighter logic
- **Solution**: Add timeout handling, review decision logic

### Debug Commands

```bash
# Test with verbose output
python atom_fight.py fighter.py dummy.py --save debug.json

# Validate specific dummy behavior
python test_atomic_dummies.py

# Compare two test runs
diff outputs/test_matrix_*/SUMMARY_REPORT.md

# Check telemetry for specific match
python -m json.tool outputs/test_matrix_*/match_tank_vs_flee_always.json | less
```

## Future Enhancements

### Planned Additions

1. **Scenario Fighters**: Complete game situations
2. **Performance Benchmarks**: FPS and computation metrics
3. **Visual Test Reports**: Graphs and charts
4. **CI/CD Integration**: Automated testing on commits
5. **Machine Learning Validation**: Test against trained models
6. **Multiplayer Testing**: Multi-fighter scenarios

### Contributing

To add new test capabilities:

1. Create dummy in appropriate category
2. Follow naming convention: `behavior_variant.py`
3. Add clear docstring with purpose
4. Update test matrix if needed
5. Document in this guide
6. Submit PR with test results

---

*Last Updated: November 2024*
*System Version: 1.0.0*