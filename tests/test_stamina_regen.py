#!/usr/bin/env python3
"""
Test stamina regeneration mechanics
"""

from src.arena import Arena1D, WorldConfig, FighterState


def test_stamina_regen_neutral():
    """Test that stamina regenerates when in neutral stance with no movement."""
    config = WorldConfig()

    fighter_a = FighterState.create("Test Fighter", mass=70.0, position=2.0, world_config=config)
    fighter_b = FighterState.create("Dummy", mass=70.0, position=10.0, world_config=config)

    # Reduce fighter_a stamina to 50%
    initial_stamina = fighter_a.max_stamina / 2
    fighter_a.stamina = initial_stamina

    arena = Arena1D(fighter_a, fighter_b, config=config)

    print("="*60)
    print("Test 1: Stamina regen in neutral stance with no movement")
    print("="*60)
    print(f"Initial stamina: {initial_stamina:.2f} / {fighter_a.max_stamina:.2f}")
    print(f"Expected regen per tick: {config.stamina_base_regen * config.stamina_neutral_bonus:.4f}")
    print()

    # Run 100 ticks with no movement, neutral stance
    for tick in range(100):
        action_a = {"acceleration": 0.0, "stance": "neutral"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}
        arena.step(action_a, action_b)

        if tick % 10 == 0:
            print(f"Tick {tick:3d}: Stamina = {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")

    print()
    stamina_gained = fighter_a.stamina - initial_stamina
    expected_regen = config.stamina_base_regen * config.stamina_neutral_bonus * 100

    print(f"Final stamina: {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")
    print(f"Stamina gained: {stamina_gained:.2f}")
    print(f"Expected gain: {expected_regen:.2f}")
    print(f"Match: {'✓ PASS' if abs(stamina_gained - expected_regen) < 0.1 else '✗ FAIL'}")
    print()

    return abs(stamina_gained - expected_regen) < 0.1


def test_stamina_cost_vs_regen():
    """Test stamina cost vs regeneration with movement."""
    config = WorldConfig()

    fighter_a = FighterState.create("Test Fighter", mass=70.0, position=2.0, world_config=config)
    fighter_b = FighterState.create("Dummy", mass=70.0, position=10.0, world_config=config)

    arena = Arena1D(fighter_a, fighter_b, config=config)

    print("="*60)
    print("Test 2: Stamina cost vs regeneration with acceleration")
    print("="*60)
    print(f"Initial stamina: {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")
    print()

    # Run 50 ticks with acceleration
    for tick in range(50):
        action_a = {"acceleration": 2.0, "stance": "neutral"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}
        arena.step(action_a, action_b)

        if tick % 10 == 0:
            print(f"Tick {tick:3d}: Stamina = {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")

    print()
    print(f"Stamina should have decreased due to acceleration cost")
    print(f"Final stamina: {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")

    stamina_after_accel = fighter_a.stamina

    # Now rest for 50 ticks
    print("\nResting for 50 ticks...")
    for tick in range(50):
        action_a = {"acceleration": 0.0, "stance": "neutral"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}
        arena.step(action_a, action_b)

        if tick % 10 == 0:
            print(f"Tick {tick:3d}: Stamina = {fighter_a.stamina:.2f} / {fighter_a.max_stamina:.2f}")

    print()
    stamina_regained = fighter_a.stamina - stamina_after_accel
    print(f"Stamina regained: {stamina_regained:.2f}")
    print(f"Match: {'✓ PASS - stamina regenerated' if stamina_regained > 0 else '✗ FAIL - no regen'}")
    print()

    return stamina_regained > 0


def test_stamina_cap():
    """Test that stamina doesn't exceed max."""
    config = WorldConfig()

    fighter_a = FighterState.create("Test Fighter", mass=70.0, position=2.0, world_config=config)
    fighter_b = FighterState.create("Dummy", mass=70.0, position=10.0, world_config=config)

    arena = Arena1D(fighter_a, fighter_b, config=config)

    print("="*60)
    print("Test 3: Stamina cap at max_stamina")
    print("="*60)
    print(f"Max stamina: {fighter_a.max_stamina:.2f}")
    print()

    # Fighter should already be at max, but let's verify it stays there
    initial_stamina = fighter_a.stamina

    # Run 100 ticks - stamina should not exceed max
    for tick in range(100):
        action_a = {"acceleration": 0.0, "stance": "neutral"}
        action_b = {"acceleration": 0.0, "stance": "neutral"}
        arena.step(action_a, action_b)

    print(f"Initial stamina: {initial_stamina:.2f}")
    print(f"Final stamina: {fighter_a.stamina:.2f}")
    print(f"Max stamina: {fighter_a.max_stamina:.2f}")
    print(f"Match: {'✓ PASS - capped at max' if fighter_a.stamina <= fighter_a.max_stamina else '✗ FAIL - exceeded max'}")
    print()

    return fighter_a.stamina <= fighter_a.max_stamina


def main():
    print("\n" + "="*60)
    print("STAMINA REGENERATION TESTS")
    print("="*60)
    print()

    test1 = test_stamina_regen_neutral()
    test2 = test_stamina_cost_vs_regen()
    test3 = test_stamina_cap()

    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (Neutral regen): {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Test 2 (Cost vs regen): {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Test 3 (Stamina cap): {'✓ PASS' if test3 else '✗ FAIL'}")
    print()

    if all([test1, test2, test3]):
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)


if __name__ == "__main__":
    main()
