"""
Test defensive counter-attack strategy.

Validates that defending is a viable strategy - a fighter can defend
long enough for the attacker to exhaust stamina, then counter-attack.
"""

import pytest
from src.arena import Arena1D, WorldConfig, FighterState


def test_defensive_stance_enables_counter_attack():
    """
    Test that a defender can survive aggression by using defending stance,
    wait for attacker to exhaust stamina, then counter-attack to win.
    """
    config = WorldConfig()

    # Equal mass fighters
    attacker = FighterState.create("Attacker", mass=70.0, position=2.0, world_config=config)
    defender = FighterState.create("Defender", mass=70.0, position=10.0, world_config=config)

    arena = Arena1D(attacker, defender, config)

    phase = "defending"  # Start by defending
    ticks_in_phase = 0

    for tick in range(2000):
        # Attacker always attacks aggressively
        action_attacker = {"acceleration": 4.5, "stance": "extended"}

        # Defender: Defend until attacker exhausted, then counter
        attacker_stamina_pct = attacker.stamina / attacker.max_stamina

        if phase == "defending":
            # Defend and regenerate stamina
            action_defender = {"acceleration": 0.0, "stance": "defending"}
            ticks_in_phase += 1

            # Switch to counter when attacker is exhausted
            if attacker_stamina_pct < 0.2:  # Less than 20% stamina
                phase = "counter"
                ticks_in_phase = 0
                print(f"Tick {tick}: Switching to counter-attack (attacker stamina: {attacker_stamina_pct*100:.1f}%)")
        else:  # counter phase
            # Counter-attack
            action_defender = {"acceleration": 3.0, "stance": "extended"}
            ticks_in_phase += 1

        arena.step(action_attacker, action_defender)

        if arena.is_finished():
            print(f"Match ended at tick {tick}")
            break

    # Get final state
    attacker_hp_pct = attacker.hp / attacker.max_hp
    defender_hp_pct = defender.hp / defender.max_hp
    attacker_stamina_pct = attacker.stamina / attacker.max_stamina
    defender_stamina_pct = defender.stamina / defender.max_stamina

    print(f"\nFinal State:")
    print(f"  Attacker: HP={attacker_hp_pct*100:.1f}%, Stamina={attacker_stamina_pct*100:.1f}%")
    print(f"  Defender: HP={defender_hp_pct*100:.1f}%, Stamina={defender_stamina_pct*100:.1f}%")
    print(f"  Phase reached: {phase}")

    # Defender should have survived the defense phase
    assert phase == "counter", \
        f"Defender should reach counter phase (got: {phase})"

    # Defender should have more HP (defending reduces damage taken)
    # OR should have won
    if arena.is_finished():
        winner = arena.get_winner()
        # If match finished, defender should have won or at least be competitive
        if winner == "Defender":
            print(f"✓ Defender won!")
        else:
            # If attacker won, it should be close (defender put up good fight)
            assert defender_hp_pct > 0.3, \
                f"Even if defender loses, should put up good fight: {defender_hp_pct*100:.1f}% HP remaining"
    else:
        # Match didn't finish - defender should be healthier or have successfully countered
        assert defender_hp_pct >= attacker_hp_pct * 0.8, \
            f"Defender should be competitive: def={defender_hp_pct*100:.1f}%, att={attacker_hp_pct*100:.1f}%"

    # Attacker should have low stamina (exhausted from constant attacking)
    assert attacker_stamina_pct < 0.5, \
        f"Attacker should be exhausted: {attacker_stamina_pct*100:.1f}%"

    # Defender should have decent stamina (or at least not less than attacker)
    assert defender_stamina_pct >= attacker_stamina_pct, \
        f"Defender should have at least as much stamina: def={defender_stamina_pct*100:.1f}%, att={attacker_stamina_pct*100:.1f}%"

    # Most importantly: Defender should be healthier (the strategy worked!)
    assert defender_hp_pct >= attacker_hp_pct, \
        f"Defensive counter strategy should work: def={defender_hp_pct*100:.1f}%, att={attacker_hp_pct*100:.1f}%"


def test_defending_reduces_damage_significantly():
    """
    Test that defending stance provides enough damage reduction
    to make defense viable as a strategy.
    """
    config = WorldConfig()

    # Run two scenarios: one with defending, one with neutral
    scenarios = []

    for defender_stance in ["neutral", "defending"]:
        attacker = FighterState.create("Attacker", mass=70.0, position=2.0, world_config=config)
        defender = FighterState.create("Defender", mass=70.0, position=10.0, world_config=config)

        arena = Arena1D(attacker, defender, config)

        # Run for fixed number of ticks
        for _ in range(300):
            action_attacker = {"acceleration": 4.0, "stance": "extended"}
            action_defender = {"acceleration": 0.0, "stance": defender_stance}

            arena.step(action_attacker, action_defender)

            if arena.is_finished():
                break

        scenarios.append({
            "stance": defender_stance,
            "hp_lost": defender.max_hp - defender.hp,
            "hp_pct": defender.hp / defender.max_hp,
            "survived": defender.hp > 0
        })

    neutral_scenario = next(s for s in scenarios if s["stance"] == "neutral")
    defending_scenario = next(s for s in scenarios if s["stance"] == "defending")

    print(f"\nNeutral stance: {neutral_scenario['hp_pct']*100:.1f}% HP remaining (lost {neutral_scenario['hp_lost']:.1f})")
    print(f"Defending stance: {defending_scenario['hp_pct']*100:.1f}% HP remaining (lost {defending_scenario['hp_lost']:.1f})")

    # Defending should take significantly less damage
    damage_reduction = (neutral_scenario['hp_lost'] - defending_scenario['hp_lost']) / neutral_scenario['hp_lost']
    print(f"Damage reduction: {damage_reduction*100:.1f}%")

    assert defending_scenario['hp_lost'] < neutral_scenario['hp_lost'], \
        f"Defending should reduce damage taken"

    # Should provide at least 20% damage reduction
    assert damage_reduction >= 0.20, \
        f"Defending should provide significant damage reduction: {damage_reduction*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
