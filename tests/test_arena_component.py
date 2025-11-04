"""
Test the new Arena component with WorldConfig system.
"""

from src.arena import Arena1D, WorldConfig, FighterState
from src.protocol.combat_protocol import generate_snapshot


def simple_aggressive_ai(snapshot):
    """Simple aggressive AI for testing."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}
    elif stamina > 2.0:
        return {"acceleration": 4.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def simple_defensive_ai(snapshot):
    """Simple defensive AI for testing."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]

    if distance < 2.0 and opp_velocity > 1.0:
        return {"acceleration": 0.0, "stance": "defending"}
    elif distance < 1.5:
        return {"acceleration": 2.0, "stance": "extended"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def main():
    # 1. Load default spectacle-optimized config
    config = WorldConfig()

    print("=== Atom Combat - Arena Component Test ===\n")
    print(f"World Config:")
    print(f"  Arena Width: {config.arena_width:.2f}")
    print(f"  Friction: {config.friction:.4f}")
    print(f"  Max Acceleration: {config.max_acceleration:.4f}")
    print(f"  Base Collision Damage: {config.base_collision_damage:.4f}")
    print()

    # 2. Create fighters using world-calculated stats
    fighter_a = FighterState.create("Aggressor", mass=75.0, position=2.0, world_config=config)
    fighter_b = FighterState.create("Defender", mass=65.0, position=10.0, world_config=config)

    print(f"Fighter A: {fighter_a.name}")
    print(f"  Mass: {fighter_a.mass:.1f}kg")
    print(f"  HP: {fighter_a.hp:.1f}/{fighter_a.max_hp:.1f}")
    print(f"  Stamina: {fighter_a.stamina:.1f}/{fighter_a.max_stamina:.1f}")
    print()

    print(f"Fighter B: {fighter_b.name}")
    print(f"  Mass: {fighter_b.mass:.1f}kg")
    print(f"  HP: {fighter_b.hp:.1f}/{fighter_b.max_hp:.1f}")
    print(f"  Stamina: {fighter_b.stamina:.1f}/{fighter_b.max_stamina:.1f}")
    print()

    # 3. Create arena with config
    arena = Arena1D(fighter_a, fighter_b, config=config)

    # 4. Run simulation
    print("=== Running Match ===\n")
    max_ticks = 500
    collision_count = 0

    for tick in range(max_ticks):
        # Generate snapshots
        snapshot_a = generate_snapshot(fighter_a, fighter_b, tick, config.arena_width)
        snapshot_b = generate_snapshot(fighter_b, fighter_a, tick, config.arena_width)

        # Get actions from AI
        action_a = simple_aggressive_ai(snapshot_a)
        action_b = simple_defensive_ai(snapshot_b)

        # Execute tick
        events = arena.step(action_a, action_b)

        # Track collisions
        for event in events:
            if event["type"] == "COLLISION":
                collision_count += 1
                print(f"Tick {tick}: COLLISION - Damage to A: {event['damage_to_a']:.1f}, Damage to B: {event['damage_to_b']:.1f}")

        # Check for finish
        if arena.is_finished():
            print(f"\nMatch ended at tick {tick}")
            print(f"Winner: {arena.get_winner()}")
            print(f"Final HP - A: {fighter_a.hp:.1f}, B: {fighter_b.hp:.1f}")
            print(f"Total collisions: {collision_count}")
            break
    else:
        print(f"\nMatch timeout at {max_ticks} ticks")
        print(f"Final HP - A: {fighter_a.hp:.1f}, B: {fighter_b.hp:.1f}")
        print(f"Total collisions: {collision_count}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
