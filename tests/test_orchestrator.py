"""
Test the Match Orchestrator component.
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator


def aggressive_ai(snapshot):
    """Aggressive AI that rushes and extends."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}
    elif stamina > 2.0:
        return {"acceleration": 4.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def defensive_ai(snapshot):
    """Defensive AI that counters and defends."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    stamina = snapshot["you"]["stamina"]

    if distance < 2.0 and opp_velocity > 1.0:
        return {"acceleration": 0.0, "stance": "defending"}
    elif distance < 1.5 and stamina > 3.0:
        return {"acceleration": 2.0, "stance": "extended"}
    elif distance > 5.0:
        return {"acceleration": 2.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def main():
    print("=== Match Orchestrator Test ===\n")

    # Load default config
    config = WorldConfig()

    # Create orchestrator
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)

    # Define fighter specs
    fighter_a_spec = {"name": "Aggressor", "mass": 75.0, "position": 2.0}
    fighter_b_spec = {"name": "Defender", "mass": 65.0, "position": 10.0}

    print(f"Running match: {fighter_a_spec['name']} (75kg) vs {fighter_b_spec['name']} (65kg)\n")

    # Run match
    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        aggressive_ai,
        defensive_ai,
        seed=42
    )

    # Display results
    print("=== Match Results ===")
    print(f"Winner: {result.winner}")
    print(f"Duration: {result.total_ticks} ticks")
    print(f"Final HP - {fighter_a_spec['name']}: {result.final_hp_a:.1f}")
    print(f"Final HP - {fighter_b_spec['name']}: {result.final_hp_b:.1f}")
    print(f"Total collisions: {len([e for e in result.events if e['type'] == 'COLLISION'])}")
    print()

    # Analyze telemetry
    if result.telemetry["ticks"]:
        print("=== Telemetry Summary ===")
        print(f"Total ticks recorded: {len(result.telemetry['ticks'])}")

        # Find peak velocities
        max_vel_a = max(tick["fighter_a"]["velocity"] for tick in result.telemetry["ticks"])
        max_vel_b = max(tick["fighter_b"]["velocity"] for tick in result.telemetry["ticks"])
        print(f"Max velocity - {fighter_a_spec['name']}: {max_vel_a:.2f}")
        print(f"Max velocity - {fighter_b_spec['name']}: {max_vel_b:.2f}")

        # Find minimum stamina
        min_stam_a = min(tick["fighter_a"]["stamina"] for tick in result.telemetry["ticks"])
        min_stam_b = min(tick["fighter_b"]["stamina"] for tick in result.telemetry["ticks"])
        print(f"Min stamina - {fighter_a_spec['name']}: {min_stam_a:.1f}")
        print(f"Min stamina - {fighter_b_spec['name']}: {min_stam_b:.1f}")

        # Collision analysis
        collision_events = [e for e in result.events if e["type"] == "COLLISION"]
        if collision_events:
            avg_damage_a = sum(e["damage_to_a"] for e in collision_events) / len(collision_events)
            avg_damage_b = sum(e["damage_to_b"] for e in collision_events) / len(collision_events)
            print(f"Avg damage per collision - {fighter_a_spec['name']}: {avg_damage_a:.1f}")
            print(f"Avg damage per collision - {fighter_b_spec['name']}: {avg_damage_b:.1f}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
