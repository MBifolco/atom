"""
Test the Telemetry & Replay Store component.
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.telemetry import ReplayStore


def aggressive_ai(snapshot):
    """Aggressive AI."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]

    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}
    elif stamina > 2.0:
        return {"acceleration": 4.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def defensive_ai(snapshot):
    """Defensive AI."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]

    if distance < 2.0 and opp_velocity > 1.0:
        return {"acceleration": 0.0, "stance": "defending"}
    elif distance < 1.5:
        return {"acceleration": 2.0, "stance": "extended"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def main():
    print("=== Telemetry & Replay Store Test ===\n")

    # 1. Run a match to generate telemetry
    print("Running match to generate telemetry...")
    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)

    fighter_a_spec = {"name": "Aggressor", "mass": 75.0, "position": 2.0}
    fighter_b_spec = {"name": "Defender", "mass": 65.0, "position": 10.0}

    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        aggressive_ai,
        defensive_ai,
        seed=42
    )

    print(f"Match complete: {result.winner} in {result.total_ticks} ticks\n")

    # 2. Save replay
    print("Saving replay...")
    replay_store = ReplayStore(replay_dir="test_replays")

    metadata = {
        "description": "Test match for telemetry component",
        "ai_a": "aggressive",
        "ai_b": "defensive"
    }

    # Save compressed
    filepath = replay_store.save(
        result.telemetry,
        result,
        metadata=metadata,
        compress=True
    )
    print(f"Saved to: {filepath}\n")

    # 3. List replays
    print("Available replays:")
    replays = replay_store.list_replays()
    for replay in replays:
        info = replay_store.get_replay_info(replay)
        print(f"  - {replay}")
        print(f"    {info['fighter_a']} vs {info['fighter_b']}")
        print(f"    Winner: {info['winner']}, Duration: {info['total_ticks']} ticks")
    print()

    # 4. Load replay
    print("Loading replay...")
    loaded_data = replay_store.load(filepath.name)

    print(f"Loaded replay version: {loaded_data['version']}")
    print(f"Timestamp: {loaded_data['timestamp']}")
    print(f"Winner: {loaded_data['result']['winner']}")
    print(f"Total ticks: {loaded_data['result']['total_ticks']}")
    print(f"Telemetry ticks recorded: {len(loaded_data['telemetry']['ticks'])}")
    print(f"Total events: {len(loaded_data['events'])}")
    print(f"Metadata: {loaded_data['metadata']}")
    print()

    # 5. Verify telemetry integrity
    print("Verifying telemetry integrity...")
    first_tick = loaded_data['telemetry']['ticks'][0]
    last_tick = loaded_data['telemetry']['ticks'][-1]

    print(f"First tick ({first_tick['tick']}):")
    print(f"  Fighter A HP: {first_tick['fighter_a']['hp']:.1f}")
    print(f"  Fighter B HP: {first_tick['fighter_b']['hp']:.1f}")

    print(f"Last tick ({last_tick['tick']}):")
    print(f"  Fighter A HP: {last_tick['fighter_a']['hp']:.1f}")
    print(f"  Fighter B HP: {last_tick['fighter_b']['hp']:.1f}")

    # Verify last tick matches result
    assert abs(last_tick['fighter_a']['hp'] - loaded_data['result']['final_hp_a']) < 0.1
    assert abs(last_tick['fighter_b']['hp'] - loaded_data['result']['final_hp_b']) < 0.1
    print("\nTelemetry integrity verified!")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
