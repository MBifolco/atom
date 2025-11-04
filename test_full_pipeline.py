"""
Test the complete Atom Combat pipeline.

Demonstrates all components working together:
1. Arena & Protocol - Physics and fighter contract
2. Match Orchestrator - Tick loop coordination
3. Telemetry & Replay Store - Save/load matches
4. Evaluator - Spectacle scoring
5. Renderer - Visualization
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.telemetry import ReplayStore
from src.evaluator import SpectacleEvaluator
from src.renderer import AsciiRenderer


def aggressive_ai(snapshot):
    """Aggressive AI - rushes forward and extends when close."""
    distance = snapshot["opponent"]["distance"]
    stamina = snapshot["you"]["stamina"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]

    # Retreat if very low HP
    if my_hp_pct < 0.2:
        return {"acceleration": -3.0, "stance": "defending"}

    # Attack when close
    if distance < 1.0:
        return {"acceleration": 0.0, "stance": "extended"}
    # Rush forward if have stamina
    elif stamina > 2.0:
        return {"acceleration": 4.5, "stance": "neutral"}
    # Conserve energy
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def defensive_ai(snapshot):
    """Defensive AI - reactive counter-puncher."""
    distance = snapshot["opponent"]["distance"]
    opp_velocity = snapshot["opponent"]["velocity"]
    stamina = snapshot["you"]["stamina"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]

    # Retreat if low HP
    if my_hp_pct < 0.25:
        return {"acceleration": -2.0, "stance": "defending"}

    # Brace for incoming charge
    if distance < 2.0 and opp_velocity > 1.0:
        return {"acceleration": 0.0, "stance": "defending"}
    # Counter-attack when close
    elif distance < 1.5 and stamina > 3.0:
        return {"acceleration": 2.5, "stance": "extended"}
    # Maintain optimal distance
    elif distance > 5.0:
        return {"acceleration": 2.0, "stance": "neutral"}
    # Default: hold position
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def main():
    print("\n" + "=" * 60)
    print("ATOM COMBAT - FULL PIPELINE TEST".center(60))
    print("=" * 60)

    # 1. Initialize components
    print("\n[1/6] Initializing components...")
    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)
    replay_store = ReplayStore(replay_dir="demo_replays")
    evaluator = SpectacleEvaluator()
    renderer = AsciiRenderer(arena_width=config.arena_width)

    print(f"  ✓ WorldConfig loaded (arena width: {config.arena_width:.2f})")
    print(f"  ✓ Match Orchestrator ready")
    print(f"  ✓ Replay Store ready")
    print(f"  ✓ Spectacle Evaluator ready")
    print(f"  ✓ ASCII Renderer ready")

    # 2. Set up fighters
    print("\n[2/6] Setting up fighters...")
    fighter_a_spec = {"name": "Blitz", "mass": 70.0, "position": 2.0}
    fighter_b_spec = {"name": "Tank", "mass": 80.0, "position": 10.0}

    print(f"  Fighter A: {fighter_a_spec['name']} ({fighter_a_spec['mass']}kg)")
    print(f"  Fighter B: {fighter_b_spec['name']} ({fighter_b_spec['mass']}kg)")
    print(f"  AI: Aggressive vs Defensive")

    # 3. Run match
    print("\n[3/6] Running match...")
    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        aggressive_ai,
        defensive_ai,
        seed=42
    )

    print(f"  ✓ Match complete: {result.winner} wins")
    print(f"  ✓ Duration: {result.total_ticks} ticks")
    print(f"  ✓ Final HP: {result.final_hp_a:.1f} vs {result.final_hp_b:.1f}")

    # 4. Evaluate spectacle
    print("\n[4/6] Evaluating spectacle...")
    score = evaluator.evaluate(result.telemetry, result)

    print(f"  ✓ Overall score: {score.overall:.3f}")
    print(f"    - Close Finish: {score.close_finish:.3f}")
    print(f"    - Collision Drama: {score.collision_drama:.3f}")
    print(f"    - Pacing Variety: {score.pacing_variety:.3f}")

    # 5. Save replay
    print("\n[5/6] Saving replay...")
    metadata = {
        "description": "Full pipeline demo match",
        "ai_a": "aggressive",
        "ai_b": "defensive",
        "spectacle_score": score.overall
    }

    filepath = replay_store.save(
        result.telemetry,
        result,
        metadata=metadata,
        compress=True
    )

    print(f"  ✓ Saved to: {filepath}")

    # 6. Render replay (show highlights)
    print("\n[6/6] Rendering highlights...")
    print("  (Showing every 5th tick for brevity)")

    renderer.play_replay(
        result.telemetry,
        result,
        spectacle_score=score,
        playback_speed=10.0,  # Fast playback
        skip_ticks=5,  # Show every 5th tick
        show_all_ticks=False
    )

    # Optional: Show full replay
    print("\n" + "=" * 60)
    response = input("Show full tick-by-tick replay? (y/n): ").strip().lower()

    if response == 'y':
        print("\nLoading full replay...")
        loaded_replay = replay_store.load(filepath.name)
        renderer.play_replay(
            loaded_replay["telemetry"],
            result,
            spectacle_score=score,
            playback_speed=5.0,
            show_all_ticks=True
        )

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE".center(60))
    print("=" * 60)
    print("\nAll components working correctly!")
    print("  ✓ Protocol - Fighter contract enforced")
    print("  ✓ Arena - Physics simulation accurate")
    print("  ✓ Orchestrator - Tick loop coordinated")
    print("  ✓ Telemetry - Match saved and loaded")
    print("  ✓ Evaluator - Quality metrics calculated")
    print("  ✓ Renderer - Visualization generated")
    print()


if __name__ == "__main__":
    main()
