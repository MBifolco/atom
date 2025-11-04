"""
Test improved stamina economy and tactical AI.
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import AsciiRenderer
from src.ai import tactical_aggressive, tactical_defensive, tactical_balanced


def main():
    print("\n" + "=" * 60)
    print("IMPROVED COMBAT TEST".center(60))
    print("=" * 60)

    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)
    evaluator = SpectacleEvaluator()
    renderer = AsciiRenderer(arena_width=config.arena_width)

    print("\n[Stamina Economy]")
    print(f"  Accel Cost: {config.stamina_accel_cost}")
    print(f"  Base Regen: {config.stamina_base_regen}")
    print(f"  Neutral Bonus: {config.stamina_neutral_bonus}x")
    print(f"  Extended Drain: {config.stances['extended'].drain}/tick")
    print(f"  Defending Drain: {config.stances['defending'].drain}/tick")

    # Calculate example stamina math
    print("\n[Stamina Math @ 70kg fighter]")
    accel_cost_max = 4.5 * config.stamina_accel_cost * config.dt
    neutral_regen = config.stamina_base_regen * config.stamina_neutral_bonus
    print(f"  Max accel (4.5) cost: {accel_cost_max:.4f}/tick")
    print(f"  Neutral regen: {neutral_regen:.4f}/tick")
    print(f"  Net at max accel + neutral: {neutral_regen - accel_cost_max:.4f}/tick")
    print(f"  Net at max accel + extended: {neutral_regen - accel_cost_max - config.stances['extended'].drain:.4f}/tick")

    # Test match
    print("\n" + "=" * 60)
    print("MATCH 1: Tactical Aggressive vs Tactical Defensive")
    print("=" * 60)

    fighter_a_spec = {"name": "Blitz", "mass": 70.0, "position": 2.0}
    fighter_b_spec = {"name": "Counter", "mass": 75.0, "position": 10.0}

    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        tactical_aggressive,
        tactical_defensive,
        seed=42
    )

    score = evaluator.evaluate(result.telemetry, result)

    # Show highlights
    print("\nRendering highlights (every 10th tick)...")
    renderer.play_replay(
        result.telemetry,
        result,
        spectacle_score=score,
        playback_speed=20.0,
        skip_ticks=10,
        show_all_ticks=False
    )

    # Analyze stamina usage
    print("\n" + "=" * 60)
    print("STAMINA ANALYSIS")
    print("=" * 60)

    min_stam_a = min(t["fighter_a"]["stamina"] for t in result.telemetry["ticks"])
    max_stam_a = max(t["fighter_a"]["stamina"] for t in result.telemetry["ticks"])
    min_stam_b = min(t["fighter_b"]["stamina"] for t in result.telemetry["ticks"])
    max_stam_b = max(t["fighter_b"]["stamina"] for t in result.telemetry["ticks"])

    print(f"{fighter_a_spec['name']}:")
    print(f"  Min: {min_stam_a:.1f}")
    print(f"  Max: {max_stam_a:.1f}")
    print(f"  Range: {max_stam_a - min_stam_a:.1f}")

    print(f"\n{fighter_b_spec['name']}:")
    print(f"  Min: {min_stam_b:.1f}")
    print(f"  Max: {max_stam_b:.1f}")
    print(f"  Range: {max_stam_b - min_stam_b:.1f}")

    # Count low stamina moments
    low_stam_ticks_a = sum(1 for t in result.telemetry["ticks"] if t["fighter_a"]["stamina"] < t["fighter_a"]["max_stamina"] * 0.3)
    low_stam_ticks_b = sum(1 for t in result.telemetry["ticks"] if t["fighter_b"]["stamina"] < t["fighter_b"]["max_stamina"] * 0.3)
    total_ticks = len(result.telemetry["ticks"])

    print(f"\nLow stamina (<30%) moments:")
    print(f"  {fighter_a_spec['name']}: {low_stam_ticks_a}/{total_ticks} ticks ({100*low_stam_ticks_a/total_ticks:.1f}%)")
    print(f"  {fighter_b_spec['name']}: {low_stam_ticks_b}/{total_ticks} ticks ({100*low_stam_ticks_b/total_ticks:.1f}%)")

    # Position analysis
    print("\n" + "=" * 60)
    print("POSITION ANALYSIS")
    print("=" * 60)

    positions_a = [t["fighter_a"]["position"] for t in result.telemetry["ticks"]]
    positions_b = [t["fighter_b"]["position"] for t in result.telemetry["ticks"]]

    # Count wall contact
    wall_ticks_a = sum(1 for p in positions_a if p < 0.5 or p > config.arena_width - 0.5)
    wall_ticks_b = sum(1 for p in positions_b if p < 0.5 or p > config.arena_width - 0.5)

    print(f"Wall contact:")
    print(f"  {fighter_a_spec['name']}: {wall_ticks_a}/{total_ticks} ticks ({100*wall_ticks_a/total_ticks:.1f}%)")
    print(f"  {fighter_b_spec['name']}: {wall_ticks_b}/{total_ticks} ticks ({100*wall_ticks_b/total_ticks:.1f}%)")

    # Position variance
    import statistics
    pos_std_a = statistics.stdev(positions_a) if len(positions_a) > 1 else 0
    pos_std_b = statistics.stdev(positions_b) if len(positions_b) > 1 else 0

    print(f"\nPosition variance (std dev):")
    print(f"  {fighter_a_spec['name']}: {pos_std_a:.2f}m")
    print(f"  {fighter_b_spec['name']}: {pos_std_b:.2f}m")

    print("\n" + "=" * 60)

    # Second match - balanced vs balanced
    print("\nMATCH 2: Tactical Balanced vs Tactical Balanced")
    print("=" * 60)

    fighter_a_spec = {"name": "Alpha", "mass": 65.0, "position": 2.0}
    fighter_b_spec = {"name": "Beta", "mass": 75.0, "position": 10.0}

    result2 = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        tactical_balanced,
        tactical_balanced,
        seed=123
    )

    score2 = evaluator.evaluate(result2.telemetry, result2)

    renderer.play_replay(
        result2.telemetry,
        result2,
        spectacle_score=score2,
        playback_speed=20.0,
        skip_ticks=10,
        show_all_ticks=False
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
