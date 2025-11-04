"""
Generate an HTML5 animated replay of a match.

Creates a standalone .html file that can be opened in any browser.
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import HtmlRenderer
from src.ai import tactical_aggressive, tactical_defensive


def main():
    print("=" * 60)
    print("HTML5 REPLAY GENERATOR".center(60))
    print("=" * 60)

    # Run a match
    print("\n[1/3] Running match...")
    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)

    fighter_a_spec = {"name": "Blitz", "mass": 70.0, "position": 2.0}
    fighter_b_spec = {"name": "Tank", "mass": 80.0, "position": 10.0}

    result = orchestrator.run_match(
        fighter_a_spec,
        fighter_b_spec,
        tactical_aggressive,
        tactical_defensive,
        seed=42
    )

    print(f"  ✓ Match complete: {result.winner}")
    print(f"  ✓ Duration: {result.total_ticks} ticks")

    # Evaluate spectacle
    print("\n[2/3] Evaluating spectacle...")
    evaluator = SpectacleEvaluator()
    score = evaluator.evaluate(result.telemetry, result)

    print(f"  ✓ Overall score: {score.overall:.3f}")

    # Generate HTML
    print("\n[3/3] Generating HTML replay...")
    renderer = HtmlRenderer()

    output_file = renderer.generate_replay_html(
        result.telemetry,
        result,
        "replay_viewer.html",
        spectacle_score=score,
        playback_speed=1.0
    )

    print(f"  ✓ Generated: {output_file}")

    print("\n" + "=" * 60)
    print("COMPLETE!".center(60))
    print("=" * 60)
    print(f"\nOpen '{output_file}' in your browser to view the animated replay!")
    print("\nFeatures:")
    print("  • Play/Pause controls")
    print("  • Adjustable playback speed (0.25x - 5x)")
    print("  • Step-by-step navigation")
    print("  • Real-time stats display")
    print("  • Smooth 60fps animation")
    print("  • Visual stance indicators")
    print("  • Collision highlighting")
    print()


if __name__ == "__main__":
    main()
