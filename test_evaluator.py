"""
Test the Spectacle Evaluator component.
"""

from src.arena import WorldConfig
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator


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


def balanced_ai(snapshot):
    """Balanced tactical AI."""
    distance = snapshot["opponent"]["distance"]
    my_hp_pct = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
    opp_hp_pct = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
    my_stamina_pct = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]

    # Emergency retreat
    if my_hp_pct < 0.3:
        return {"acceleration": -4.0, "stance": "neutral"}

    # Winning - press advantage
    if my_hp_pct > opp_hp_pct + 0.2 and my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 4.0, "stance": "neutral"}

    # Losing - play defensive
    if my_hp_pct < opp_hp_pct - 0.2:
        if distance < 2.0:
            return {"acceleration": -3.0, "stance": "defending"}
        else:
            return {"acceleration": 0.0, "stance": "neutral"}

    # Even match - measured aggression
    if my_stamina_pct > 0.4:
        if distance < 1.5:
            return {"acceleration": 0.0, "stance": "extended"}
        else:
            return {"acceleration": 3.0, "stance": "neutral"}
    else:
        return {"acceleration": 0.0, "stance": "neutral"}


def main():
    print("=== Spectacle Evaluator Test ===\n")

    config = WorldConfig()
    orchestrator = MatchOrchestrator(config, max_ticks=1000, record_telemetry=True)
    evaluator = SpectacleEvaluator()

    # Test multiple matchups
    matchups = [
        ("Aggressor vs Defender", aggressive_ai, defensive_ai, 75.0, 65.0),
        ("Balanced vs Balanced", balanced_ai, balanced_ai, 70.0, 70.0),
        ("Heavy vs Light", aggressive_ai, defensive_ai, 85.0, 50.0),
    ]

    for name, ai_a, ai_b, mass_a, mass_b in matchups:
        print(f"=== {name} ===")

        fighter_a_spec = {"name": "Fighter_A", "mass": mass_a, "position": 2.0}
        fighter_b_spec = {"name": "Fighter_B", "mass": mass_b, "position": 10.0}

        # Run match
        result = orchestrator.run_match(fighter_a_spec, fighter_b_spec, ai_a, ai_b, seed=42)

        # Evaluate spectacle
        score = evaluator.evaluate(result.telemetry, result)

        print(f"Winner: {result.winner}")
        print(f"Duration: {result.total_ticks} ticks")
        print(f"Final HP: {result.final_hp_a:.1f} vs {result.final_hp_b:.1f}")
        print()

        print("Spectacle Scores:")
        print(f"  Duration:            {score.duration:.3f}")
        print(f"  Close Finish:        {score.close_finish:.3f}")
        print(f"  Stamina Drama:       {score.stamina_drama:.3f}")
        print(f"  Comeback Potential:  {score.comeback_potential:.3f}")
        print(f"  Positional Exchange: {score.positional_exchange:.3f}")
        print(f"  Pacing Variety:      {score.pacing_variety:.3f}")
        print(f"  Collision Drama:     {score.collision_drama:.3f}")
        print(f"  OVERALL:             {score.overall:.3f}")
        print()

        # Provide narrative assessment
        if score.overall >= 0.8:
            assessment = "EXCELLENT - Highly entertaining match!"
        elif score.overall >= 0.6:
            assessment = "GOOD - Solid competitive fight"
        elif score.overall >= 0.4:
            assessment = "FAIR - Some exciting moments"
        else:
            assessment = "POOR - Needs improvement"

        print(f"Assessment: {assessment}")
        print()
        print("-" * 60)
        print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    main()
