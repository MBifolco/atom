#!/usr/bin/env python3
"""
Analyze population training progress to determine if more generations are worthwhile.

Usage:
    python scripts/analysis/analyze_population_progress.py outputs/progressive_20251112_085705
"""

import sys
from pathlib import Path
import re


def parse_rankings(rankings_file: Path):
    """Parse rankings.txt file to extract Elo scores."""
    if not rankings_file.exists():
        return []

    with open(rankings_file) as f:
        lines = f.readlines()

    rankings = []
    for line in lines[2:]:  # Skip header lines
        # Parse: "1. fighter_name: ELO=1542, Record=45-12-3"
        match = re.match(r'\d+\.\s+(.+?):\s+ELO=(\d+)', line)
        if match:
            name, elo = match.groups()
            rankings.append((name, int(elo)))

    return rankings


def analyze_diversity(generation_dir: Path):
    """Analyze genetic diversity by checking fighter lineage."""
    fighters = [f.stem for f in generation_dir.glob("*.zip")]

    # Count by generation lineage
    lineage_counts = {}
    for fighter in fighters:
        match = re.search(r'_G(\d+)', fighter)
        if match:
            gen = int(match.group(1))
            lineage_counts[gen] = lineage_counts.get(gen, 0) + 1
        else:
            lineage_counts[0] = lineage_counts.get(0, 0) + 1  # Original fighters

    return lineage_counts, len(fighters)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analysis/analyze_population_progress.py <output_directory>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    models_dir = output_dir / "population" / "models"

    if not models_dir.exists():
        print(f"Error: {models_dir} not found")
        sys.exit(1)

    # Find all generations
    generations = sorted([
        int(d.name.split('_')[1])
        for d in models_dir.glob("generation_*")
    ])

    if len(generations) < 2:
        print("Need at least 2 generations to analyze progress")
        sys.exit(1)

    print("=" * 80)
    print("POPULATION TRAINING PROGRESS ANALYSIS")
    print("=" * 80)

    # Analyze Elo progression
    print("\n📊 ELO PROGRESSION")
    print("-" * 80)

    elo_history = []
    for gen in generations[-10:]:  # Last 10 generations
        rankings_file = models_dir / f"generation_{gen}" / "rankings.txt"
        rankings = parse_rankings(rankings_file)

        if rankings:
            top_elo = rankings[0][1]
            top_name = rankings[0][0]
            avg_top3 = sum(r[1] for r in rankings[:3]) / min(3, len(rankings))

            elo_history.append((gen, top_elo, avg_top3))
            print(f"Gen {gen:3d}: Top={top_elo:4d} ({top_name[:30]:30s}) Avg(Top3)={avg_top3:.0f}")

    # Calculate improvement rate
    if len(elo_history) >= 5:
        recent_5 = elo_history[-5:]
        elo_gain = recent_5[-1][1] - recent_5[0][1]
        gens_span = recent_5[-1][0] - recent_5[0][0]

        print(f"\n📈 Recent Trend (last {gens_span} gens):")
        print(f"   Top Elo gain: {elo_gain:+d} points ({elo_gain/gens_span:+.1f} per gen)")

        # Verdict on Elo
        if elo_gain > 100:
            print("   ✅ STRONG improvement - continue training")
        elif elo_gain > 50:
            print("   ⚠️  MODERATE improvement - diminishing returns")
        elif elo_gain > 20:
            print("   ⚠️  WEAK improvement - approaching plateau")
        else:
            print("   ❌ PLATEAU - more generations unlikely to help")

    # Analyze genetic diversity
    print("\n🧬 GENETIC DIVERSITY")
    print("-" * 80)

    latest_gen = generations[-1]
    gen_dir = models_dir / f"generation_{latest_gen}"
    lineage_counts, total = analyze_diversity(gen_dir)

    print(f"Generation {latest_gen} population breakdown:")
    for gen in sorted(lineage_counts.keys(), reverse=True):
        count = lineage_counts[gen]
        pct = count / total * 100
        gen_label = f"G{gen}" if gen > 0 else "Original"
        bar = "█" * int(pct / 2)
        print(f"  {gen_label:10s}: {count:2d} fighters ({pct:5.1f}%) {bar}")

    # Diversity verdict
    recent_lineage = sum(v for k, v in lineage_counts.items() if k >= latest_gen - 3)
    diversity_pct = (total - recent_lineage) / total * 100

    print(f"\n   Diversity score: {diversity_pct:.1f}% from older lineages")

    if diversity_pct > 40:
        print("   ✅ HIGH diversity - healthy competition")
    elif diversity_pct > 20:
        print("   ⚠️  MODERATE diversity - some convergence")
    else:
        print("   ❌ LOW diversity - population converged")

    # Overall recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Decision logic
    elo_improving = len(elo_history) >= 5 and (elo_history[-1][1] - elo_history[-5][1]) > 50
    diverse = diversity_pct > 20

    if elo_improving and diverse:
        print("✅ CONTINUE TRAINING - Still making progress with good diversity")
        print(f"   Suggest: {min(latest_gen + 20, 100)} generations")
    elif elo_improving and not diverse:
        print("⚠️  CONTINUE CAUTIOUSLY - Improving but losing diversity")
        print(f"   Suggest: {min(latest_gen + 10, 100)} generations, watch for plateau")
    elif not elo_improving and diverse:
        print("⚠️  CONSIDER STOPPING - Plateau despite diversity")
        print("   Suggest: Try different hyperparameters or stop here")
    else:
        print("❌ STOP TRAINING - Plateau with low diversity")
        print("   Further training unlikely to improve results")

    print(f"\nCurrent champion: {models_dir.parent / 'champion.py'}")
    print("Test against benchmarks:")
    print("  python atom_fight.py outputs/.../champion.py fighters/examples/balanced.py --episodes 100")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
