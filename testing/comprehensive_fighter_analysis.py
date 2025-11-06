#!/usr/bin/env python3
"""
Comprehensive Fighter Analysis Tool
Runs every fighter against every other fighter and analyzes behavior in detail.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time

# Fighter list
FIGHTERS = ["tank", "rusher", "balanced", "grappler", "berserker", "zoner", "dodger"]
FIGHTER_DIR = Path("fighters/examples")

class MatchAnalyzer:
    def __init__(self):
        self.results = []

    def run_match(self, fighter_a: str, fighter_b: str, seed: int = 42) -> Dict:
        """Run a single match and capture detailed telemetry."""
        print(f"\n{'='*60}")
        print(f"Running: {fighter_a.upper()} vs {fighter_b.upper()}")
        print(f"{'='*60}")

        # Run match with save flag to get telemetry
        cmd = [
            "python", "atom_fight.py",
            f"{FIGHTER_DIR}/{fighter_a}.py",
            f"{FIGHTER_DIR}/{fighter_b}.py",
            "--seed", str(seed),
            "--save", f"outputs/analysis/{fighter_a}_vs_{fighter_b}.json"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output for basic stats
        output = result.stdout
        winner = self.extract_value(output, r"Winner: (\w+)")
        duration = self.extract_value(output, r"Duration: (\d+) ticks")
        final_hp = self.extract_value(output, r"Final HP: ([\d\.]+ vs [\d\.]+)")
        collisions = self.extract_value(output, r"Collisions: (\d+)")

        # Load telemetry for detailed analysis
        telemetry_path = Path(f"outputs/analysis/{fighter_a}_vs_{fighter_b}.json")
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)

        telemetry = None
        if telemetry_path.exists():
            with open(telemetry_path, 'r') as f:
                telemetry = json.load(f)

        return {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner,
            "duration": int(duration) if duration else 0,
            "final_hp": final_hp,
            "collisions": int(collisions) if collisions else 0,
            "telemetry": telemetry,
            "output": output
        }

    def extract_value(self, text: str, pattern: str) -> str:
        """Extract value from text using regex pattern."""
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def analyze_telemetry(self, match_data: Dict) -> Dict:
        """Analyze telemetry data for movement and behavior patterns."""
        telemetry = match_data.get("telemetry")
        if not telemetry:
            return {"error": "No telemetry data"}

        ticks = telemetry.get("ticks", [])
        if not ticks:
            return {"error": "No tick data"}

        # Initialize analysis metrics
        analysis = {
            "fighter_a": {
                "name": match_data["fighter_a"],
                "wall_hits": 0,
                "time_stationary": 0,  # velocity near 0
                "time_retreating": 0,  # negative velocity
                "time_advancing": 0,   # positive velocity
                "stance_usage": {"extended": 0, "neutral": 0, "defending": 0, "retracted": 0},
                "avg_distance": 0,
                "min_distance": float('inf'),
                "max_distance": 0,
                "position_changes": 0,
                "velocity_changes": 0,
                "last_position": None,
                "last_velocity": None
            },
            "fighter_b": {
                "name": match_data["fighter_b"],
                "wall_hits": 0,
                "time_stationary": 0,
                "time_retreating": 0,
                "time_advancing": 0,
                "stance_usage": {"extended": 0, "neutral": 0, "defending": 0, "retracted": 0},
                "avg_distance": 0,
                "min_distance": float('inf'),
                "max_distance": 0,
                "position_changes": 0,
                "velocity_changes": 0,
                "last_position": None,
                "last_velocity": None
            },
            "match_dynamics": {
                "total_ticks": len(ticks),
                "chasing_phases": 0,  # One chasing, other retreating
                "standoff_phases": 0,  # Both stationary or slow
                "brawling_phases": 0,  # Close combat
                "inactive_ticks": 0    # Both barely moving
            }
        }

        arena_width = 12.0  # Standard arena width
        total_distance = 0

        for i, tick in enumerate(ticks):
            if not tick.get("fighters"):
                continue

            fighters = tick["fighters"]
            if len(fighters) < 2:
                continue

            f_a = fighters[0]
            f_b = fighters[1]

            # Calculate distance
            distance = abs(f_a["position"] - f_b["position"])
            total_distance += distance

            # Update min/max distance
            analysis["fighter_a"]["min_distance"] = min(analysis["fighter_a"]["min_distance"], distance)
            analysis["fighter_a"]["max_distance"] = max(analysis["fighter_a"]["max_distance"], distance)
            analysis["fighter_b"]["min_distance"] = analysis["fighter_a"]["min_distance"]
            analysis["fighter_b"]["max_distance"] = analysis["fighter_a"]["max_distance"]

            # Analyze fighter A
            self.analyze_fighter_tick(f_a, analysis["fighter_a"], arena_width)

            # Analyze fighter B
            self.analyze_fighter_tick(f_b, analysis["fighter_b"], arena_width)

            # Analyze match dynamics
            vel_a = f_a.get("velocity", 0)
            vel_b = f_b.get("velocity", 0)

            # Chasing: one advancing, other retreating
            if (vel_a > 1 and vel_b < -1) or (vel_a < -1 and vel_b > 1):
                analysis["match_dynamics"]["chasing_phases"] += 1

            # Standoff: both slow or stationary
            elif abs(vel_a) < 0.5 and abs(vel_b) < 0.5:
                analysis["match_dynamics"]["standoff_phases"] += 1
                if abs(vel_a) < 0.1 and abs(vel_b) < 0.1:
                    analysis["match_dynamics"]["inactive_ticks"] += 1

            # Brawling: close combat
            if distance < 2.0:
                analysis["match_dynamics"]["brawling_phases"] += 1

        # Calculate averages
        if len(ticks) > 0:
            analysis["fighter_a"]["avg_distance"] = total_distance / len(ticks)
            analysis["fighter_b"]["avg_distance"] = total_distance / len(ticks)

        return analysis

    def analyze_fighter_tick(self, fighter_data: Dict, analysis: Dict, arena_width: float):
        """Analyze a single fighter's behavior in a tick."""
        pos = fighter_data.get("position", 0)
        vel = fighter_data.get("velocity", 0)
        stance = fighter_data.get("stance", "neutral")

        # Check wall proximity
        if pos <= 0.1 or pos >= arena_width - 0.1:
            analysis["wall_hits"] += 1

        # Track movement
        if abs(vel) < 0.5:
            analysis["time_stationary"] += 1
        elif vel < -0.5:
            analysis["time_retreating"] += 1
        elif vel > 0.5:
            analysis["time_advancing"] += 1

        # Track stance usage
        if stance in analysis["stance_usage"]:
            analysis["stance_usage"][stance] += 1

        # Track position changes
        if analysis["last_position"] is not None:
            if abs(pos - analysis["last_position"]) > 0.1:
                analysis["position_changes"] += 1

        # Track velocity changes (acceleration changes)
        if analysis["last_velocity"] is not None:
            if abs(vel - analysis["last_velocity"]) > 0.5:
                analysis["velocity_changes"] += 1

        analysis["last_position"] = pos
        analysis["last_velocity"] = vel

    def run_all_matches(self):
        """Run all fighter combinations."""
        total_matches = len(FIGHTERS) * len(FIGHTERS)
        match_num = 0

        for fighter_a in FIGHTERS:
            for fighter_b in FIGHTERS:
                match_num += 1
                print(f"\n[{match_num}/{total_matches}] Testing {fighter_a} vs {fighter_b}")

                match_data = self.run_match(fighter_a, fighter_b)
                analysis = self.analyze_telemetry(match_data)

                self.results.append({
                    "match": match_data,
                    "analysis": analysis
                })

                # Quick summary
                if analysis.get("error"):
                    print(f"  ERROR: {analysis['error']}")
                else:
                    self.print_quick_summary(analysis)

        return self.results

    def print_quick_summary(self, analysis: Dict):
        """Print a quick summary of the match analysis."""
        a = analysis["fighter_a"]
        b = analysis["fighter_b"]
        d = analysis["match_dynamics"]

        print(f"\n  {a['name'].upper()}:")
        print(f"    Movement: Advance {a['time_advancing']}t, Retreat {a['time_retreating']}t, Still {a['time_stationary']}t")
        print(f"    Stances: Ext {a['stance_usage']['extended']}, Neu {a['stance_usage']['neutral']}, Def {a['stance_usage']['defending']}, Ret {a['stance_usage']['retracted']}")
        print(f"    Wall hits: {a['wall_hits']}")

        print(f"\n  {b['name'].upper()}:")
        print(f"    Movement: Advance {b['time_advancing']}t, Retreat {b['time_retreating']}t, Still {b['time_stationary']}t")
        print(f"    Stances: Ext {b['stance_usage']['extended']}, Neu {b['stance_usage']['neutral']}, Def {b['stance_usage']['defending']}, Ret {b['stance_usage']['retracted']}")
        print(f"    Wall hits: {b['wall_hits']}")

        print(f"\n  DYNAMICS:")
        print(f"    Chasing: {d['chasing_phases']}t, Standoffs: {d['standoff_phases']}t, Brawling: {d['brawling_phases']}t")
        print(f"    Inactive: {d['inactive_ticks']}t ({d['inactive_ticks']*100//d['total_ticks']}%)")

if __name__ == "__main__":
    analyzer = MatchAnalyzer()
    results = analyzer.run_all_matches()

    # Save results
    with open("outputs/analysis/comprehensive_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("Results saved to outputs/analysis/comprehensive_results.json")