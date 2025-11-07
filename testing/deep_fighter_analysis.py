#!/usr/bin/env python3
"""
Deep Fighter Behavior Analysis
Runs all fighter combinations and analyzes behavior patterns in detail.
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, List
import json

FIGHTERS = ["tank", "rusher", "balanced", "grappler", "berserker", "zoner", "dodger"]

class DeepAnalyzer:
    def __init__(self):
        self.results = {}
        self.output_dir = Path("outputs/deep_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_match(self, fighter_a: str, fighter_b: str, seed: int = 42) -> Dict:
        """Run a match and capture all output for analysis."""
        # Run match and capture telemetry
        cmd = [
            "python", "atom_fight.py",
            f"fighters/examples/{fighter_a}.py",
            f"fighters/examples/{fighter_b}.py",
            "--seed", str(seed),
            "--save", f"{self.output_dir}/{fighter_a}_vs_{fighter_b}.json"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout

        # Extract basic stats
        stats = {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": self.extract_pattern(output, r"Winner: (\w+)"),
            "duration": self.extract_pattern(output, r"Duration: (\d+)"),
            "collisions": self.extract_pattern(output, r"Collisions: (\d+)"),
            "final_hp": self.extract_pattern(output, r"Final HP: ([\d\.]+ vs [\d\.]+)"),
            "timeout": "timeout" in output.lower()
        }

        # Load and analyze telemetry
        telemetry_file = self.output_dir / f"{fighter_a}_vs_{fighter_b}.json"
        if telemetry_file.exists():
            with open(telemetry_file, 'r') as f:
                telemetry = json.load(f)
                behavior = self.analyze_behavior(telemetry, stats)
                stats["behavior"] = behavior

        return stats

    def extract_pattern(self, text: str, pattern: str):
        """Extract value using regex."""
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def analyze_behavior(self, telemetry: Dict, stats: Dict) -> Dict:
        """Deeply analyze fighter behavior from telemetry."""
        ticks = telemetry.get("ticks", [])
        if not ticks:
            return {"error": "No telemetry data"}

        # Initialize tracking for both fighters
        behavior = {
            "fighter_a": {
                "name": stats["fighter_a"],
                "wall_collisions": 0,
                "time_near_wall": 0,
                "stationary_ticks": 0,
                "retreating_ticks": 0,
                "advancing_ticks": 0,
                "stance_breakdown": {"extended": 0, "neutral": 0, "defending": 0, "retracted": 0},
                "avg_velocity": 0,
                "max_velocity": 0,
                "direction_changes": 0,
                "time_in_combat_range": 0,  # < 2m
                "time_at_distance": 0,       # > 5m
                "aggressive_actions": 0,     # extended + advancing
                "defensive_actions": 0,      # defending + retreating
                "wall_grinding_time": 0
            },
            "fighter_b": {
                "name": stats["fighter_b"],
                "wall_collisions": 0,
                "time_near_wall": 0,
                "stationary_ticks": 0,
                "retreating_ticks": 0,
                "advancing_ticks": 0,
                "stance_breakdown": {"extended": 0, "neutral": 0, "defending": 0, "retracted": 0},
                "avg_velocity": 0,
                "max_velocity": 0,
                "direction_changes": 0,
                "time_in_combat_range": 0,
                "time_at_distance": 0,
                "aggressive_actions": 0,
                "defensive_actions": 0,
                "wall_grinding_time": 0
            },
            "match_patterns": {
                "chase_sequences": 0,
                "mutual_retreat": 0,
                "standoffs": 0,
                "close_combat_time": 0,
                "separation_events": 0,
                "longest_inactive_streak": 0,
                "current_inactive_streak": 0
            }
        }

        arena_width = 12.0
        prev_vel_a = 0
        prev_vel_b = 0
        total_vel_a = 0
        total_vel_b = 0

        for i, tick in enumerate(ticks):
            if "fighters" not in tick or len(tick["fighters"]) < 2:
                continue

            f_a = tick["fighters"][0]
            f_b = tick["fighters"][1]

            # Distance between fighters
            distance = abs(f_a["position"] - f_b["position"])

            # Analyze Fighter A
            pos_a = f_a.get("position", 0)
            vel_a = f_a.get("velocity", 0)
            stance_a = f_a.get("stance", "neutral")

            # Wall detection
            if pos_a <= 0.5 or pos_a >= arena_width - 0.5:
                behavior["fighter_a"]["time_near_wall"] += 1
                if abs(vel_a) < 0.1:  # Stuck on wall
                    behavior["fighter_a"]["wall_grinding_time"] += 1
            if pos_a <= 0.1 or pos_a >= arena_width - 0.1:
                behavior["fighter_a"]["wall_collisions"] += 1

            # Movement analysis
            if abs(vel_a) < 0.3:
                behavior["fighter_a"]["stationary_ticks"] += 1
            elif vel_a < -0.5:
                behavior["fighter_a"]["retreating_ticks"] += 1
            elif vel_a > 0.5:
                behavior["fighter_a"]["advancing_ticks"] += 1

            # Stance tracking
            behavior["fighter_a"]["stance_breakdown"][stance_a] += 1

            # Velocity tracking
            total_vel_a += abs(vel_a)
            behavior["fighter_a"]["max_velocity"] = max(behavior["fighter_a"]["max_velocity"], abs(vel_a))

            # Direction changes
            if prev_vel_a * vel_a < 0 and abs(prev_vel_a) > 1 and abs(vel_a) > 1:
                behavior["fighter_a"]["direction_changes"] += 1
            prev_vel_a = vel_a

            # Combat range analysis
            if distance < 2.0:
                behavior["fighter_a"]["time_in_combat_range"] += 1
            if distance > 5.0:
                behavior["fighter_a"]["time_at_distance"] += 1

            # Aggressive vs defensive
            if stance_a == "extended" and vel_a > 0.5:
                behavior["fighter_a"]["aggressive_actions"] += 1
            if (stance_a == "defending" or stance_a == "retracted") and vel_a < -0.5:
                behavior["fighter_a"]["defensive_actions"] += 1

            # Analyze Fighter B (same logic)
            pos_b = f_b.get("position", 0)
            vel_b = f_b.get("velocity", 0)
            stance_b = f_b.get("stance", "neutral")

            if pos_b <= 0.5 or pos_b >= arena_width - 0.5:
                behavior["fighter_b"]["time_near_wall"] += 1
                if abs(vel_b) < 0.1:
                    behavior["fighter_b"]["wall_grinding_time"] += 1
            if pos_b <= 0.1 or pos_b >= arena_width - 0.1:
                behavior["fighter_b"]["wall_collisions"] += 1

            if abs(vel_b) < 0.3:
                behavior["fighter_b"]["stationary_ticks"] += 1
            elif vel_b < -0.5:
                behavior["fighter_b"]["retreating_ticks"] += 1
            elif vel_b > 0.5:
                behavior["fighter_b"]["advancing_ticks"] += 1

            behavior["fighter_b"]["stance_breakdown"][stance_b] += 1
            total_vel_b += abs(vel_b)
            behavior["fighter_b"]["max_velocity"] = max(behavior["fighter_b"]["max_velocity"], abs(vel_b))

            if prev_vel_b * vel_b < 0 and abs(prev_vel_b) > 1 and abs(vel_b) > 1:
                behavior["fighter_b"]["direction_changes"] += 1
            prev_vel_b = vel_b

            if distance < 2.0:
                behavior["fighter_b"]["time_in_combat_range"] += 1
            if distance > 5.0:
                behavior["fighter_b"]["time_at_distance"] += 1

            if stance_b == "extended" and vel_b > 0.5:
                behavior["fighter_b"]["aggressive_actions"] += 1
            if (stance_b == "defending" or stance_b == "retracted") and vel_b < -0.5:
                behavior["fighter_b"]["defensive_actions"] += 1

            # Match pattern analysis
            # Chase: one advancing, other retreating
            if (vel_a > 1 and vel_b < -1) or (vel_b > 1 and vel_a < -1):
                behavior["match_patterns"]["chase_sequences"] += 1

            # Mutual retreat
            if vel_a < -1 and vel_b < -1:
                behavior["match_patterns"]["mutual_retreat"] += 1

            # Standoff: both barely moving
            if abs(vel_a) < 0.5 and abs(vel_b) < 0.5:
                behavior["match_patterns"]["standoffs"] += 1
                behavior["match_patterns"]["current_inactive_streak"] += 1
            else:
                behavior["match_patterns"]["longest_inactive_streak"] = max(
                    behavior["match_patterns"]["longest_inactive_streak"],
                    behavior["match_patterns"]["current_inactive_streak"]
                )
                behavior["match_patterns"]["current_inactive_streak"] = 0

            # Close combat
            if distance < 1.5:
                behavior["match_patterns"]["close_combat_time"] += 1

            # Separation (going from close to far)
            if i > 0 and distance > 3.0:
                prev_tick = ticks[i-1]["fighters"]
                prev_dist = abs(prev_tick[0]["position"] - prev_tick[1]["position"])
                if prev_dist < 2.0:
                    behavior["match_patterns"]["separation_events"] += 1

        # Calculate averages
        num_ticks = len(ticks)
        if num_ticks > 0:
            behavior["fighter_a"]["avg_velocity"] = total_vel_a / num_ticks
            behavior["fighter_b"]["avg_velocity"] = total_vel_b / num_ticks

        return behavior

    def run_all_matches(self):
        """Run all 49 combinations."""
        print("DEEP FIGHTER BEHAVIOR ANALYSIS")
        print("=" * 80)

        for i, fighter_a in enumerate(FIGHTERS):
            for j, fighter_b in enumerate(FIGHTERS):
                match_id = f"{fighter_a}_vs_{fighter_b}"
                print(f"\n[{i*7+j+1}/49] {fighter_a.upper()} vs {fighter_b.upper()}")

                result = self.run_match(fighter_a, fighter_b)
                self.results[match_id] = result

                # Print immediate insights
                self.print_match_summary(result)

        return self.results

    def print_match_summary(self, result: Dict):
        """Print detailed match summary."""
        print(f"  Winner: {result['winner']} | Collisions: {result['collisions']} | Duration: {result['duration']}")

        if 'behavior' in result and 'error' not in result['behavior']:
            b = result['behavior']
            fa = b['fighter_a']
            fb = b['fighter_b']
            patterns = b['match_patterns']

            # Fighter A summary
            print(f"\n  {fa['name'].upper()} Behavior:")
            print(f"    Movement: Adv={fa['advancing_ticks']}t Ret={fa['retreating_ticks']}t Still={fa['stationary_ticks']}t")
            print(f"    Stances: Ext={fa['stance_breakdown']['extended']} Neu={fa['stance_breakdown']['neutral']} "
                  f"Def={fa['stance_breakdown']['defending']} Ret={fa['stance_breakdown']['retracted']}")
            print(f"    Wall: Collisions={fa['wall_collisions']} NearWall={fa['time_near_wall']}t Grinding={fa['wall_grinding_time']}t")
            print(f"    Combat: CloseRange={fa['time_in_combat_range']}t Aggressive={fa['aggressive_actions']} Defensive={fa['defensive_actions']}")

            # Fighter B summary
            print(f"\n  {fb['name'].upper()} Behavior:")
            print(f"    Movement: Adv={fb['advancing_ticks']}t Ret={fb['retreating_ticks']}t Still={fb['stationary_ticks']}t")
            print(f"    Stances: Ext={fb['stance_breakdown']['extended']} Neu={fb['stance_breakdown']['neutral']} "
                  f"Def={fb['stance_breakdown']['defending']} Ret={fb['stance_breakdown']['retracted']}")
            print(f"    Wall: Collisions={fb['wall_collisions']} NearWall={fb['time_near_wall']}t Grinding={fb['wall_grinding_time']}t")
            print(f"    Combat: CloseRange={fb['time_in_combat_range']}t Aggressive={fb['aggressive_actions']} Defensive={fb['defensive_actions']}")

            # Match dynamics
            print(f"\n  Match Dynamics:")
            print(f"    Chases={patterns['chase_sequences']}t Standoffs={patterns['standoffs']}t MutualRetreat={patterns['mutual_retreat']}t")
            print(f"    CloseCombat={patterns['close_combat_time']}t Separations={patterns['separation_events']}")
            print(f"    LongestInactive={patterns['longest_inactive_streak']}t")

            # Issues/Anomalies
            issues = []
            if fa['wall_grinding_time'] > 50:
                issues.append(f"{fa['name']} stuck on wall ({fa['wall_grinding_time']}t)")
            if fb['wall_grinding_time'] > 50:
                issues.append(f"{fb['name']} stuck on wall ({fb['wall_grinding_time']}t)")
            if patterns['standoffs'] > 500:
                issues.append(f"Excessive standoffs ({patterns['standoffs']}t)")
            if patterns['mutual_retreat'] > 100:
                issues.append(f"Both fighters retreating too much ({patterns['mutual_retreat']}t)")
            if fa['stationary_ticks'] > 300:
                issues.append(f"{fa['name']} too passive ({fa['stationary_ticks']}t still)")
            if fb['stationary_ticks'] > 300:
                issues.append(f"{fb['name']} too passive ({fb['stationary_ticks']}t still)")

            if issues:
                print(f"\n  ⚠️  ISSUES DETECTED:")
                for issue in issues:
                    print(f"    - {issue}")

if __name__ == "__main__":
    analyzer = DeepAnalyzer()
    results = analyzer.run_all_matches()

    # Save comprehensive results
    with open("outputs/deep_analysis/full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Results saved to outputs/deep_analysis/")