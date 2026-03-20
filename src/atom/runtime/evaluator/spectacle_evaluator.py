"""
Atom Combat - Spectacle Evaluator

Evaluates match quality based on entertainment value, not balance.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SpectacleScore:
    """
    Multi-dimensional spectacle scoring for a match.
    Each metric ranges from 0.0 (poor) to 1.0 (excellent).
    """
    duration: float  # Match pacing (ideal 100-400 ticks)
    close_finish: float  # How close was the ending
    stamina_drama: float  # Exhaustion moments (10-30% time at <30% stamina)
    comeback_potential: float  # HP lead changes
    positional_exchange: float  # Movement variety (5-20% position swaps)
    pacing_variety: float  # Speed variance (std_dev 0.5-1.5)
    collision_drama: float  # Impactful exchanges (8-25 collisions)
    overall: float  # Weighted average

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "duration": self.duration,
            "close_finish": self.close_finish,
            "stamina_drama": self.stamina_drama,
            "comeback_potential": self.comeback_potential,
            "positional_exchange": self.positional_exchange,
            "pacing_variety": self.pacing_variety,
            "collision_drama": self.collision_drama,
            "overall": self.overall
        }


class SpectacleEvaluator:
    """
    Evaluates match quality on entertainment/spectacle.

    Philosophy: WHO wins doesn't matter - only QUALITY of the fight matters.
    Focus on drama, momentum swings, close finishes, and variety.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize evaluator with metric weights.

        Args:
            weights: Optional custom weights for each metric (default: all equal)
        """
        self.weights = weights or {
            "duration": 1.0,
            "close_finish": 1.0,
            "stamina_drama": 1.0,
            "comeback_potential": 1.0,
            "positional_exchange": 1.0,
            "pacing_variety": 1.0,
            "collision_drama": 1.0
        }

    def evaluate(self, telemetry: Dict[str, Any], match_result: Any) -> SpectacleScore:
        """
        Evaluate a match for spectacle quality.

        Args:
            telemetry: Match telemetry from MatchOrchestrator
            match_result: MatchResult object

        Returns:
            SpectacleScore with all metrics
        """
        scores = {}

        # 1. Match Duration (want 100-400 ticks - not too quick, not endless)
        duration = match_result.total_ticks
        if duration < 30:
            scores['duration'] = 0.0  # Instant KO, no drama
        elif duration > 500:
            scores['duration'] = 0.2  # Boring slugfest
        elif 100 <= duration <= 400:
            scores['duration'] = 1.0  # Perfect length
        else:
            if duration < 100:
                scores['duration'] = duration / 100
            else:
                scores['duration'] = max(0.2, 1.0 - (duration - 400) / 200)

        # 2. Close Finish (nail-biters are exciting!)
        winner_hp = max(match_result.final_hp_a, match_result.final_hp_b)
        loser_hp = min(match_result.final_hp_a, match_result.final_hp_b)

        # Get max_hp from first tick
        if telemetry["ticks"]:
            first_tick = telemetry["ticks"][0]
            max_hp_a = first_tick["fighter_a"]["max_hp"]
            max_hp_b = first_tick["fighter_b"]["max_hp"]
            winner_max_hp = max_hp_a if match_result.final_hp_a > match_result.final_hp_b else max_hp_b
            winner_hp_pct = winner_hp / winner_max_hp if winner_max_hp > 0 else 1.0
        else:
            winner_hp_pct = 1.0

        if winner_hp_pct < 0.2:
            scores['close_finish'] = 1.0  # Photo finish!
        elif winner_hp_pct < 0.4:
            scores['close_finish'] = 0.9  # Close call
        elif winner_hp_pct < 0.6:
            scores['close_finish'] = 0.7  # Competitive
        elif winner_hp_pct < 0.8:
            scores['close_finish'] = 0.4  # Dominant
        else:
            scores['close_finish'] = 0.0  # Boring stomp

        # 3. Stamina Drama (exhaustion moments = tension!)
        stamina_samples = []
        for tick_data in telemetry.get("ticks", []):
            a_stam_pct = tick_data["fighter_a"]["stamina"] / tick_data["fighter_a"]["max_stamina"]
            b_stam_pct = tick_data["fighter_b"]["stamina"] / tick_data["fighter_b"]["max_stamina"]
            stamina_samples.append(min(a_stam_pct, b_stam_pct))

        if stamina_samples:
            critical_moments = sum(1 for s in stamina_samples if s < 0.3)
            drama_rate = critical_moments / len(stamina_samples)

            # Want 10-30% of fight at critical stamina
            if 0.1 <= drama_rate <= 0.3:
                scores['stamina_drama'] = 1.0
            elif 0.05 <= drama_rate < 0.1:
                scores['stamina_drama'] = 0.7
            elif drama_rate > 0.3:
                scores['stamina_drama'] = 0.5  # Too much exhaustion
            else:
                scores['stamina_drama'] = 0.3  # No drama
        else:
            scores['stamina_drama'] = 0.0

        # 4. Comeback Potential (HP lead changes = exciting!)
        hp_samples = []
        for tick_data in telemetry.get("ticks", []):
            hp_a = tick_data["fighter_a"]["hp"]
            hp_b = tick_data["fighter_b"]["hp"]
            hp_samples.append((hp_a, hp_b))

        if len(hp_samples) >= 5:
            lead_changes = 0
            for i in range(1, len(hp_samples)):
                prev_lead = hp_samples[i-1][0] - hp_samples[i-1][1]
                curr_lead = hp_samples[i][0] - hp_samples[i][1]
                if prev_lead * curr_lead < 0:  # Lead swapped
                    lead_changes += 1

            # More lead changes = more exciting
            if lead_changes >= 3:
                scores['comeback_potential'] = 1.0
            elif lead_changes == 2:
                scores['comeback_potential'] = 0.8
            elif lead_changes == 1:
                scores['comeback_potential'] = 0.5
            else:
                scores['comeback_potential'] = 0.2
        else:
            scores['comeback_potential'] = 0.3

        # 5. Positional Exchanges (movement across arena, not wall grinding)
        positions = []
        for tick_data in telemetry.get("ticks", []):
            pos_a = tick_data["fighter_a"]["position"]
            pos_b = tick_data["fighter_b"]["position"]
            positions.append((pos_a, pos_b))

        if positions and len(positions) >= 10:
            swaps = 0
            for i in range(1, len(positions)):
                prev_a, prev_b = positions[i-1]
                curr_a, curr_b = positions[i]
                # Swap = relative position flips
                if (prev_a < prev_b and curr_a > curr_b) or (prev_a > prev_b and curr_a < curr_b):
                    swaps += 1

            swap_rate = swaps / len(positions)
            # Want 5-20% of ticks to have position swaps
            if 0.05 <= swap_rate <= 0.2:
                scores['positional_exchange'] = 1.0
            elif swap_rate > 0.2:
                scores['positional_exchange'] = 0.6  # Too chaotic
            else:
                scores['positional_exchange'] = swap_rate / 0.05  # Scale up to 0.05
        else:
            scores['positional_exchange'] = 0.0

        # 6. Pacing Variety (mix of speeds, not monotonic)
        velocity_samples = []
        for tick_data in telemetry.get("ticks", []):
            # Use relative velocity (closing speed)
            vel_a = tick_data["fighter_a"]["velocity"]
            vel_b = tick_data["fighter_b"]["velocity"]
            relative_vel = abs(vel_a - vel_b)
            velocity_samples.append(relative_vel)

        if velocity_samples and len(velocity_samples) >= 10:
            speeds = [abs(v) for v in velocity_samples]
            avg_speed = sum(speeds) / len(speeds)

            if avg_speed > 0.1:
                variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
                std_dev = variance ** 0.5

                # Normalize: want std_dev around 0.5-1.5 m/s
                if 0.5 <= std_dev <= 1.5:
                    scores['pacing_variety'] = 1.0
                elif std_dev < 0.5:
                    scores['pacing_variety'] = std_dev / 0.5
                else:
                    scores['pacing_variety'] = max(0.3, 1.0 - (std_dev - 1.5) / 2.0)
            else:
                scores['pacing_variety'] = 0.0  # No movement
        else:
            scores['pacing_variety'] = 0.5

        # 7. Collision Drama (want impactful exchanges, not grinding)
        collision_events = [e for e in match_result.events if e["type"] == "COLLISION"]
        collision_count = len(collision_events)

        if collision_events:
            total_damage = sum(e["damage_to_a"] + e["damage_to_b"] for e in collision_events)
            avg_collision_damage = total_damage / (2 * collision_count)
        else:
            avg_collision_damage = 0

        # Want 8-25 collisions with meaningful damage
        if 8 <= collision_count <= 25 and avg_collision_damage > 3.0:
            scores['collision_drama'] = 1.0
        elif collision_count < 8:
            scores['collision_drama'] = collision_count / 8
        elif collision_count > 25:
            scores['collision_drama'] = 0.4  # Wall grinding
        else:
            scores['collision_drama'] = 0.5

        # Calculate weighted overall score
        total_weight = sum(self.weights.values())
        weighted_sum = sum(scores[k] * self.weights[k] for k in scores)
        overall = weighted_sum / total_weight

        return SpectacleScore(
            duration=scores['duration'],
            close_finish=scores['close_finish'],
            stamina_drama=scores['stamina_drama'],
            comeback_potential=scores['comeback_potential'],
            positional_exchange=scores['positional_exchange'],
            pacing_variety=scores['pacing_variety'],
            collision_drama=scores['collision_drama'],
            overall=overall
        )
