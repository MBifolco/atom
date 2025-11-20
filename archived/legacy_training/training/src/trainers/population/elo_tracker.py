"""
ELO Rating System for Population-Based Training

Tracks fighter performance using the ELO rating system.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FighterStats:
    """Statistics for a fighter in the population."""
    name: str
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    match_history: List[dict] = field(default_factory=list)

    @property
    def matches_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def damage_ratio(self) -> float:
        if self.total_damage_taken == 0:
            return float('inf') if self.total_damage_dealt > 0 else 1.0
        return self.total_damage_dealt / self.total_damage_taken


class EloTracker:
    """
    Tracks ELO ratings and statistics for population-based training.

    The ELO system provides a way to rank fighters based on their
    relative performance against each other.
    """

    def __init__(self, k_factor: float = 32.0, initial_elo: float = 1500.0):
        """
        Initialize the ELO tracker.

        Args:
            k_factor: The K-factor determines how much ratings change after each match
            initial_elo: Starting ELO rating for new fighters
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.fighters: Dict[str, FighterStats] = {}

    def add_fighter(self, name: str) -> None:
        """Add a new fighter to the tracker."""
        if name not in self.fighters:
            self.fighters[name] = FighterStats(name=name, elo=self.initial_elo)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for fighter A against fighter B.

        Returns probability of A winning (0 to 1).
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self,
                       fighter_a: str,
                       fighter_b: str,
                       result: str,
                       damage_a: float = 0,
                       damage_b: float = 0,
                       match_info: dict = None) -> Tuple[float, float]:
        """
        Update ELO ratings after a match.

        Args:
            fighter_a: Name of fighter A
            fighter_b: Name of fighter B
            result: "a_wins", "b_wins", or "draw"
            damage_a: Damage dealt by fighter A
            damage_b: Damage dealt by fighter B
            match_info: Additional match information to store

        Returns:
            Tuple of (new_elo_a, new_elo_b)
        """
        # Ensure fighters exist
        self.add_fighter(fighter_a)
        self.add_fighter(fighter_b)

        stats_a = self.fighters[fighter_a]
        stats_b = self.fighters[fighter_b]

        # Calculate expected scores
        expected_a = self.expected_score(stats_a.elo, stats_b.elo)
        expected_b = self.expected_score(stats_b.elo, stats_a.elo)

        # Determine actual scores
        if result == "a_wins":
            score_a, score_b = 1.0, 0.0
            stats_a.wins += 1
            stats_b.losses += 1
        elif result == "b_wins":
            score_a, score_b = 0.0, 1.0
            stats_a.losses += 1
            stats_b.wins += 1
        else:  # draw
            score_a, score_b = 0.5, 0.5
            stats_a.draws += 1
            stats_b.draws += 1

        # Update ELO ratings
        new_elo_a = stats_a.elo + self.k_factor * (score_a - expected_a)
        new_elo_b = stats_b.elo + self.k_factor * (score_b - expected_b)

        stats_a.elo = new_elo_a
        stats_b.elo = new_elo_b

        # Update damage statistics
        stats_a.total_damage_dealt += damage_a
        stats_a.total_damage_taken += damage_b
        stats_b.total_damage_dealt += damage_b
        stats_b.total_damage_taken += damage_a

        # Record match in history
        if match_info is None:
            match_info = {}

        match_record = {
            "timestamp": datetime.now().isoformat(),
            "opponent": fighter_b,
            "result": result,
            "elo_before": stats_a.elo - self.k_factor * (score_a - expected_a),
            "elo_after": stats_a.elo,
            "damage_dealt": damage_a,
            "damage_taken": damage_b,
            **match_info
        }
        stats_a.match_history.append(match_record)

        match_record_b = {
            "timestamp": datetime.now().isoformat(),
            "opponent": fighter_a,
            "result": "b_wins" if result == "a_wins" else ("a_wins" if result == "b_wins" else "draw"),
            "elo_before": stats_b.elo - self.k_factor * (score_b - expected_b),
            "elo_after": stats_b.elo,
            "damage_dealt": damage_b,
            "damage_taken": damage_a,
            **match_info
        }
        stats_b.match_history.append(match_record_b)

        return new_elo_a, new_elo_b

    def get_rankings(self) -> List[FighterStats]:
        """Get fighters sorted by ELO rating."""
        return sorted(self.fighters.values(), key=lambda x: x.elo, reverse=True)

    def get_matchup_prediction(self, fighter_a: str, fighter_b: str) -> dict:
        """
        Get prediction for a matchup between two fighters.

        Returns dict with win probabilities and expected outcome.
        """
        self.add_fighter(fighter_a)
        self.add_fighter(fighter_b)

        elo_a = self.fighters[fighter_a].elo
        elo_b = self.fighters[fighter_b].elo

        prob_a = self.expected_score(elo_a, elo_b)
        prob_b = 1 - prob_a

        return {
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "win_prob_a": prob_a,
            "win_prob_b": prob_b,
            "elo_diff": elo_a - elo_b,
            "favorite": fighter_a if prob_a > 0.5 else fighter_b
        }

    def suggest_balanced_matches(self, num_matches: int = 4) -> List[Tuple[str, str]]:
        """
        Suggest balanced matches based on similar ELO ratings.

        Returns list of (fighter_a, fighter_b) tuples.
        """
        if len(self.fighters) < 2:
            return []

        # Sort fighters by ELO
        ranked = self.get_rankings()
        matches = []
        used_fighters = set()

        # Pair adjacent fighters in ranking (similar skill)
        for i in range(0, min(len(ranked) - 1, num_matches * 2), 2):
            if ranked[i].name not in used_fighters and ranked[i+1].name not in used_fighters:
                matches.append((ranked[i].name, ranked[i+1].name))
                used_fighters.add(ranked[i].name)
                used_fighters.add(ranked[i+1].name)

                if len(matches) >= num_matches:
                    break

        return matches

    def get_diversity_metrics(self) -> dict:
        """
        Calculate diversity metrics for the population.

        Returns metrics about ELO spread, win rate variance, etc.
        """
        if not self.fighters:
            return {}

        elos = [f.elo for f in self.fighters.values()]
        win_rates = [f.win_rate for f in self.fighters.values() if f.matches_played > 0]
        damage_ratios = [f.damage_ratio for f in self.fighters.values() if f.matches_played > 0]

        metrics = {
            "population_size": len(self.fighters),
            "elo_mean": np.mean(elos),
            "elo_std": np.std(elos),
            "elo_range": max(elos) - min(elos),
            "elo_top": max(elos),
            "elo_bottom": min(elos),
        }

        if win_rates:
            metrics["win_rate_mean"] = np.mean(win_rates)
            metrics["win_rate_std"] = np.std(win_rates)

        if damage_ratios:
            valid_ratios = [r for r in damage_ratios if r != float('inf')]
            if valid_ratios:
                metrics["damage_ratio_mean"] = np.mean(valid_ratios)
                metrics["damage_ratio_std"] = np.std(valid_ratios)

        return metrics

    def print_leaderboard(self, top_n: int = None) -> None:
        """Print a formatted leaderboard."""
        rankings = self.get_rankings()

        if top_n:
            rankings = rankings[:top_n]

        print("\n" + "="*80)
        print("POPULATION LEADERBOARD")
        print("="*80)
        print(f"{'Rank':<6} {'Fighter':<20} {'ELO':<8} {'W-L-D':<12} {'Win%':<8} {'DMG Ratio':<10}")
        print("-"*80)

        for i, fighter in enumerate(rankings, 1):
            record = f"{fighter.wins}-{fighter.losses}-{fighter.draws}"
            win_pct = f"{fighter.win_rate:.1%}" if fighter.matches_played > 0 else "N/A"
            dmg_ratio = f"{fighter.damage_ratio:.2f}" if fighter.damage_ratio != float('inf') else "∞"

            print(f"{i:<6} {fighter.name:<20} {fighter.elo:<8.0f} {record:<12} {win_pct:<8} {dmg_ratio:<10}")

        print("\n" + "="*80)

        # Print diversity metrics
        metrics = self.get_diversity_metrics()
        if metrics:
            print("POPULATION DIVERSITY")
            print("-"*80)
            print(f"Population Size: {metrics['population_size']}")
            print(f"ELO Range: {metrics['elo_range']:.0f} ({metrics['elo_bottom']:.0f} - {metrics['elo_top']:.0f})")
            print(f"ELO Std Dev: {metrics['elo_std']:.1f}")
            if 'win_rate_std' in metrics:
                print(f"Win Rate Variance: {metrics['win_rate_std']:.3f}")
        print("="*80)