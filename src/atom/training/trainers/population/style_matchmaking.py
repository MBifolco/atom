"""
Style fingerprint and diversity-based opponent matchmaking for population training.

Characterizes each fighter's behavioral style as a numeric vector and assigns
opponents that maximize style diversity within each generation. This breaks the
mirrored-training-dynamics problem where paired fighters develop identical policies.

Usage:
    fingerprinter = StyleFingerprinter()
    matchmaker = DiversityMatchmaker(DiversityMatchmakingContext())

    # After evaluation round-robin:
    fingerprints = fingerprinter.compute_fingerprints(eval_stats, max_ticks, names)

    # After evolution, inherit for new children:
    fingerprinter.inherit_for_children(fingerprints, lineage_events)

    # Before training:
    pairs = matchmaker.assign_opponents(population, fingerprints, elo_tracker)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Style fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StyleFingerprint:
    """Behavioral style vector for a fighter.

    All numeric dimensions are pre-normalized to [0, 1]. The ``source`` field
    tracks provenance (evaluation data, inherited from parent, or default)
    and is excluded from distance computations.
    """

    name: str
    stance_neutral_pct: float
    stance_extended_pct: float
    stance_defending_pct: float
    damage_efficiency: float      # dealt / (dealt + taken), 0.5 when both 0
    avg_fight_length_pct: float   # avg_ticks / max_ticks
    source: Literal["evaluation", "inherited", "default"] = "evaluation"

    _NUM_DIMENSIONS: int = 5  # class-level constant for normalization

    def to_vector(self) -> np.ndarray:
        """Return the 5D style vector (excludes source)."""
        return np.array([
            self.stance_neutral_pct,
            self.stance_extended_pct,
            self.stance_defending_pct,
            self.damage_efficiency,
            self.avg_fight_length_pct,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fingerprinter
# ---------------------------------------------------------------------------

_DEFAULT_FINGERPRINT_VALUES = dict(
    stance_neutral_pct=1.0 / 3.0,
    stance_extended_pct=1.0 / 3.0,
    stance_defending_pct=1.0 / 3.0,
    damage_efficiency=0.5,
    avg_fight_length_pct=0.5,
)


class StyleFingerprinter:
    """Computes style fingerprints from evaluation data.

    Stateless — all inputs passed via method arguments.
    """

    def compute_fingerprints(
        self,
        evaluation_stats: Dict[str, Any],
        max_ticks: int,
        active_fighter_names: List[str],
    ) -> Dict[str, StyleFingerprint]:
        """Build fingerprints from per-fighter evaluation stats.

        Args:
            evaluation_stats: name -> PerFighterEvaluationStats mapping
            max_ticks: episode length cap (for normalizing fight length)
            active_fighter_names: only include these fighters

        Returns:
            Dict of name -> StyleFingerprint. Empty if no stats.
        """
        result: Dict[str, StyleFingerprint] = {}
        active_set = set(active_fighter_names)

        for name, stats in evaluation_stats.items():
            if name not in active_set:
                continue

            total_ticks = stats.total_ticks
            if total_ticks <= 0:
                continue

            stance = stats.total_stance_ticks
            dealt = stats.total_damage_dealt
            taken = stats.total_damage_taken
            total_damage = dealt + taken

            result[name] = StyleFingerprint(
                name=name,
                stance_neutral_pct=stance.get("neutral", 0) / total_ticks,
                stance_extended_pct=stance.get("extended", 0) / total_ticks,
                stance_defending_pct=stance.get("defending", 0) / total_ticks,
                damage_efficiency=dealt / total_damage if total_damage > 0 else 0.5,
                avg_fight_length_pct=min(1.0, (total_ticks / stats.match_count) / max_ticks)
                    if stats.match_count > 0 else 0.5,
                source="evaluation",
            )

        return result

    def inherit_for_children(
        self,
        fingerprints: Dict[str, StyleFingerprint],
        lineage_events: List[Any],
    ) -> None:
        """Update fingerprint map after evolution: inherit + prune.

        Modifies ``fingerprints`` in place:
        1. Removes replaced fighters.
        2. Copies parent fingerprint to child (source="inherited").
        3. Falls back to neutral default if parent fingerprint is missing.
        """
        for event in lineage_events:
            # Prune replaced fighter
            replaced = event.replaced_fighter_name
            fingerprints.pop(replaced, None)

            # Inherit from parent
            parent_fp = fingerprints.get(event.parent_name)
            if parent_fp is not None:
                fingerprints[event.child_name] = StyleFingerprint(
                    name=event.child_name,
                    stance_neutral_pct=parent_fp.stance_neutral_pct,
                    stance_extended_pct=parent_fp.stance_extended_pct,
                    stance_defending_pct=parent_fp.stance_defending_pct,
                    damage_efficiency=parent_fp.damage_efficiency,
                    avg_fight_length_pct=parent_fp.avg_fight_length_pct,
                    source="inherited",
                )
            else:
                fingerprints[event.child_name] = StyleFingerprint(
                    name=event.child_name,
                    source="default",
                    **_DEFAULT_FINGERPRINT_VALUES,
                )


# ---------------------------------------------------------------------------
# Diversity matchmaker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiversityMatchmakingContext:
    """Configuration for diversity-based opponent assignment."""

    opponents_per_fighter: int = 3
    diversity_weight: float = 0.7    # [0, 1] blend of style vs skill
    elo_band: float = 200.0          # normalizes ELO diff to [0, 1]


class DiversityMatchmaker:
    """Assigns multiple diverse opponents per fighter using style fingerprints."""

    def __init__(self, context: DiversityMatchmakingContext):
        self.context = context

    def assign_opponents(
        self,
        population: List[Any],
        fingerprints: Dict[str, StyleFingerprint],
        elo_tracker: Any,
    ) -> List[Tuple[Any, List[Any]]]:
        """Produce (fighter, [opponents]) pairs for the entire population.

        Uses a composite score blending style diversity and ELO proximity.
        Children (source="inherited") are guaranteed at least 1 incumbent
        (source="evaluation") opponent when available.
        """
        n = len(population)
        opf = min(self.context.opponents_per_fighter, n - 1)
        if opf <= 0:
            return [(f, []) for f in population]

        # Build lookup tables
        name_to_fighter = {f.name: f for f in population}
        elo_map = self._build_elo_map(elo_tracker, population)

        # Check if any incumbents exist (for mixing rule)
        incumbent_names = {
            name for name, fp in fingerprints.items()
            if fp.source == "evaluation"
        }

        result: List[Tuple[Any, List[Any]]] = []
        for fighter in population:
            fp = fingerprints.get(fighter.name)
            if fp is None:
                # Shouldn't happen if lifecycle is correct, but be defensive
                opponents = self._random_opponents(fighter, population, opf)
                result.append((fighter, opponents))
                continue

            # Score all candidates
            scores: List[Tuple[str, float]] = []
            for other in population:
                if other.name == fighter.name:
                    continue
                other_fp = fingerprints.get(other.name)
                if other_fp is None:
                    scores.append((other.name, 0.0))
                    continue

                style_dist = self._style_distance(fp, other_fp)
                elo_prox = self._elo_proximity(
                    elo_map.get(fighter.name, 1500.0),
                    elo_map.get(other.name, 1500.0),
                )
                score = (
                    self.context.diversity_weight * style_dist
                    + (1.0 - self.context.diversity_weight) * elo_prox
                )
                scores.append((other.name, score))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Select top opponents
            selected_names = [name for name, _ in scores[:opf]]

            # Child/incumbent mixing rule
            if (
                fp.source in ("inherited", "default")
                and incumbent_names
                and not any(n in incumbent_names for n in selected_names)
            ):
                # Swap lowest-scoring selection with the best-scoring incumbent
                best_incumbent = next(
                    (name for name, _ in scores if name in incumbent_names),
                    None,
                )
                if best_incumbent and best_incumbent not in selected_names:
                    selected_names[-1] = best_incumbent

            opponents = [name_to_fighter[n] for n in selected_names if n in name_to_fighter]
            result.append((fighter, opponents))

        return result

    def assign_random_opponents(
        self,
        population: List[Any],
        opponents_per_fighter: int = 3,
    ) -> List[Tuple[Any, List[Any]]]:
        """Gen 0 fallback: assign random opponents (no fingerprints needed)."""
        opf = min(opponents_per_fighter, len(population) - 1)
        result: List[Tuple[Any, List[Any]]] = []
        for fighter in population:
            candidates = [f for f in population if f.name != fighter.name]
            opponents = random.sample(candidates, opf)
            result.append((fighter, opponents))
        return result

    def _style_distance(self, a: StyleFingerprint, b: StyleFingerprint) -> float:
        """Euclidean distance normalized to [0, 1]."""
        va = a.to_vector()
        vb = b.to_vector()
        raw_dist = float(np.linalg.norm(va - vb))
        max_dist = math.sqrt(StyleFingerprint._NUM_DIMENSIONS)
        return min(1.0, raw_dist / max_dist) if max_dist > 0 else 0.0

    def _elo_proximity(self, elo_a: float, elo_b: float) -> float:
        """ELO closeness score in [0, 1]. Higher = closer in skill."""
        return 1.0 - min(abs(elo_a - elo_b) / self.context.elo_band, 1.0)

    def _build_elo_map(self, elo_tracker: Any, population: List[Any]) -> Dict[str, float]:
        """Extract name -> ELO mapping from tracker."""
        elo_map: Dict[str, float] = {}
        if hasattr(elo_tracker, "fighters"):
            for name, stats in elo_tracker.fighters.items():
                elo_map[name] = stats.elo
        # Ensure all population members have an entry
        for f in population:
            elo_map.setdefault(f.name, 1500.0)
        return elo_map

    def _random_opponents(
        self, fighter: Any, population: List[Any], count: int,
    ) -> List[Any]:
        """Pick random opponents excluding self."""
        candidates = [f for f in population if f.name != fighter.name]
        return random.sample(candidates, min(count, len(candidates)))
