"""
Replay Recording System for Training Montage

Records fights during curriculum and population training with spectacle-based sampling.
Saves bottom, middle, and top spectacle scores from each stage for montage creation.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import logging

from ..orchestrator.match_orchestrator import MatchOrchestrator, MatchResult
from ..evaluator.spectacle_evaluator import SpectacleEvaluator, SpectacleScore
from ..telemetry.replay_store import save_replay
from ..arena import WorldConfig


@dataclass
class ReplayMetadata:
    """Metadata for a recorded replay."""
    stage: str  # "curriculum_level_1" or "population_gen_5"
    stage_type: str  # "curriculum" or "population"
    spectacle_score: float
    spectacle_rank: str  # "bottom", "middle", or "top"
    fighter_a: str
    fighter_b: str
    winner: str
    notes: str = ""


class ReplayRecorder:
    """
    Records training fights with spectacle-based sampling.

    Strategy:
    - Run evaluation matches at key points (level graduation, generation milestones)
    - Calculate spectacle scores for all matches
    - Save replays for bottom, middle, and top spectacle scores
    - Store metadata for montage creation
    """

    def __init__(
        self,
        output_dir: str,
        config: WorldConfig = None,
        max_ticks: int = 250,
        samples_per_stage: int = 3,  # bottom, middle, top
        min_matches_for_sampling: int = 5,
        verbose: bool = True
    ):
        """
        Initialize replay recorder.

        Args:
            output_dir: Base directory for saving replays
            config: WorldConfig for matches (default: create new one)
            max_ticks: Maximum ticks per recorded match
            samples_per_stage: Number of samples per stage (default: 3 for bottom/middle/top)
            min_matches_for_sampling: Minimum matches needed to do spectacle sampling
            verbose: Enable logging
        """
        self.output_dir = Path(output_dir)
        self.replays_dir = self.output_dir / "replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or WorldConfig()
        self.max_ticks = max_ticks
        self.samples_per_stage = samples_per_stage
        self.min_matches_for_sampling = min_matches_for_sampling
        self.verbose = verbose

        # Create orchestrator and evaluator
        self.orchestrator = MatchOrchestrator(
            config=self.config,
            max_ticks=max_ticks,
            record_telemetry=True
        )
        self.spectacle_evaluator = SpectacleEvaluator()

        # Setup logging
        self.logger = logging.getLogger('replay_recorder')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Track saved replays
        self.replay_index: List[ReplayMetadata] = []

    def record_curriculum_stage(
        self,
        stage_name: str,
        level_num: int,
        model,
        opponent_paths: List[str],
        num_matches_per_opponent: int = 3
    ):
        """
        Record sample fights from a curriculum training stage.

        Args:
            stage_name: Name of the curriculum stage (e.g., "Fundamentals")
            level_num: Level number (1-indexed)
            model: Trained RL model
            opponent_paths: List of opponent script paths
            num_matches_per_opponent: Matches to run per opponent
        """
        stage_id = f"curriculum_level_{level_num}_{stage_name.lower().replace(' ', '_')}"

        if self.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎬 RECORDING REPLAYS: {stage_name} (Level {level_num})")
            self.logger.info(f"{'='*60}")

        # Run evaluation matches
        matches_with_scores = self._run_curriculum_eval_matches(
            model=model,
            opponent_paths=opponent_paths,
            num_matches_per_opponent=num_matches_per_opponent,
            stage_name=stage_name
        )

        if not matches_with_scores:
            self.logger.warning(f"No matches recorded for {stage_name}")
            return

        # Sample and save replays
        self._save_sampled_replays(
            matches_with_scores=matches_with_scores,
            stage_id=stage_id,
            stage_type="curriculum"
        )

    def record_population_generation(
        self,
        generation: int,
        fighters: List[Any],  # List of PopulationFighter objects
        num_matches_per_pair: int = 2
    ):
        """
        Record sample fights from a population training generation.

        Args:
            generation: Generation number
            fighters: List of PopulationFighter objects with trained models
            num_matches_per_pair: Matches to run per fighter pair
        """
        stage_id = f"population_gen_{generation}"

        if self.verbose:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎬 RECORDING REPLAYS: Generation {generation}")
            self.logger.info(f"{'='*60}")

        # Run evaluation matches
        matches_with_scores = self._run_population_eval_matches(
            fighters=fighters,
            num_matches_per_pair=num_matches_per_pair,
            generation=generation
        )

        if not matches_with_scores:
            self.logger.warning(f"No matches recorded for generation {generation}")
            return

        # Sample and save replays
        self._save_sampled_replays(
            matches_with_scores=matches_with_scores,
            stage_id=stage_id,
            stage_type="population"
        )

    def _run_curriculum_eval_matches(
        self,
        model,
        opponent_paths: List[str],
        num_matches_per_opponent: int,
        stage_name: str
    ) -> List[Tuple[MatchResult, SpectacleScore, str, str]]:
        """Run evaluation matches and calculate spectacle scores."""
        matches_with_scores = []

        # Import opponent loader
        import importlib.util

        for opp_path in opponent_paths:
            opp_name = Path(opp_path).stem

            # Load opponent function
            try:
                if not Path(opp_path).exists():
                    self.logger.warning(f"Opponent file not found: {opp_path}")
                    continue

                spec = importlib.util.spec_from_file_location("opponent", opp_path)
                if spec is None:
                    self.logger.warning(f"Failed to create spec for {opp_path}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, 'decide'):
                    self.logger.warning(f"No decide function in {opp_path}")
                    continue

                opponent_func = module.decide
            except Exception as e:
                self.logger.warning(f"Failed to load opponent {opp_path}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                continue

            # Run matches
            for match_num in range(num_matches_per_opponent):
                # Create model decision function
                def model_decide(snapshot):
                    # Convert snapshot to observation format
                    obs = self._snapshot_to_obs(snapshot)
                    if model is None or not hasattr(model, 'predict'):
                        # Fallback to random action if model unavailable
                        action = np.array([0.0, 0])  # No movement, neutral stance
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    return self._action_to_dict(action)

                # Run match
                result = self.orchestrator.run_match(
                    fighter_a_spec={"name": "AI_Fighter", "mass": 70.0, "position": 3.0},
                    fighter_b_spec={"name": opp_name, "mass": 70.0, "position": 7.0},
                    decision_func_a=model_decide,
                    decision_func_b=opponent_func,
                    seed=42 + match_num
                )

                # Calculate spectacle score
                spectacle_score = self.spectacle_evaluator.evaluate(
                    telemetry=result.telemetry,
                    match_result=result
                )

                matches_with_scores.append((result, spectacle_score, "AI_Fighter", opp_name))

        if self.verbose:
            self.logger.info(f"Ran {len(matches_with_scores)} evaluation matches")

        return matches_with_scores

    def _run_population_eval_matches(
        self,
        fighters: List[Any],
        num_matches_per_pair: int,
        generation: int
    ) -> List[Tuple[MatchResult, SpectacleScore, str, str]]:
        """Run population evaluation matches and calculate spectacle scores."""
        matches_with_scores = []

        # Run matches between different fighters
        for i in range(len(fighters)):
            for j in range(i + 1, len(fighters)):
                fighter_a = fighters[i]
                fighter_b = fighters[j]

                for match_num in range(num_matches_per_pair):
                    # Create decision functions
                    def decide_a(snapshot):
                        obs = self._snapshot_to_obs(snapshot)
                        action, _ = fighter_a.model.predict(obs, deterministic=True)
                        return self._action_to_dict(action)

                    def decide_b(snapshot):
                        obs = self._snapshot_to_obs(snapshot)
                        action, _ = fighter_b.model.predict(obs, deterministic=True)
                        return self._action_to_dict(action)

                    # Run match
                    result = self.orchestrator.run_match(
                        fighter_a_spec={"name": fighter_a.name, "mass": fighter_a.mass, "position": 3.0},
                        fighter_b_spec={"name": fighter_b.name, "mass": fighter_b.mass, "position": 7.0},
                        decision_func_a=decide_a,
                        decision_func_b=decide_b,
                        seed=42 + match_num
                    )

                    # Calculate spectacle score
                    spectacle_score = self.spectacle_evaluator.evaluate(
                        telemetry=result.telemetry,
                        match_result=result
                    )

                    matches_with_scores.append((result, spectacle_score, fighter_a.name, fighter_b.name))

        if self.verbose:
            self.logger.info(f"Ran {len(matches_with_scores)} evaluation matches")

        return matches_with_scores

    def _save_sampled_replays(
        self,
        matches_with_scores: List[Tuple[MatchResult, SpectacleScore, str, str]],
        stage_id: str,
        stage_type: str
    ):
        """Save replays sampled from bottom, middle, and top spectacle scores."""
        if len(matches_with_scores) < self.min_matches_for_sampling:
            self.logger.warning(
                f"Only {len(matches_with_scores)} matches - need at least "
                f"{self.min_matches_for_sampling} for spectacle sampling"
            )
            return

        # Sort by spectacle score
        sorted_matches = sorted(matches_with_scores, key=lambda x: x[1].overall)

        # Calculate indices for bottom, middle, top samples
        n = len(sorted_matches)

        # Bottom third
        bottom_idx = n // 6  # 1/6 point (middle of bottom third)
        # Middle third
        middle_idx = n // 2  # Median
        # Top third
        top_idx = (5 * n) // 6  # 5/6 point (middle of top third)

        samples = [
            (sorted_matches[bottom_idx], "bottom"),
            (sorted_matches[middle_idx], "middle"),
            (sorted_matches[top_idx], "top")
        ]

        # Save sampled replays
        for (result, spectacle_score, fighter_a, fighter_b), rank in samples:
            # Create filename
            filename = f"{stage_id}_{rank}_spectacle_{spectacle_score.overall:.3f}.json.gz"
            filepath = str(self.replays_dir / filename)

            # Create metadata
            metadata = {
                "stage": stage_id,
                "stage_type": stage_type,
                "spectacle_score": spectacle_score.overall,
                "spectacle_breakdown": spectacle_score.to_dict(),
                "spectacle_rank": rank,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "notes": f"Sampled {rank} spectacle from {len(matches_with_scores)} matches"
            }

            # Save replay
            save_replay(
                telemetry=result.telemetry,
                match_result=result,
                filepath=filepath,
                compress=True,
                metadata=metadata
            )

            # Track in index
            self.replay_index.append(ReplayMetadata(
                stage=stage_id,
                stage_type=stage_type,
                spectacle_score=spectacle_score.overall,
                spectacle_rank=rank,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                winner=result.winner,
                notes=metadata["notes"]
            ))

            if self.verbose:
                self.logger.info(
                    f"  💾 Saved {rank:6s} spectacle ({spectacle_score.overall:.3f}): "
                    f"{fighter_a} vs {fighter_b} → {result.winner}"
                )

        if self.verbose:
            self.logger.info(f"Saved {len(samples)} replays for {stage_id}")

    def _snapshot_to_obs(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """Convert snapshot to enhanced observation vector for RL model."""
        # Match the enhanced observation space (13 values)
        you_hp_norm = snapshot["you"]["hp"] / snapshot["you"]["max_hp"]
        you_stamina_norm = snapshot["you"]["stamina"] / snapshot["you"]["max_stamina"]
        opp_hp_norm = snapshot["opponent"]["hp"] / snapshot["opponent"]["max_hp"]
        opp_stamina_norm = snapshot["opponent"]["stamina"] / snapshot["opponent"]["max_stamina"]

        distance = snapshot["opponent"]["distance"]
        rel_velocity = snapshot["opponent"]["velocity"]  # Already relative in snapshot

        # Get arena width from config
        arena_width = self.config.arena_width if hasattr(self.config, 'arena_width') else 10.0

        # Wall distances
        wall_dist_left = snapshot["you"]["position"]
        wall_dist_right = arena_width - snapshot["you"]["position"]

        # Opponent stance as integer (0=neutral, 1=extended, 2=defending)
        opp_stance_hint = snapshot["opponent"].get("stance_hint", "neutral")
        stance_map = {"neutral": 0, "extended": 1, "defending": 2}
        opp_stance_int = stance_map.get(opp_stance_hint, 0)

        # Recent damage (placeholder - would need tracking)
        recent_damage = 0.0

        return np.array([
            snapshot["you"]["position"],  # 0: position
            snapshot["you"]["velocity"],  # 1: velocity
            you_hp_norm,                   # 2: hp_norm
            you_stamina_norm,              # 3: stamina_norm
            distance,                      # 4: distance
            rel_velocity,                  # 5: rel_velocity
            opp_hp_norm,                   # 6: opp_hp_norm
            opp_stamina_norm,              # 7: opp_stamina_norm
            arena_width,                   # 8: arena_width
            wall_dist_left,                # 9: wall_dist_left
            wall_dist_right,               # 10: wall_dist_right
            opp_stance_int,                # 11: opp_stance
            recent_damage                  # 12: recent_damage
        ], dtype=np.float32)

    def _action_to_dict(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert RL model action to decision dict."""
        # Match the action space from AtomCombatEnv
        # action[0]: acceleration (-1 to 1)
        # action[1]: stance (0=neutral, 1=extended, 2=defending)

        acceleration = float(np.clip(action[0], -1.0, 1.0))

        stance_idx = int(np.clip(action[1], 0, 2))
        stance_map = ["neutral", "extended", "defending"]
        stance = stance_map[stance_idx]

        return {
            "acceleration": acceleration,
            "stance": stance
        }

    def save_replay_index(self):
        """Save index of all recorded replays to JSON."""
        import json

        index_path = self.output_dir / "replay_index.json"

        index_data = {
            "total_replays": len(self.replay_index),
            "replays": [
                {
                    "stage": r.stage,
                    "stage_type": r.stage_type,
                    "spectacle_score": r.spectacle_score,
                    "spectacle_rank": r.spectacle_rank,
                    "fighter_a": r.fighter_a,
                    "fighter_b": r.fighter_b,
                    "winner": r.winner,
                    "notes": r.notes
                }
                for r in self.replay_index
            ]
        }

        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        if self.verbose:
            self.logger.info(f"\n📋 Saved replay index to: {index_path}")
            self.logger.info(f"   Total replays: {len(self.replay_index)}")
