"""
Progressive Replay Recorder for Training Montage

Records fights throughout training to show learning progression,
not just at key milestones. Captures the journey from novice to expert.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from datetime import datetime

from src.atom.runtime.orchestrator import MatchOrchestrator, MatchResult
from src.atom.runtime.telemetry import save_replay
from src.atom.runtime.arena import WorldConfig


@dataclass
class ProgressiveReplayMetadata:
    """Metadata for a progressive replay."""
    level_name: str
    level_num: int
    episode: int
    total_episodes: int
    win_rate: float
    recent_rewards: List[float]
    timestamp: str
    fighter_a: str
    fighter_b: str
    winner: str
    duration_ticks: int
    notes: str = ""


class ProgressiveReplayRecorder:
    """
    Records training fights progressively to show learning journey.

    Unlike spectacle-based sampling, this records fights throughout
    training to capture the progression from failure to mastery.
    """

    def __init__(
        self,
        output_dir: str,
        max_ticks: int = 250,
        verbose: bool = True
    ):
        """
        Initialize progressive replay recorder.

        Args:
            output_dir: Base directory for saving replays
            max_ticks: Maximum ticks per recorded match
            verbose: Enable logging
        """
        self.output_dir = Path(output_dir)
        self.replays_dir = self.output_dir / "progressive_replays"
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        self.max_ticks = max_ticks
        self.verbose = verbose

        # Recording strategy parameters
        self.early_phase_interval = 25    # Record every 25 episodes in early phase
        self.mid_phase_interval = 50      # Record every 50 episodes in mid phase
        self.late_phase_interval = 100    # Record every 100 episodes in late phase

        # Phase boundaries (as fraction of total episodes)
        self.early_phase_end = 0.2   # First 20% is early phase
        self.mid_phase_end = 0.8     # 20-80% is mid phase

        # Track recorded replays
        self.replay_index: List[ProgressiveReplayMetadata] = []

        # Setup logging
        self.logger = logging.getLogger('progressive_replay_recorder')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def should_record(
        self,
        episode: int,
        total_episodes: int,
        force_record: bool = False
    ) -> bool:
        """
        Determine if we should record this episode.

        Args:
            episode: Current episode number
            total_episodes: Total episodes in this level
            force_record: Force recording regardless of strategy

        Returns:
            True if we should record this episode
        """
        if force_record:
            return True

        # Always record first episode
        if episode <= 1:
            return True

        # For episode-based recording, use absolute episode counts
        # This prevents issues when episode count exceeds expected total
        if episode < 200:
            # Early phase: every 25 episodes
            return episode % self.early_phase_interval == 0
        elif episode < 1000:
            # Mid phase: every 50 episodes
            return episode % self.mid_phase_interval == 0
        else:
            # Late phase: every 100 episodes
            return episode % self.late_phase_interval == 0

    def record_episode_replay(
        self,
        telemetry: Dict[str, Any],
        match_result: MatchResult,
        level_name: str,
        level_num: int,
        episode: int,
        total_episodes: int,
        win_rate: float,
        recent_rewards: List[float],
        fighter_a_name: str = "AI_Fighter",
        fighter_b_name: str = "Opponent"
    ):
        """
        Record a single episode's replay with progression metadata.

        Args:
            telemetry: Match telemetry data
            match_result: Match result object
            level_name: Name of the current level
            level_num: Level number (1-indexed)
            episode: Current episode number
            total_episodes: Total episodes planned for this level
            win_rate: Current win rate (0-1)
            recent_rewards: Recent reward history
            fighter_a_name: Name of the learning fighter
            fighter_b_name: Name of the opponent
        """
        # Create filename with episode info
        filename = (
            f"level_{level_num}_{level_name.lower().replace(' ', '_')}_"
            f"ep_{episode:05d}_wr_{int(win_rate*100):03d}.json.gz"
        )
        filepath = str(self.replays_dir / filename)

        # Create metadata
        metadata = ProgressiveReplayMetadata(
            level_name=level_name,
            level_num=level_num,
            episode=episode,
            total_episodes=total_episodes,
            win_rate=win_rate,
            recent_rewards=recent_rewards[-10:] if recent_rewards else [],
            timestamp=datetime.now().isoformat(),
            fighter_a=fighter_a_name,
            fighter_b=fighter_b_name,
            winner=match_result.winner,
            duration_ticks=match_result.total_ticks,
            notes=self._generate_progress_note(episode, total_episodes, win_rate)
        )

        # Check if telemetry is empty
        if not telemetry or len(telemetry.get('ticks', [])) == 0:
            if self.verbose:
                self.logger.warning(f"  ⚠️  Skipping save - empty telemetry (0 ticks)")
            return

        # Save replay with metadata
        save_replay(
            telemetry=telemetry,
            match_result=match_result,
            filepath=filepath,
            compress=True,
            metadata=metadata.__dict__
        )

        # Track in index
        self.replay_index.append(metadata)

        if self.verbose:
            progress_pct = (episode / total_episodes) * 100
            self.logger.info(
                f"  ✅ Saved replay: {filename} | Episode {episode}/{total_episodes} "
                f"({progress_pct:.1f}%) | {match_result.total_ticks} ticks | Winner: {match_result.winner}"
            )

    def _generate_progress_note(
        self,
        episode: int,
        total_episodes: int,
        win_rate: float
    ) -> str:
        """Generate a descriptive note about training progress."""
        progress = episode / total_episodes

        if progress < 0.1:
            phase = "Initial exploration"
        elif progress < 0.3:
            phase = "Early learning"
        elif progress < 0.5:
            phase = "Skill development"
        elif progress < 0.7:
            phase = "Refinement"
        elif progress < 0.9:
            phase = "Mastery"
        else:
            phase = "Graduation preparation"

        performance = ""
        if win_rate < 0.2:
            performance = "struggling"
        elif win_rate < 0.4:
            performance = "improving"
        elif win_rate < 0.6:
            performance = "competitive"
        elif win_rate < 0.8:
            performance = "strong"
        else:
            performance = "dominant"

        return f"{phase} - {performance} ({win_rate:.1%} win rate)"

    def save_progressive_index(self):
        """Save index of all progressive replays to JSON."""
        index_path = self.output_dir / "progressive_replay_index.json"

        index_data = {
            "total_replays": len(self.replay_index),
            "recording_strategy": {
                "early_phase_interval": self.early_phase_interval,
                "mid_phase_interval": self.mid_phase_interval,
                "late_phase_interval": self.late_phase_interval,
                "early_phase_end": self.early_phase_end,
                "mid_phase_end": self.mid_phase_end
            },
            "replays": [
                {
                    "level_name": r.level_name,
                    "level_num": r.level_num,
                    "episode": r.episode,
                    "total_episodes": r.total_episodes,
                    "win_rate": r.win_rate,
                    "recent_rewards": r.recent_rewards,
                    "timestamp": r.timestamp,
                    "fighter_a": r.fighter_a,
                    "fighter_b": r.fighter_b,
                    "winner": r.winner,
                    "duration_ticks": r.duration_ticks,
                    "notes": r.notes
                }
                for r in self.replay_index
            ]
        }

        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        if self.verbose:
            self.logger.info(f"\n📋 Saved progressive replay index to: {index_path}")
            self.logger.info(f"   Total replays: {len(self.replay_index)}")

            # Show summary by level
            levels = {}
            for replay in self.replay_index:
                if replay.level_num not in levels:
                    levels[replay.level_num] = []
                levels[replay.level_num].append(replay)

            for level_num in sorted(levels.keys()):
                level_replays = levels[level_num]
                self.logger.info(
                    f"   Level {level_num}: {len(level_replays)} replays "
                    f"(episodes {level_replays[0].episode}-{level_replays[-1].episode})"
                )