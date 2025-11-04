"""
Atom Combat - Replay Store

Save and load match replays with telemetry.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ReplayStore:
    """
    Manages storage and retrieval of match replays.

    Supports:
    - JSON format (human-readable)
    - Compressed JSON (smaller file size)
    - Metadata indexing
    """

    def __init__(self, replay_dir: str = "replays"):
        """
        Initialize replay store.

        Args:
            replay_dir: Directory to store replays
        """
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        telemetry: Dict[str, Any],
        match_result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save match replay to disk.

        Args:
            telemetry: Full match telemetry from MatchOrchestrator
            match_result: MatchResult object
            metadata: Optional additional metadata
            compress: Whether to gzip compress
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved replay file
        """
        # Build replay data
        replay_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "result": {
                "winner": match_result.winner,
                "total_ticks": match_result.total_ticks,
                "final_hp_a": match_result.final_hp_a,
                "final_hp_b": match_result.final_hp_b,
            },
            "telemetry": telemetry,
            "events": match_result.events,
            "metadata": metadata or {}
        }

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fighter_a = telemetry.get("fighter_a_name", "unknown")
            fighter_b = telemetry.get("fighter_b_name", "unknown")
            ext = ".json.gz" if compress else ".json"
            filename = f"replay_{timestamp}_{fighter_a}_vs_{fighter_b}{ext}"

        filepath = self.replay_dir / filename

        # Save to disk
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(replay_data, f, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(replay_data, f, indent=2)

        return filepath

    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load replay from disk.

        Args:
            filename: Replay filename

        Returns:
            Replay data dictionary
        """
        filepath = self.replay_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Replay not found: {filepath}")

        # Auto-detect compression
        if filename.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filepath, 'r') as f:
                return json.load(f)

    def list_replays(self) -> list:
        """
        List all available replays.

        Returns:
            List of replay filenames
        """
        replays = []
        for filepath in self.replay_dir.glob("replay_*.json*"):
            replays.append(filepath.name)
        return sorted(replays, reverse=True)  # Most recent first

    def get_replay_info(self, filename: str) -> Dict[str, Any]:
        """
        Get replay metadata without loading full telemetry.

        Args:
            filename: Replay filename

        Returns:
            Metadata dictionary
        """
        replay_data = self.load(filename)
        return {
            "filename": filename,
            "timestamp": replay_data.get("timestamp"),
            "fighter_a": replay_data["telemetry"].get("fighter_a_name"),
            "fighter_b": replay_data["telemetry"].get("fighter_b_name"),
            "winner": replay_data["result"]["winner"],
            "total_ticks": replay_data["result"]["total_ticks"],
            "metadata": replay_data.get("metadata", {})
        }


# Convenience functions for single-file operations

def save_replay(
    telemetry: Dict[str, Any],
    match_result: Any,
    filepath: str,
    compress: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a replay to a specific file path.

    Args:
        telemetry: Match telemetry
        match_result: MatchResult object
        filepath: Full path to save file
        compress: Whether to gzip compress
        metadata: Optional metadata
    """
    replay_data = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "result": {
            "winner": match_result.winner,
            "total_ticks": match_result.total_ticks,
            "final_hp_a": match_result.final_hp_a,
            "final_hp_b": match_result.final_hp_b,
        },
        "telemetry": telemetry,
        "events": match_result.events,
        "metadata": metadata or {}
    }

    if compress and not filepath.endswith('.gz'):
        filepath = filepath + '.gz'

    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(replay_data, f, indent=2)
    else:
        with open(filepath, 'w') as f:
            json.dump(replay_data, f, indent=2)


def load_replay(filepath: str) -> Dict[str, Any]:
    """
    Load a replay from a specific file path.

    Args:
        filepath: Path to replay file

    Returns:
        Replay data dictionary
    """
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)
