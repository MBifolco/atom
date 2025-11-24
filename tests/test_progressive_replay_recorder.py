"""
Tests for ProgressiveReplayRecorder
"""

import pytest
import tempfile
from pathlib import Path
import json
import gzip

from src.training.progressive_replay_recorder import ProgressiveReplayRecorder, ProgressiveReplayMetadata
from src.orchestrator.match_orchestrator import MatchResult


class TestProgressiveReplayRecorder:
    """Test progressive replay recording functionality."""

    def test_initialization(self):
        """Test recorder initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(
                output_dir=tmpdir,
                max_ticks=100,
                verbose=False
            )

            assert recorder.output_dir == Path(tmpdir)
            assert recorder.replays_dir.exists()
            assert recorder.max_ticks == 100
            assert recorder.early_phase_interval == 10
            assert recorder.mid_phase_interval == 50
            assert recorder.late_phase_interval == 100

    def test_should_record_strategy(self):
        """Test recording strategy at different phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            total_episodes = 1000

            # Always record first episode
            assert recorder.should_record(1, total_episodes)

            # Always record last episode
            assert recorder.should_record(999, total_episodes)

            # Early phase (0-20%): every 10 episodes
            assert recorder.should_record(10, total_episodes)  # Yes
            assert recorder.should_record(20, total_episodes)  # Yes
            assert not recorder.should_record(15, total_episodes)  # No
            assert not recorder.should_record(25, total_episodes)  # No

            # Mid phase (20-80%): every 50 episodes
            assert recorder.should_record(250, total_episodes)  # Yes (episode 250)
            assert recorder.should_record(300, total_episodes)  # Yes
            assert not recorder.should_record(275, total_episodes)  # No
            assert not recorder.should_record(325, total_episodes)  # No

            # Late phase (80-100%): every 100 episodes
            assert recorder.should_record(900, total_episodes)  # Yes
            assert not recorder.should_record(850, total_episodes)  # No
            assert not recorder.should_record(875, total_episodes)  # No

            # Force record
            assert recorder.should_record(123, total_episodes, force_record=True)

    def test_record_episode_replay(self):
        """Test recording a single episode replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            # Create mock data
            telemetry = {
                "ticks": [
                    {"tick": 0, "fighter_a": {"hp": 100}, "fighter_b": {"hp": 100}},
                    {"tick": 1, "fighter_a": {"hp": 95}, "fighter_b": {"hp": 100}}
                ]
            }

            match_result = MatchResult(
                winner="AI_Fighter",
                total_ticks=50,
                final_hp_a=75.0,
                final_hp_b=0.0,
                telemetry=telemetry,
                events=[]
            )

            # Record a replay
            recorder.record_episode_replay(
                telemetry=telemetry,
                match_result=match_result,
                level_name="Fundamentals",
                level_num=1,
                episode=42,
                total_episodes=1000,
                win_rate=0.35,
                recent_rewards=[10.5, 15.2, -5.0],
                fighter_a_name="AI_Fighter",
                fighter_b_name="Dummy"
            )

            # Check replay was saved
            replay_files = list(recorder.replays_dir.glob("*.json.gz"))
            assert len(replay_files) == 1

            # Check filename format
            filename = replay_files[0].name
            assert "level_1_fundamentals" in filename
            assert "ep_00042" in filename
            assert "wr_035" in filename  # 35% win rate

            # Check replay index
            assert len(recorder.replay_index) == 1
            metadata = recorder.replay_index[0]
            assert metadata.level_name == "Fundamentals"
            assert metadata.level_num == 1
            assert metadata.episode == 42
            assert metadata.win_rate == 0.35
            assert metadata.winner == "AI_Fighter"

    def test_progress_notes(self):
        """Test progress note generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            # Early phase, low win rate
            note = recorder._generate_progress_note(50, 1000, 0.15)
            assert "Initial exploration" in note
            assert "struggling" in note

            # Mid phase, competitive
            note = recorder._generate_progress_note(400, 1000, 0.45)
            assert "Skill development" in note
            assert "competitive" in note  # 45% is competitive, not improving

            # Late phase, strong performance
            note = recorder._generate_progress_note(850, 1000, 0.75)
            assert "Mastery" in note
            assert "strong" in note

            # Graduation, dominant
            note = recorder._generate_progress_note(950, 1000, 0.85)
            assert "Graduation preparation" in note
            assert "dominant" in note

    def test_save_progressive_index(self):
        """Test saving the progressive replay index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            # Create mock match result
            match_result = MatchResult(
                winner="AI_Fighter",
                total_ticks=50,
                final_hp_a=75.0,
                final_hp_b=0.0,
                telemetry={},
                events=[]
            )

            # Record multiple replays
            for episode in [1, 10, 50, 100, 200]:
                # Create valid telemetry with at least one tick
                telemetry = {"ticks": [{"tick": 0}]}

                recorder.record_episode_replay(
                    telemetry=telemetry,
                    match_result=match_result,
                    level_name="Test Level",
                    level_num=1,
                    episode=episode,
                    total_episodes=500,
                    win_rate=min(0.9, episode / 200),  # Improving win rate
                    recent_rewards=[episode * 0.5]
                )

            # Save index
            recorder.save_progressive_index()

            # Check index file exists
            index_path = Path(tmpdir) / "progressive_replay_index.json"
            assert index_path.exists()

            # Load and verify index
            with open(index_path) as f:
                index_data = json.load(f)

            assert index_data["total_replays"] == 5
            assert "recording_strategy" in index_data
            assert len(index_data["replays"]) == 5

            # Check replays are in order
            episodes = [r["episode"] for r in index_data["replays"]]
            assert episodes == [1, 10, 50, 100, 200]

            # Check win rates improve
            win_rates = [r["win_rate"] for r in index_data["replays"]]
            assert win_rates[0] < win_rates[-1]  # Should improve over time

    def test_recording_distribution(self):
        """Test that recording distribution follows the intended strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            total_episodes = 1000
            recorded = []

            for episode in range(1, total_episodes + 1):
                if recorder.should_record(episode, total_episodes):
                    recorded.append(episode)

            # Should record reasonable number of episodes
            assert 15 <= len(recorded) <= 40  # More recordings than expected due to first/last always recorded

            # Check distribution across phases
            early_phase = [e for e in recorded if e <= 200]
            mid_phase = [e for e in recorded if 200 < e <= 800]
            late_phase = [e for e in recorded if e > 800]

            # Early phase should have more recordings (every 10)
            assert len(early_phase) >= 10

            # Mid phase should have moderate recordings (every 50)
            assert 5 <= len(mid_phase) <= 15

            # Late phase should have fewer recordings (every 100)
            assert len(late_phase) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])