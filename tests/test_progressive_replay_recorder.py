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
            assert recorder.early_phase_interval == 25
            assert recorder.mid_phase_interval == 50
            assert recorder.late_phase_interval == 100

    def test_should_record_strategy(self):
        """Test recording strategy at different phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ProgressiveReplayRecorder(tmpdir, verbose=False)

            total_episodes = 1000

            # Always record first episode
            assert recorder.should_record(1, total_episodes)

            # Episode-based recording (not phase-based anymore)
            # Early (< 200): every 25 episodes
            assert recorder.should_record(25, total_episodes)  # Yes
            assert recorder.should_record(50, total_episodes)  # Yes
            assert recorder.should_record(75, total_episodes)  # Yes
            assert not recorder.should_record(10, total_episodes)  # No
            assert not recorder.should_record(15, total_episodes)  # No
            assert not recorder.should_record(30, total_episodes)  # No

            # Mid (200-999): every 50 episodes
            assert recorder.should_record(250, total_episodes)  # Yes
            assert recorder.should_record(300, total_episodes)  # Yes
            assert not recorder.should_record(275, total_episodes)  # No
            assert not recorder.should_record(325, total_episodes)  # No

            # Late (1000+): every 100 episodes
            assert recorder.should_record(1000, total_episodes)  # Yes
            assert recorder.should_record(1100, total_episodes)  # Yes
            assert not recorder.should_record(1050, total_episodes)  # No

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
            assert 15 <= len(recorded) <= 30  # Updated for new intervals

            # Check distribution using episode-based phases (not percentage)
            early_phase = [e for e in recorded if e < 200]  # < 200: every 25
            mid_phase = [e for e in recorded if 200 <= e < 1000]  # 200-999: every 50
            late_phase = [e for e in recorded if e >= 1000]  # 1000+: every 100

            # Early phase should have recordings every 25 (1, 25, 50, 75, 100, 125, 150, 175)
            assert 8 <= len(early_phase) <= 9  # Including episode 1

            # Mid phase should have recordings every 50 (200, 250, 300, ..., 950)
            assert 15 <= len(mid_phase) <= 17

            # Late phase - we only go to 1000, so just episode 1000
            assert len(late_phase) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])