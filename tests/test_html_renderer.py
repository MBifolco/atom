"""
Tests for HtmlRenderer.

Tests cover:
- HtmlRenderer initialization
- generate_replay_html with and without spectacle_score
- HTML file creation and content
- Custom playback speed
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.renderer.html_renderer import HtmlRenderer
from src.orchestrator.match_orchestrator import MatchResult
from src.evaluator.spectacle_evaluator import SpectacleScore


class TestHtmlRendererInit:
    """Test HtmlRenderer initialization."""

    def test_html_renderer_initializes(self):
        """HtmlRenderer can be initialized."""
        renderer = HtmlRenderer()

        assert renderer is not None
        assert hasattr(renderer, 'template')

    def test_template_is_not_empty(self):
        """Template contains HTML content."""
        renderer = HtmlRenderer()

        assert len(renderer.template) > 0
        assert '<!DOCTYPE html>' in renderer.template
        assert '<canvas' in renderer.template


class TestGenerateReplayHtml:
    """Test generate_replay_html method."""

    def test_generates_html_file(self):
        """generate_replay_html creates an HTML file."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "FighterA", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "FighterB", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "FighterA",
            "fighter_b_name": "FighterB",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="FighterA",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_replay.html")
            result_path = renderer.generate_replay_html(telemetry, match_result, output_path)

            assert os.path.exists(result_path)
            assert result_path.suffix == '.html'

    def test_html_contains_replay_data(self):
        """Generated HTML contains replay data."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "TestFighter", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "Opponent", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "TestFighter",
            "fighter_b_name": "Opponent",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="TestFighter",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_replay.html")
            result_path = renderer.generate_replay_html(telemetry, match_result, output_path)

            # Read the generated HTML
            with open(result_path, 'r') as f:
                html_content = f.read()

            # Check that replay data is embedded
            assert 'const REPLAY_DATA' in html_content
            assert 'TestFighter' in html_content
            assert 'Opponent' in html_content

    def test_html_with_spectacle_score(self):
        """Generated HTML includes spectacle score when provided."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "FighterA", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "FighterB", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "FighterA",
            "fighter_b_name": "FighterB",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="FighterA",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        spectacle_score = SpectacleScore(
            duration=0.8,
            close_finish=0.9,
            stamina_drama=0.7,
            comeback_potential=0.6,
            positional_exchange=0.5,
            pacing_variety=0.4,
            collision_drama=0.3,
            overall=0.6
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_replay.html")
            result_path = renderer.generate_replay_html(
                telemetry, match_result, output_path, spectacle_score=spectacle_score
            )

            # Read the generated HTML
            with open(result_path, 'r') as f:
                html_content = f.read()

            # Check that spectacle score is included
            assert 'spectacle_score' in html_content
            assert '0.6' in html_content  # overall score

    def test_html_without_spectacle_score(self):
        """Generated HTML works without spectacle score."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "FighterA", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "FighterB", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "FighterA",
            "fighter_b_name": "FighterB",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="FighterA",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_replay.html")
            result_path = renderer.generate_replay_html(
                telemetry, match_result, output_path
            )

            # Read the generated HTML
            with open(result_path, 'r') as f:
                html_content = f.read()

            # Spectacle score should be null
            assert '"spectacle_score": null' in html_content

    def test_custom_playback_speed(self):
        """Custom playback speed is embedded in HTML."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "FighterA", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "FighterB", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "FighterA",
            "fighter_b_name": "FighterB",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="FighterA",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_replay.html")
            result_path = renderer.generate_replay_html(
                telemetry, match_result, output_path, playback_speed=2.5
            )

            # Read the generated HTML
            with open(result_path, 'r') as f:
                html_content = f.read()

            # Custom speed should be in the data
            assert '"playback_speed": 2.5' in html_content

    def test_creates_parent_directories(self):
        """generate_replay_html creates parent directories if needed."""
        renderer = HtmlRenderer()

        telemetry = {
            "ticks": [{
                "tick": 0,
                "fighter_a": {
                    "name": "FighterA", "mass": 70.0, "position": 2.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                },
                "fighter_b": {
                    "name": "FighterB", "mass": 70.0, "position": 10.0,
                    "velocity": 0.0, "hp": 100.0, "max_hp": 100.0,
                    "stamina": 10.0, "max_stamina": 10.0, "stance": "neutral"
                }
            }],
            "fighter_a_name": "FighterA",
            "fighter_b_name": "FighterB",
            "config": {
                "arena_width": 12.5,
                "dt": 0.08
            }
        }

        match_result = MatchResult(
            winner="FighterA",
            total_ticks=1,
            final_hp_a=100.0,
            final_hp_b=0.0,
            telemetry=telemetry,
            events=[]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist yet
            output_path = os.path.join(tmpdir, "nested", "dir", "test_replay.html")
            result_path = renderer.generate_replay_html(telemetry, match_result, output_path)

            # Should create the directories and file
            assert os.path.exists(result_path)
            assert os.path.exists(os.path.dirname(result_path))
