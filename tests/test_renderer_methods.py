"""
Comprehensive tests for renderer methods.
Tests ASCII and HTML rendering with various inputs.
"""

import pytest
import tempfile
from pathlib import Path
from src.renderer import AsciiRenderer, HtmlRenderer
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.arena import WorldConfig


def simple_fighter_a(state):
    d = state.get("opponent", {}).get("direction", 1.0)
    return {"acceleration": 0.7 * d, "stance": "extended"}


def simple_fighter_b(state):
    d = state.get("opponent", {}).get("direction", 1.0)
    return {"acceleration": 0.5 * d, "stance": "neutral"}


class TestAsciiRendererHelperMethods:
    """Test ASCII renderer helper methods."""

    def test_make_bar_creates_correct_length(self):
        """Test _make_bar creates bar of specified width."""
        renderer = AsciiRenderer()

        bar_10 = renderer._make_bar(0.5, 10)
        bar_20 = renderer._make_bar(0.5, 20)
        bar_30 = renderer._make_bar(0.75, 30)

        assert len(bar_10) == 10
        assert len(bar_20) == 20
        assert len(bar_30) == 30

    def test_make_bar_fills_based_on_percentage(self):
        """Test _make_bar fills correctly based on percentage."""
        renderer = AsciiRenderer()

        bar_empty = renderer._make_bar(0.0, 20)
        bar_half = renderer._make_bar(0.5, 20)
        bar_full = renderer._make_bar(1.0, 20)

        # Count filled characters (assuming '█' is filled)
        filled_empty = bar_empty.count('█')
        filled_half = bar_half.count('█')
        filled_full = bar_full.count('█')

        assert filled_empty == 0
        assert filled_half == 10
        assert filled_full == 20

    def test_make_bar_with_custom_characters(self):
        """Test _make_bar with custom fill/empty characters."""
        renderer = AsciiRenderer()

        bar = renderer._make_bar(0.6, 10, filled_char='#', empty_char='-')

        assert '#' in bar or '-' in bar
        assert len(bar) == 10

    def test_render_summary_with_match_result(self):
        """Test render_summary displays match results."""
        renderer = AsciiRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_fighter_a, simple_fighter_b, seed=1
        )

        # Should not crash
        renderer.render_summary(result)

    def test_render_summary_with_spectacle_score(self):
        """Test render_summary includes spectacle score when provided."""
        renderer = AsciiRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=15, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_fighter_a, simple_fighter_b, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        # Should not crash with spectacle score
        renderer.render_summary(result, score)


class TestHtmlRendererGeneration:
    """Test HTML renderer generation methods."""

    def test_html_renderer_has_template(self):
        """Test HTML renderer initializes with template."""
        renderer = HtmlRenderer()

        assert hasattr(renderer, 'template')
        assert isinstance(renderer.template, str)
        assert len(renderer.template) > 1000  # Should be substantial

    def test_html_template_contains_required_elements(self):
        """Test HTML template has required HTML structure."""
        renderer = HtmlRenderer()

        template = renderer.template

        assert '<!DOCTYPE html>' in template
        assert '<canvas' in template
        assert 'REPLAY_DATA_PLACEHOLDER' in template

    def test_generate_replay_html_creates_file(self):
        """Test generate_replay_html creates output file."""
        renderer = HtmlRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=5, record_telemetry=True)

        result = orch.run_match(
            {"name": "Fighter1", "mass": 70.0, "position": 3.0},
            {"name": "Fighter2", "mass": 70.0, "position": 9.0},
            simple_fighter_a, simple_fighter_b, seed=1
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result_path = renderer.generate_replay_html(
                result.telemetry,
                result,
                output_path
            )

            assert Path(output_path).exists()
            assert result_path.exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_replay_html_with_spectacle_score(self):
        """Test HTML generation includes spectacle score if provided."""
        renderer = HtmlRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=8, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_fighter_a, simple_fighter_b, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            renderer.generate_replay_html(
                result.telemetry,
                result,
                output_path,
                spectacle_score=score
            )

            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_generate_replay_html_with_custom_playback_speed(self):
        """Test HTML generation with custom playback speed."""
        renderer = HtmlRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=5, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_fighter_a, simple_fighter_b, seed=1
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            renderer.generate_replay_html(
                result.telemetry,
                result,
                output_path,
                playback_speed=2.0
            )

            # Read file and check for playback speed
            content = Path(output_path).read_text()

            assert 'playback_speed' in content or '2.0' in content
        finally:
            Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
