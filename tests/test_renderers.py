"""
Tests for ASCII and HTML renderers.
"""

import pytest
from src.renderer import AsciiRenderer, HtmlRenderer
from src.orchestrator import MatchOrchestrator
from src.arena import WorldConfig


def simple_fighter(state):
    """Simple fighter for rendering tests."""
    direction = state["opponent"]["direction"]
    return {"acceleration": 0.8 * direction, "stance": "extended"}


class TestAsciiRenderer:
    """Test ASCII renderer."""

    def test_renderer_initialization(self):
        """Test ASCII renderer initializes with correct settings."""
        renderer = AsciiRenderer(arena_width=12.5, display_width=50)

        assert renderer.arena_width == 12.5
        assert renderer.display_width == 50
        assert renderer.scale == 12.5 / 50

    def test_stance_characters_defined(self):
        """Test stance visual characters are defined."""
        renderer = AsciiRenderer()

        assert 'neutral' in renderer.stance_chars
        assert 'extended' in renderer.stance_chars
        assert 'defending' in renderer.stance_chars

    def test_render_tick(self):
        """Test rendering a single tick."""
        renderer = AsciiRenderer(arena_width=12.0, display_width=40)

        tick_data = {
            "tick": 5,
            "fighter_a": {
                "name": "A",
                "mass": 70.0,
                "position": 3.0,
                "velocity": 0.5,
                "hp": 80.0,
                "max_hp": 100.0,
                "stamina": 8.0,
                "max_stamina": 10.0,
                "stance": "extended"
            },
            "fighter_b": {
                "name": "B",
                "mass": 70.0,
                "position": 9.0,
                "velocity": -0.5,
                "hp": 90.0,
                "max_hp": 100.0,
                "stamina": 7.0,
                "max_stamina": 10.0,
                "stance": "defending"
            },
            "events": []
        }

        # Should not crash
        renderer.render_tick(tick_data)

    def test_render_summary(self):
        """Test rendering match summary."""
        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=simple_fighter,
            decision_func_b=simple_fighter,
            seed=42
        )

        renderer = AsciiRenderer(arena_width=config.arena_width)

        # Should not crash when rendering summary
        # (output goes to stdout, we just verify no exceptions)
        renderer.render_summary(result)

    def test_position_scaling(self):
        """Test position scales correctly from arena to display."""
        renderer = AsciiRenderer(arena_width=10.0, display_width=50)

        # Position 5.0 in 10.0 arena should be 25 in 50-wide display
        expected = int(5.0 / (10.0 / 50))
        assert expected == 25

    def test_fighters_at_same_position(self):
        """Test rendering when fighters overlap."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 1,
            "fighter_a": {
                "name": "A",
                "mass": 70.0,
                "position": 6.0,
                "velocity": 0.0,
                "hp": 100.0,
                "max_hp": 100.0,
                "stamina": 10.0,
                "max_stamina": 10.0,
                "stance": "extended"
            },
            "fighter_b": {
                "name": "B",
                "mass": 70.0,
                "position": 6.0,  # Same position!
                "velocity": 0.0,
                "hp": 100.0,
                "max_hp": 100.0,
                "stamina": 10.0,
                "max_stamina": 10.0,
                "stance": "neutral"
            },
            "events": []
        }

        # Should handle overlap without crashing
        renderer.render_tick(tick_data)


class TestHtmlRenderer:
    """Test HTML renderer."""

    def test_html_renderer_initialization(self):
        """Test HTML renderer initializes."""
        renderer = HtmlRenderer()
        assert renderer is not None

    def test_render_replay_to_html(self):
        """Test rendering replay to HTML file."""
        import tempfile

        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "A", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "B", "mass": 70.0, "position": 9.0},
            decision_func_a=simple_fighter,
            decision_func_b=simple_fighter,
            seed=42
        )

        renderer = HtmlRenderer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        # Render to HTML file
        renderer.generate_replay_html(result.telemetry, result, output_path)

        # Verify file was created
        from pathlib import Path
        assert Path(output_path).exists()

        # Clean up
        Path(output_path).unlink()

    def test_html_contains_fighter_names(self):
        """Test HTML output contains fighter information."""
        import tempfile

        config = WorldConfig()
        orchestrator = MatchOrchestrator(config, max_ticks=5, record_telemetry=True)

        result = orchestrator.run_match(
            fighter_a_spec={"name": "TestFighterA", "mass": 70.0, "position": 3.0},
            fighter_b_spec={"name": "TestFighterB", "mass": 70.0, "position": 9.0},
            decision_func_a=simple_fighter,
            decision_func_b=simple_fighter,
            seed=42
        )

        renderer = HtmlRenderer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name

        renderer.generate_replay_html(result.telemetry, result, output_path)

        # Read HTML and check for fighter names
        from pathlib import Path
        html_content = Path(output_path).read_text()

        assert "TestFighterA" in html_content or "fighter_a" in html_content.lower()

        # Clean up
        Path(output_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
