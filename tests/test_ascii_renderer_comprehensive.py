"""
Comprehensive tests for ascii_renderer to cover uncovered paths.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from src.renderer.ascii_renderer import AsciiRenderer
from src.evaluator.spectacle_evaluator import SpectacleScore


class TestAsciiRendererInit:
    """Tests for AsciiRenderer initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        renderer = AsciiRenderer()
        assert renderer.arena_width == 12.0
        assert renderer.display_width == 50

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        renderer = AsciiRenderer(arena_width=20.0, display_width=80)
        assert renderer.arena_width == 20.0
        assert renderer.display_width == 80

    def test_init_stance_chars(self):
        """Test stance characters are set up."""
        renderer = AsciiRenderer()
        assert 'neutral' in renderer.stance_chars
        assert 'extended' in renderer.stance_chars
        assert 'defending' in renderer.stance_chars


class TestRenderTick:
    """Tests for render_tick method."""

    def _create_fighter(self, name, position, velocity, hp, stamina, stance, mass=70.0):
        """Helper to create fighter data with all required fields."""
        return {
            "name": name, "position": position, "velocity": velocity,
            "hp": hp, "max_hp": 100.0, "stamina": stamina, "max_stamina": 100.0,
            "stance": stance, "mass": mass
        }

    def test_render_tick_basic(self, capsys):
        """Test basic tick rendering."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 0,
            "dt": 0.0842,
            "fighter_a": self._create_fighter("Alpha", 3.0, 0.5, 100.0, 100.0, "neutral"),
            "fighter_b": self._create_fighter("Beta", 9.0, -0.3, 100.0, 100.0, "neutral"),
            "arena_width": 12.0,
            "events": []
        }

        renderer.render_tick(tick_data, dt=0.0842)
        captured = capsys.readouterr()

        assert len(captured.out) > 0

    def test_render_tick_collision(self, capsys):
        """Test tick rendering with collision."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 50,
            "dt": 0.0842,
            "fighter_a": self._create_fighter("Alpha", 5.0, 1.0, 80.0, 70.0, "extended"),
            "fighter_b": self._create_fighter("Beta", 5.5, -0.5, 90.0, 80.0, "defending"),
            "arena_width": 12.0,
            "events": [{
                "type": "COLLISION",
                "damage_to_a": 5.0,
                "damage_to_b": 8.0,
                "relative_velocity": 1.5
            }]
        }

        renderer.render_tick(tick_data, dt=0.0842)
        captured = capsys.readouterr()

        # Should render something
        assert len(captured.out) > 0

    def test_render_tick_low_hp(self, capsys):
        """Test tick rendering with low HP."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 100,
            "dt": 0.0842,
            "fighter_a": self._create_fighter("Alpha", 4.0, 0.0, 10.0, 20.0, "defending"),
            "fighter_b": self._create_fighter("Beta", 8.0, 0.0, 90.0, 80.0, "extended"),
            "arena_width": 12.0,
            "events": []
        }

        renderer.render_tick(tick_data, dt=0.0842)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_render_tick_same_position(self, capsys):
        """Test tick rendering when fighters at same position."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 75,
            "dt": 0.0842,
            "fighter_a": self._create_fighter("Alpha", 6.0, 0.5, 50.0, 50.0, "extended"),
            "fighter_b": self._create_fighter("Beta", 6.0, -0.5, 50.0, 50.0, "extended"),
            "arena_width": 12.0,
            "events": []
        }

        renderer.render_tick(tick_data, dt=0.0842)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestRenderSummary:
    """Tests for render_summary method."""

    def test_render_summary_basic(self, capsys):
        """Test basic summary rendering."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 200
        match_result.final_hp_a = 80.0
        match_result.final_hp_b = 0.0
        match_result.ko = True
        match_result.fighter_a_name = "Alpha"
        match_result.fighter_b_name = "Beta"
        match_result.events = [{"type": "COLLISION"}, {"type": "COLLISION"}]

        renderer.render_summary(match_result)
        captured = capsys.readouterr()

        assert len(captured.out) > 0
        assert "MATCH COMPLETE" in captured.out

    def test_render_summary_with_spectacle_excellent(self, capsys):
        """Test summary rendering with excellent spectacle score."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 150
        match_result.final_hp_a = 50.0
        match_result.final_hp_b = 0.0
        match_result.ko = True
        match_result.fighter_a_name = "Alpha"
        match_result.fighter_b_name = "Beta"
        match_result.events = []

        spectacle = SpectacleScore(
            duration=0.9, close_finish=0.9, stamina_drama=0.8,
            comeback_potential=0.8, positional_exchange=0.8,
            pacing_variety=0.8, collision_drama=0.8, overall=0.85
        )

        renderer.render_summary(match_result, spectacle)
        captured = capsys.readouterr()

        assert "EXCELLENT" in captured.out

    def test_render_summary_with_spectacle_good(self, capsys):
        """Test summary rendering with good spectacle score."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 150
        match_result.final_hp_a = 50.0
        match_result.final_hp_b = 0.0
        match_result.events = []

        spectacle = SpectacleScore(
            duration=0.7, close_finish=0.7, stamina_drama=0.6,
            comeback_potential=0.6, positional_exchange=0.6,
            pacing_variety=0.6, collision_drama=0.6, overall=0.65
        )

        renderer.render_summary(match_result, spectacle)
        captured = capsys.readouterr()

        assert "GOOD" in captured.out

    def test_render_summary_with_spectacle_fair(self, capsys):
        """Test summary rendering with fair spectacle score."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 150
        match_result.final_hp_a = 50.0
        match_result.final_hp_b = 0.0
        match_result.events = []

        spectacle = SpectacleScore(
            duration=0.5, close_finish=0.5, stamina_drama=0.4,
            comeback_potential=0.4, positional_exchange=0.4,
            pacing_variety=0.4, collision_drama=0.4, overall=0.45
        )

        renderer.render_summary(match_result, spectacle)
        captured = capsys.readouterr()

        assert "FAIR" in captured.out

    def test_render_summary_with_spectacle_poor(self, capsys):
        """Test summary rendering with poor spectacle score."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 150
        match_result.final_hp_a = 50.0
        match_result.final_hp_b = 0.0
        match_result.events = []

        spectacle = SpectacleScore(
            duration=0.2, close_finish=0.2, stamina_drama=0.2,
            comeback_potential=0.2, positional_exchange=0.2,
            pacing_variety=0.2, collision_drama=0.2, overall=0.2
        )

        renderer.render_summary(match_result, spectacle)
        captured = capsys.readouterr()

        assert "POOR" in captured.out

    def test_render_summary_draw(self, capsys):
        """Test summary rendering for a draw."""
        renderer = AsciiRenderer()

        match_result = Mock()
        match_result.winner = "draw"
        match_result.total_ticks = 300
        match_result.final_hp_a = 50.0
        match_result.final_hp_b = 50.0
        match_result.ko = False
        match_result.fighter_a_name = "Alpha"
        match_result.fighter_b_name = "Beta"
        match_result.events = []

        renderer.render_summary(match_result)
        captured = capsys.readouterr()

        assert len(captured.out) > 0


class TestPlayReplay:
    """Tests for play_replay method."""

    def _create_tick(self, i, pos_a=3.0, pos_b=9.0):
        """Helper to create a tick with all required fields."""
        return {
            "tick": i,
            "dt": 0.0842,
            "fighter_a": {
                "name": "Alpha", "position": pos_a + i * 0.1, "velocity": 0.5,
                "hp": 100.0 - i, "max_hp": 100.0, "stamina": 100.0, "max_stamina": 100.0,
                "stance": "neutral", "mass": 70.0
            },
            "fighter_b": {
                "name": "Beta", "position": pos_b - i * 0.1, "velocity": -0.3,
                "hp": 100.0, "max_hp": 100.0, "stamina": 100.0, "max_stamina": 100.0,
                "stance": "neutral", "mass": 70.0
            },
            "arena_width": 12.0,
            "events": []
        }

    def test_play_replay_empty_telemetry(self, capsys):
        """Test play_replay with empty telemetry."""
        renderer = AsciiRenderer()

        telemetry = {"ticks": [], "events": []}
        match_result = Mock()

        renderer.play_replay(telemetry, match_result)
        captured = capsys.readouterr()

        assert "No telemetry data to render" in captured.out

    def test_play_replay_with_ticks(self, capsys):
        """Test play_replay with tick data."""
        renderer = AsciiRenderer()

        ticks = [self._create_tick(i) for i in range(3)]
        telemetry = {"ticks": ticks, "events": [], "config": {"dt": 0.0842}}

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 3
        match_result.final_hp_a = 97.0
        match_result.final_hp_b = 100.0
        match_result.ko = False
        match_result.fighter_a_name = "Alpha"
        match_result.fighter_b_name = "Beta"
        match_result.events = []

        with patch('time.sleep'):
            renderer.play_replay(telemetry, match_result, playback_speed=1000.0, skip_ticks=1)

        captured = capsys.readouterr()
        assert "REPLAY START" in captured.out

    def test_play_replay_skip_ticks(self, capsys):
        """Test play_replay with skip_ticks > 1."""
        renderer = AsciiRenderer()

        ticks = [self._create_tick(i) for i in range(10)]
        telemetry = {"ticks": ticks}

        match_result = Mock()
        match_result.winner = "draw"
        match_result.total_ticks = 10
        match_result.final_hp_a = 100
        match_result.final_hp_b = 100
        match_result.ko = False
        match_result.fighter_a_name = "A"
        match_result.fighter_b_name = "B"
        match_result.events = []

        with patch('time.sleep'):
            renderer.play_replay(telemetry, match_result, playback_speed=1000.0, skip_ticks=5, show_all_ticks=False)

        captured = capsys.readouterr()
        assert "REPLAY START" in captured.out

    def test_play_replay_show_all_ticks(self, capsys):
        """Test play_replay with show_all_ticks=True."""
        renderer = AsciiRenderer()

        ticks = [self._create_tick(i) for i in range(5)]
        telemetry = {"ticks": ticks}

        match_result = Mock()
        match_result.winner = "fighter_a"
        match_result.total_ticks = 5
        match_result.final_hp_a = 100
        match_result.final_hp_b = 0
        match_result.ko = True
        match_result.fighter_a_name = "A"
        match_result.fighter_b_name = "B"
        match_result.events = []

        with patch('time.sleep'):
            renderer.play_replay(telemetry, match_result, playback_speed=1000.0, skip_ticks=10, show_all_ticks=True)

        captured = capsys.readouterr()
        assert "REPLAY START" in captured.out


class TestMakeBar:
    """Tests for _make_bar helper method."""

    def test_make_bar_full(self):
        """Test making a full bar."""
        renderer = AsciiRenderer()
        bar = renderer._make_bar(1.0, 10)
        assert len(bar) == 10
        assert '█' in bar

    def test_make_bar_half(self):
        """Test making a half-full bar."""
        renderer = AsciiRenderer()
        bar = renderer._make_bar(0.5, 10)
        assert len(bar) == 10
        assert '█' in bar
        assert '░' in bar

    def test_make_bar_empty(self):
        """Test making an empty bar."""
        renderer = AsciiRenderer()
        bar = renderer._make_bar(0.0, 10)
        assert len(bar) == 10
        assert '░' in bar

    def test_make_bar_custom_chars(self):
        """Test making a bar with custom characters."""
        renderer = AsciiRenderer()
        bar = renderer._make_bar(0.5, 10, filled_char='#', empty_char='-')
        assert '#' in bar
        assert '-' in bar
