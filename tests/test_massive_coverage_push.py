"""
Massive test file to push coverage to 50%.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.arena import WorldConfig
from src.atom.training.gym_env import AtomCombatEnv
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.telemetry import ReplayStore
from src.renderer import AsciiRenderer, HtmlRenderer
from src.registry import FighterRegistry, FighterMetadata


def f1(s): return {"acceleration": 0.8 * s.get("opponent", {}).get("direction", 1), "stance": "extended"}
def f2(s): return {"acceleration": 0.5 * s.get("opponent", {}).get("direction", 1), "stance": "neutral"}
def f3(s): return {"acceleration": -0.5 * s.get("opponent", {}).get("direction", 1), "stance": "defending"}


class T1:
    def test_gym_env_reset_seed(self):
        env = AtomCombatEnv(f1, seed=123)
        obs, _ = env.reset(seed=456)
        assert obs.shape == (9,)

    def test_gym_env_different_masses(self):
        env = AtomCombatEnv(f1, fighter_mass=60.0, opponent_mass=80.0)
        env.reset()
        env.step(np.array([0.5, 1.0]))

    def test_gym_env_max_ticks_custom(self):
        env = AtomCombatEnv(f1, max_ticks=30)
        assert env.max_ticks == 30

    def test_orchestrator_no_telemetry(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=False)
        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            f1, f2, seed=1
        )
        assert result.telemetry.get("config", {}) == {}

    def test_evaluator_default_weights(self):
        evaluator = SpectacleEvaluator()
        assert hasattr(evaluator, 'weights')

    def test_evaluator_with_minimal_telemetry(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=2, record_telemetry=True)
        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 5.0},
            {"name": "B", "mass": 70.0, "position": 6.0},
            f1, f2, seed=1
        )
        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)
        assert score.overall >= 0

    def test_replay_store_auto_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            config = WorldConfig()
            orch = MatchOrchestrator(config, max_ticks=3, record_telemetry=True)
            result = orch.run_match(
                {"name": "Fighter1", "mass": 70.0, "position": 3.0},
                {"name": "Fighter2", "mass": 70.0, "position": 9.0},
                f1, f2, seed=1
            )
            path = store.save(result.telemetry, result)
            assert "Fighter" in path.name or "replay" in path.name

    def test_renderer_different_display_widths(self):
        for width in [30, 50, 70]:
            renderer = AsciiRenderer(display_width=width)
            assert renderer.display_width == width

    def test_html_renderer_generation(self):
        renderer = HtmlRenderer()
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=3, record_telemetry=True)
        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            f1, f2, seed=1
        )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output = renderer.generate_replay_html(result.telemetry, result, f.name)
            assert output.exists()
            Path(f.name).unlink()

    def test_registry_scan_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            reg = FighterRegistry(Path(tmpdir) / "r.json", load_existing=False)
            count = reg.scan_directory(empty_dir)
            assert count == 0

    def test_registry_multiple_save_load_cycles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "test.json"
            reg1 = FighterRegistry(reg_path, load_existing=False)
            reg1.register_fighter(FighterMetadata(
                id="f1", name="F1", description="T", creator="t", type="rule-based", file_path="t.py"
            ))
            reg1.save()

            reg2 = FighterRegistry(reg_path, load_existing=True)
            assert len(reg2.fighters) == 1

            reg2.register_fighter(FighterMetadata(
                id="f2", name="F2", description="T", creator="t", type="rule-based", file_path="t.py"
            ))
            reg2.save()

            reg3 = FighterRegistry(reg_path, load_existing=True)
            assert len(reg3.fighters) == 2

    def test_metadata_with_performance_stats(self):
        meta = FighterMetadata(
            id="stats_test",
            name="Stats Test",
            description="Test",
            creator="test",
            type="onnx-ai",
            file_path="test.py",
            performance_stats={"elo": 1650, "win_rate": 0.75, "matches": 100}
        )
        assert meta.performance_stats["elo"] == 1650

    def test_metadata_created_date_generation(self):
        meta = FighterMetadata(
            id="date_test",
            name="Date Test",
            description="Test",
            creator="test",
            type="rule-based",
            file_path="test.py"
        )
        assert meta.created_date is not None

    def test_replay_store_with_metadata_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            config = WorldConfig()
            orch = MatchOrchestrator(config, max_ticks=3, record_telemetry=True)
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                f1, f2, seed=1
            )
            meta = {"tournament": "Test", "round": 1}
            path = store.save(result.telemetry, result, metadata=meta, compress=False)
            data = store.load(str(path))
            assert data["metadata"]["tournament"] == "Test"

    def test_gym_env_action_clamping(self):
        env = AtomCombatEnv(f1, max_ticks=20)
        env.reset()

        # Test extreme actions that need clamping
        extreme_action = np.array([10.0, 99.0])  # Way out of bounds
        obs, reward, done, truncated, info = env.step(extreme_action)
        assert not np.any(np.isnan(obs))

    def test_multiple_match_orchestration(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10)

        for i in range(5):
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                f1 if i % 2 == 0 else f2,
                f2 if i % 2 == 0 else f3,
                seed=i
            )
            assert result.total_ticks > 0

    def test_spectacle_score_components(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=20, record_telemetry=True)
        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 5.0},
            {"name": "B", "mass": 70.0, "position": 6.0},
            f1, f3, seed=1
        )
        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert 0 <= score.duration <= 1
        assert 0 <= score.close_finish <= 1
        assert 0 <= score.stamina_drama <= 1
        assert 0 <= score.overall <= 1

    def test_ascii_renderer_scale_calculation(self):
        renderer = AsciiRenderer(arena_width=10.0, display_width=100)
        assert renderer.scale == 0.1

        renderer2 = AsciiRenderer(arena_width=20.0, display_width=50)
        assert renderer2.scale == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
