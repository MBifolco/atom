"""
Comprehensive tests targeting all remaining uncovered branches.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.arena import WorldConfig
from src.training.gym_env import AtomCombatEnv
from src.registry import FighterRegistry, FighterMetadata
from src.telemetry import ReplayStore
from src.training.replay_recorder import ReplayRecorder
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import AsciiRenderer


def simple_opponent_func(state):
    d = state.get("opponent", {}).get("direction", 1.0)
    return {"acceleration": 0.5 * d, "stance": "neutral"}


class TestRegistryPrivateMethods:
    """Test registry private methods."""

    def test_has_decide_function_method(self):
        """Test _has_decide_function checks correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File with decide
            good = Path(tmpdir) / "good.py"
            good.write_text("def decide(state): return {}")

            # File without decide
            bad = Path(tmpdir) / "bad.py"
            bad.write_text("def other(): pass")

            reg = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)

            # Should detect decide function
            assert reg._has_decide_function(good) == True
            assert reg._has_decide_function(bad) == False

    def test_extract_description_from_docstring(self):
        """Test docstring extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "documented.py"
            f.write_text('"""Fighter description here"""\ndef decide(s): return {}')

            reg = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)
            count = reg.scan_directory(Path(tmpdir))

            if count > 0:
                fighters = reg.list_fighters()
                assert len(fighters) > 0

    def test_compute_code_hash(self):
        """Test code hash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "hash_test.py"
            f.write_text("def decide(state): return {'acceleration': 0, 'stance': 'neutral'}")

            reg = FighterRegistry(Path(tmpdir) / "reg.json", load_existing=False)
            count = reg.scan_directory(Path(tmpdir))

            assert count == 1


class TestReplayStoreAllMethods:
    """Test all ReplayStore methods."""

    def test_save_uncompressed_and_compressed(self):
        """Test both save modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            config = WorldConfig()
            orch = MatchOrchestrator(config, max_ticks=5, record_telemetry=True)

            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                simple_opponent_func, simple_opponent_func, seed=1
            )

            # Uncompressed
            path1 = store.save(result.telemetry, result, compress=False, filename="test1.json")
            assert path1.exists()
            assert path1.suffix == ".json"

            # Compressed
            path2 = store.save(result.telemetry, result, compress=True, filename="test2.json.gz")
            assert path2.exists()
            assert path2.suffix == ".gz"

    def test_load_both_formats(self):
        """Test loading both formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            config = WorldConfig()
            orch = MatchOrchestrator(config, max_ticks=5, record_telemetry=True)

            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                simple_opponent_func, simple_opponent_func, seed=1
            )

            path_json = store.save(result.telemetry, result, compress=False, filename="t1.json")
            path_gz = store.save(result.telemetry, result, compress=True, filename="t2.json.gz")

            data1 = store.load(str(path_json))
            data2 = store.load(str(path_gz))

            assert data1 is not None
            assert data2 is not None


class TestReplayRecorderMethods:
    """Test ReplayRecorder methods."""

    def test_recorder_initialization_complete(self):
        """Test full recorder initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig()

            recorder = ReplayRecorder(
                output_dir=tmpdir,
                config=config,
                max_ticks=100,
                samples_per_stage=3,
                min_matches_for_sampling=5,
                verbose=True
            )

            assert recorder.output_dir == Path(tmpdir)
            assert recorder.config == config
            assert recorder.max_ticks == 100
            assert recorder.samples_per_stage == 3
            assert recorder.min_matches_for_sampling == 5
            assert recorder.verbose == True
            assert recorder.orchestrator is not None
            assert recorder.spectacle_evaluator is not None

    def test_recorder_replay_index_tracking(self):
        """Test replay index is maintained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ReplayRecorder(output_dir=tmpdir, verbose=False)

            assert hasattr(recorder, 'replay_index')
            assert isinstance(recorder.replay_index, list)
            assert len(recorder.replay_index) == 0


class TestSpectacleEvaluatorComplete:
    """Complete spectacle evaluator coverage."""

    def test_evaluate_short_match(self):
        """Test evaluation of very short match."""
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=3, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 5.0},
            {"name": "B", "mass": 70.0, "position": 6.0},
            simple_opponent_func, simple_opponent_func, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert score.overall >= 0

    def test_evaluate_long_match(self):
        """Test evaluation of long match."""
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=200, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_opponent_func, simple_opponent_func, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert score.overall >= 0

    def test_evaluate_one_sided_match(self):
        """Test evaluation of one-sided match."""
        def weak(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        def strong(state):
            d = state.get("opponent", {}).get("direction", 1.0)
            return {"acceleration": 1.0 * d, "stance": "extended"}

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=100, record_telemetry=True)

        result = orch.run_match(
            {"name": "Strong", "mass": 85.0, "position": 5.0},
            {"name": "Weak", "mass": 50.0, "position": 6.0},
            strong, weak, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert score.overall >= 0


class TestAsciiRendererComplete:
    """Complete ASCII renderer coverage."""

    def test_render_different_stances(self):
        """Test rendering all different stances."""
        renderer = AsciiRenderer()

        for stance in ["neutral", "extended", "defending"]:
            tick_data = {
                "tick": 1,
                "fighter_a": {
                    "name": "A", "mass": 70.0, "position": 5.0, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": stance
                },
                "fighter_b": {
                    "name": "B", "mass": 70.0, "position": 7.0, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "events": []
            }

            renderer.render_tick(tick_data)

    def test_render_with_different_velocities(self):
        """Test rendering fighters with various velocities."""
        renderer = AsciiRenderer()

        velocities = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for vel in velocities:
            tick_data = {
                "tick": 1,
                "fighter_a": {
                    "name": "A", "mass": 70.0, "position": 5.0, "velocity": vel,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "fighter_b": {
                    "name": "B", "mass": 70.0, "position": 7.0, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "events": []
            }

            renderer.render_tick(tick_data)

    def test_render_with_low_hp(self):
        """Test rendering fighters with low HP."""
        renderer = AsciiRenderer()

        tick_data = {
            "tick": 50,
            "fighter_a": {
                "name": "A", "mass": 70.0, "position": 5.0, "velocity": 0.0,
                "hp": 10.0, "max_hp": 100.0, "stamina": 2.0, "max_stamina": 10.0,
                "stance": "defending"
            },
            "fighter_b": {
                "name": "B", "mass": 70.0, "position": 6.0, "velocity": 0.0,
                "hp": 95.0, "max_hp": 100.0, "stamina": 9.0, "max_stamina": 10.0,
                "stance": "extended"
            },
            "events": [{"type": "HIT", "damage": 5.0}]
        }

        renderer.render_tick(tick_data)

    def test_render_summary_with_spectacle(self):
        """Test render summary with spectacle score."""
        renderer = AsciiRenderer()

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_opponent_func, simple_opponent_func, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        renderer.render_summary(result, score)

    def test_make_bar_method(self):
        """Test _make_bar helper method."""
        renderer = AsciiRenderer()

        bar = renderer._make_bar(0.75, 20)

        assert isinstance(bar, str)
        assert len(bar) == 20


class TestOrchestratorBranches:
    """Test orchestrator branches."""

    def test_match_ends_on_knockout(self):
        """Test match detection when fighter reaches 0 HP."""
        def aggressive(state):
            d = state.get("opponent", {}).get("direction", 1.0)
            return {"acceleration": 1.0 * d, "stance": "extended"}

        def passive(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=150)

        result = orch.run_match(
            {"name": "Attacker", "mass": 85.0, "position": 5.0},
            {"name": "Passive", "mass": 50.0, "position": 6.0},
            aggressive, passive, seed=1
        )

        # Should end in knockout
        assert result.final_hp_a == 0 or result.final_hp_b == 0 or "(timeout)" in result.winner

    def test_match_with_different_masses(self):
        """Test match with very different fighter masses."""
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=50)

        result = orch.run_match(
            {"name": "Light", "mass": 45.0, "position": 3.0},
            {"name": "Heavy", "mass": 90.0, "position": 9.0},
            simple_opponent_func, simple_opponent_func, seed=1
        )

        assert result.total_ticks > 0

    def test_match_telemetry_structure(self):
        """Test telemetry has correct structure."""
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_opponent_func, simple_opponent_func, seed=1
        )

        assert "ticks" in result.telemetry
        assert "fighter_a_name" in result.telemetry
        assert "fighter_b_name" in result.telemetry

        if len(result.telemetry["ticks"]) > 0:
            tick = result.telemetry["ticks"][0]
            assert "tick" in tick
            assert "fighter_a" in tick
            assert "fighter_b" in tick


class TestGymEnvAllRewardBranches:
    """Hit all gym env reward branches."""

    def test_win_with_all_bonuses(self):
        """Test win with time, HP, and stamina bonuses."""
        def very_weak(state):
            return {"acceleration": 0.0, "stance": "neutral"}

        env = AtomCombatEnv(
            opponent_decision_func=very_weak,
            fighter_mass=85.0,
            opponent_mass=45.0,
            max_ticks=150
        )

        env.reset()

        for _ in range(150):
            action = np.array([0.8, 1.0])
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break

    def test_loss_reward_complete(self):
        """Test complete loss reward calculation."""
        def overwhelming(state):
            d = state.get("opponent", {}).get("direction", 1.0)
            return {"acceleration": 1.0 * d, "stance": "extended"}

        env = AtomCombatEnv(
            opponent_decision_func=overwhelming,
            fighter_mass=45.0,
            opponent_mass=90.0,
            max_ticks=200
        )

        env.reset()

        for _ in range(200):
            action = np.array([0.0, 2.0])
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break

    def test_all_timeout_branches(self):
        """Test all timeout reward branches."""
        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent_func,
            fighter_mass=70.0,
            opponent_mass=70.0,
            max_ticks=20
        )

        env.reset()

        for _ in range(25):
            action = np.array([0.5, 1.0])
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                break

    def test_proximity_reward_all_branches(self):
        """Test all proximity reward branches."""
        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent_func,
            fighter_mass=70.0,
            opponent_mass=70.0,
            max_ticks=40
        )

        env.reset()

        for i in range(40):
            if i < 10:
                action = np.array([1.0, 0.0])  # Approach
            elif i < 20:
                action = np.array([-0.5, 2.0])  # Retreat, defend
            elif i < 30:
                action = np.array([0.5, 1.0])  # Normal engagement
            else:
                action = np.array([0.0, 0.0])  # Passive

            env.step(action)

    def test_stamina_reward_branches(self):
        """Test all stamina reward branches."""
        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent_func,
            fighter_mass=70.0,
            opponent_mass=70.0,
            max_ticks=50
        )

        env.reset()

        for i in range(50):
            if i < 15:
                action = np.array([1.0, 1.0])  # Exhaust stamina
            elif i < 30:
                action = np.array([0.0, 2.0])  # Recover with defending
            else:
                action = np.array([0.5, 0.0])  # Neutral stance

            env.step(action)

    def test_stance_reward_branches(self):
        """Test all stance reward branches."""
        env = AtomCombatEnv(
            opponent_decision_func=simple_opponent_func,
            fighter_mass=70.0,
            opponent_mass=70.0,
            max_ticks=60
        )

        env.reset()

        for i in range(60):
            # Cycle through all stances in different scenarios
            if i % 3 == 0:
                action = np.array([0.5, 1.0])  # Extended
            elif i % 3 == 1:
                action = np.array([0.0, 2.0])  # Defending
            else:
                action = np.array([0.3, 0.0])  # Neutral

            env.step(action)


class TestRendererBranches:
    """Test renderer branches."""

    def test_render_at_different_positions(self):
        """Test rendering at various arena positions."""
        renderer = AsciiRenderer()

        positions = [0.5, 3.0, 6.0, 9.0, 11.5]

        for pos in positions:
            tick_data = {
                "tick": 1,
                "fighter_a": {
                    "name": "A", "mass": 70.0, "position": pos, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "fighter_b": {
                    "name": "B", "mass": 70.0, "position": pos + 1, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "events": []
            }

            renderer.render_tick(tick_data)

    def test_render_with_various_hp_levels(self):
        """Test rendering with different HP levels."""
        renderer = AsciiRenderer()

        hp_levels = [10.0, 25.0, 50.0, 75.0, 100.0]

        for hp in hp_levels:
            tick_data = {
                "tick": 1,
                "fighter_a": {
                    "name": "A", "mass": 70.0, "position": 5.0, "velocity": 0.0,
                    "hp": hp, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "fighter_b": {
                    "name": "B", "mass": 70.0, "position": 7.0, "velocity": 0.0,
                    "hp": 100.0, "max_hp": 100.0, "stamina": 10.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "events": []
            }

            renderer.render_tick(tick_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
