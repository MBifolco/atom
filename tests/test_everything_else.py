"""
Test all remaining untested code paths.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.arena import WorldConfig, FighterState
from src.arena.arena_1d_jax_jit import Arena1DJAXJit
from src.training.gym_env import AtomCombatEnv


def opp(s): return {"acceleration": 0.5 * s.get("opponent", {}).get("direction", 1), "stance": "neutral"}


class TestEverythingGymEnv:
    def test_a(self): AtomCombatEnv(opp, fighter_mass=50.0).reset()
    def test_b(self): AtomCombatEnv(opp, fighter_mass=90.0).reset()
    def test_c(self): AtomCombatEnv(opp, opponent_mass=50.0).reset()
    def test_d(self): AtomCombatEnv(opp, opponent_mass=90.0).reset()
    def test_e(self): AtomCombatEnv(opp, max_ticks=50).reset()
    def test_f(self): AtomCombatEnv(opp, max_ticks=300).reset()

    def test_g(self):
        env = AtomCombatEnv(opp)
        env.reset()
        for _ in range(10): env.step(np.array([1.0, 1.0]))

    def test_h(self):
        env = AtomCombatEnv(opp)
        env.reset()
        for _ in range(10): env.step(np.array([-1.0, 2.0]))

    def test_i(self):
        env = AtomCombatEnv(opp)
        env.reset()
        for _ in range(10): env.step(np.array([0.0, 0.0]))

    def test_j(self):
        env = AtomCombatEnv(opp)
        env.reset()
        for i in range(20):
            env.step(np.array([float(i % 3 - 1), float(i % 3)]))

    def test_k(self):
        def aggressive(s):
            d = s.get("opponent", {}).get("direction", 1)
            return {"acceleration": 1.0 * d, "stance": "extended"}
        env = AtomCombatEnv(aggressive, fighter_mass=85.0, opponent_mass=50.0)
        env.reset()
        for _ in range(100):
            obs, r, d, t, i = env.step(np.array([1.0, 1.0]))
            if d or t: break

    def test_l(self):
        def defensive(s): return {"acceleration": 0.0, "stance": "defending"}
        env = AtomCombatEnv(defensive, fighter_mass=50.0, opponent_mass=85.0)
        env.reset()
        for _ in range(100):
            obs, r, d, t, i = env.step(np.array([0.0, 2.0]))
            if d or t: break

    def test_m(self):
        env = AtomCombatEnv(opp, max_ticks=15)
        env.reset()
        for _ in range(20):
            obs, r, d, t, i = env.step(np.array([0.3, 1.0]))
            if t: break

    def test_n(self):
        env = AtomCombatEnv(opp)
        env.reset()
        for i in range(30):
            if i < 10: a = np.array([1.0, 1.0])
            elif i < 20: a = np.array([0.0, 2.0])
            else: a = np.array([-0.5, 0.0])
            env.step(a)

    def test_o(self):
        env = AtomCombatEnv(opp)
        obs, _ = env.reset()
        assert obs[0] > 0  # Position
        assert -3 <= obs[1] <= 3  # Velocity
        assert 0 <= obs[2] <= 1  # HP norm
        assert 0 <= obs[3] <= 1  # Stamina norm

    def test_p(self):
        config = WorldConfig()
        env = AtomCombatEnv(opp, config=config)
        assert env.config == config


class TestEverythingOrchestrator:
    def test_a(self):
        config = WorldConfig()
        for max_t in [5, 10, 20, 50]:
            orch = MatchOrchestrator(config, max_ticks=max_t)
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                opp, opp, seed=1
            )
            assert result.total_ticks <= max_t

    def test_b(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=30)
        for seed in [1, 10, 100, 1000]:
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                opp, opp, seed=seed
            )
            assert result.total_ticks > 0

    def test_c(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=25, record_telemetry=True)
        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            opp, opp, seed=1
        )
        assert len(result.telemetry["ticks"]) > 0


class TestEverythingEvaluator:
    def test_a(self):
        config = WorldConfig()
        for max_t in [5, 15, 30]:
            orch = MatchOrchestrator(config, max_ticks=max_t, record_telemetry=True)
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                opp, opp, seed=1
            )
            evaluator = SpectacleEvaluator()
            score = evaluator.evaluate(result.telemetry, result)
            assert 0 <= score.overall <= 1

    def test_b(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)
        result = orch.run_match(
            {"name": "A", "mass": 60.0, "position": 3.0},
            {"name": "B", "mass": 80.0, "position": 9.0},
            opp, opp, seed=1
        )
        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)
        score_dict = score.to_dict()
        assert isinstance(score_dict, dict)
        assert len(score_dict) > 0


class TestEverythingRenderer:
    def test_a(self):
        renderer = AsciiRenderer()
        for pos in [1.0, 3.0, 6.0, 9.0, 11.0]:
            tick = {
                "tick": 1,
                "fighter_a": {
                    "name": "A", "mass": 70.0, "position": pos, "velocity": 0.5,
                    "hp": 90.0, "max_hp": 100.0, "stamina": 8.0, "max_stamina": 10.0,
                    "stance": "neutral"
                },
                "fighter_b": {
                    "name": "B", "mass": 70.0, "position": pos+1.5, "velocity": -0.5,
                    "hp": 85.0, "max_hp": 100.0, "stamina": 7.0, "max_stamina": 10.0,
                    "stance": "extended"
                },
                "events": []
            }
            renderer.render_tick(tick)

    def test_b(self):
        renderer = AsciiRenderer(arena_width=15.0, display_width=60)
        tick = {
            "tick": 10,
            "fighter_a": {
                "name": "Alpha", "mass": 75.0, "position": 7.5, "velocity": 1.5,
                "hp": 50.0, "max_hp": 100.0, "stamina": 3.0, "max_stamina": 10.0,
                "stance": "extended"
            },
            "fighter_b": {
                "name": "Beta", "mass": 65.0, "position": 8.5, "velocity": -1.0,
                "hp": 60.0, "max_hp": 100.0, "stamina": 9.0, "max_stamina": 10.0,
                "stance": "defending"
            },
            "events": [{"type": "HIT", "damage": 5.0}]
        }
        renderer.render_tick(tick)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
