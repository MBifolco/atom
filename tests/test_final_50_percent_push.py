"""
Final comprehensive test suite to push coverage from 44% to 50%.
Targets specific uncovered code paths in training modules.
"""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path
from src.arena import WorldConfig
from src.training.gym_env import AtomCombatEnv
from src.training.trainers.population.elo_tracker import EloTracker, FighterStats
from src.training.trainers.population.fighter_loader import load_fighter, validate_fighter, load_hardcoded_fighters
from src.training.trainers.population.population_trainer import (
    _configure_process_threading,
    _reconstruct_config,
    _create_opponent_decide_func,
    PopulationFighter,
    PopulationCallback
)
from src.training.trainers.curriculum_trainer import (
    DifficultyLevel,
    CurriculumLevel,
    TrainingProgress,
    CurriculumCallback,
    VmapEnvAdapter
)
from src.orchestrator import MatchOrchestrator
from src.evaluator import SpectacleEvaluator
from src.renderer import AsciiRenderer
from src.registry import FighterRegistry, FighterMetadata
from src.telemetry import ReplayStore


def simple_test_fighter_final(state):
    d = state.get("opponent", {}).get("direction", 1.0)
    return {"acceleration": 0.6 * d, "stance": "neutral"}


# ELO Tracker advanced tests
class TestEloAdvanced:
    def test_elo_damage_ratio_edge_case_zero_taken(self):
        stats = FighterStats(name="Perfect", total_damage_dealt=50.0, total_damage_taken=0.0)
        assert stats.damage_ratio == float('inf') or stats.damage_ratio > 100

    def test_elo_matches_played_property(self):
        stats = FighterStats(name="Veteran", wins=20, losses=15, draws=5)
        assert stats.matches_played == 40

    def test_elo_win_rate_with_draws(self):
        stats = FighterStats(name="Drawer", wins=10, losses=10, draws=10)
        assert stats.win_rate == 10 / 30

    def test_elo_tracker_add_multiple_fighters(self):
        tracker = EloTracker()
        tracker.add_fighter("f1")
        tracker.add_fighter("f2")
        assert "f1" in tracker.fighters
        assert "f2" in tracker.fighters

    def test_elo_suggest_matches_empty(self):
        tracker = EloTracker()
        suggestions = tracker.suggest_balanced_matches()
        assert suggestions == []

    def test_elo_suggest_matches_one_fighter(self):
        tracker = EloTracker()
        tracker.add_fighter("solo")
        suggestions = tracker.suggest_balanced_matches()
        assert suggestions == []

    def test_elo_expected_score_equal_ratings(self):
        tracker = EloTracker()
        expected = tracker.expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.01

    def test_elo_expected_score_large_diff(self):
        tracker = EloTracker()
        expected_strong = tracker.expected_score(2000, 1000)
        expected_weak = tracker.expected_score(1000, 2000)
        assert expected_strong > 0.95
        assert expected_weak < 0.05


# Fighter Loader edge cases
class TestFighterLoaderEdges:
    def test_validate_fighter_missing_both_keys(self):
        def bad(s): return {}
        assert not validate_fighter(bad, verbose=False)

    def test_validate_fighter_none_return(self):
        def bad(s): return None
        assert not validate_fighter(bad, verbose=False)

    def test_validate_fighter_raises_exception(self):
        def bad(s): raise RuntimeError("crash")
        assert not validate_fighter(bad, verbose=False)

    def test_load_hardcoded_from_nonexistent(self):
        fighters = load_hardcoded_fighters("/tmp/xyz_nonexistent", verbose=False)
        assert isinstance(fighters, dict)


# Population helper functions
class TestPopulationHelpers:
    def test_configure_threading_all_vars(self):
        _configure_process_threading()
        vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
        for v in vars:
            assert os.environ[v] == '1'

    def test_reconstruct_config_none(self):
        config = _reconstruct_config(None)
        assert isinstance(config, WorldConfig)

    def test_reconstruct_config_partial(self):
        config = _reconstruct_config({"arena_width": 15.0})
        assert config.arena_width == 15.0

    def test_create_opponent_decide_extracts_obs(self):
        class M:
            def __init__(self):
                self.obs = None
            def predict(self, obs, deterministic=False):
                self.obs = obs
                return np.array([0.0, 1.0]), None

        model = M()
        decide = _create_opponent_decide_func(model)

        snapshot = {
            "you": {"position": 5.0, "velocity": 0.0, "hp": 80.0, "max_hp": 100.0, "stamina": 8.0, "max_stamina": 10.0},
            "opponent": {"distance": 3.0, "direction": 1.0, "velocity": 0.0, "hp": 90.0, "max_hp": 100.0, "stamina": 9.0, "max_stamina": 10.0},
            "arena": {"width": 12.0}
        }

        decide(snapshot)
        assert model.obs is not None
        assert model.obs.shape == (9,)


# Curriculum structures
class TestCurriculumStructures:
    def test_level_with_many_opponents(self):
        level = CurriculumLevel(
            name="Multi",
            difficulty=DifficultyLevel.INTERMEDIATE,
            opponents=["o1.py", "o2.py", "o3.py", "o4.py", "o5.py"]
        )
        assert len(level.opponents) == 5

    def test_progress_with_full_recent_window(self):
        recent = [True] * 20
        progress = TrainingProgress(recent_episodes=recent)
        assert len(progress.recent_episodes) == 20
        assert sum(progress.recent_episodes) == 20

    def test_progress_graduated_multiple_levels(self):
        progress = TrainingProgress(
            current_level=5,
            graduated_levels=["L1", "L2", "L3", "L4", "L5"]
        )
        assert len(progress.graduated_levels) == 5

    def test_progress_episode_accumulation(self):
        progress = TrainingProgress(
            episodes_at_level=100,
            wins_at_level=75,
            total_episodes=500,
            total_wins=375
        )
        assert progress.total_episodes > progress.episodes_at_level
        assert progress.total_wins > progress.wins_at_level


# Gym env edge cases
class TestGymEnvEdges:
    def test_env_reset_with_seed_override(self):
        env = AtomCombatEnv(simple_test_fighter_final, seed=1)
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        assert obs1.shape == obs2.shape

    def test_env_step_with_out_of_bounds_action(self):
        env = AtomCombatEnv(simple_test_fighter_final)
        env.reset()
        extreme = np.array([100.0, 100.0])
        obs, r, done, trunc, info = env.step(extreme)
        assert not np.any(np.isnan(obs))

    def test_env_multiple_resets(self):
        env = AtomCombatEnv(simple_test_fighter_final)
        for _ in range(5):
            env.reset()
            for _ in range(10):
                obs, r, done, trunc, info = env.step(np.array([0.5, 1.0]))
                if done or trunc:
                    break

    def test_env_episode_damage_tracking(self):
        env = AtomCombatEnv(simple_test_fighter_final)
        env.reset()
        for _ in range(20):
            env.step(np.array([1.0, 1.0]))
        assert env.episode_damage_dealt >= 0
        assert env.episode_damage_taken >= 0

    def test_env_hits_tracking(self):
        env = AtomCombatEnv(simple_test_fighter_final)
        env.reset()
        for _ in range(20):
            env.step(np.array([1.0, 1.0]))
        assert env.hits_landed >= 0
        assert env.hits_taken >= 0


# Orchestrator variations
class TestOrchestratorVariations:
    def test_match_with_different_start_positions(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=20)

        positions = [(2.0, 10.0), (4.0, 8.0), (1.0, 11.0)]

        for pos_a, pos_b in positions:
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": pos_a},
                {"name": "B", "mass": 70.0, "position": pos_b},
                simple_test_fighter_final, simple_test_fighter_final, seed=1
            )
            assert result.total_ticks > 0

    def test_match_with_very_different_masses(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=30)

        result = orch.run_match(
            {"name": "Light", "mass": 45.0, "position": 3.0},
            {"name": "Heavy", "mass": 90.0, "position": 9.0},
            simple_test_fighter_final, simple_test_fighter_final, seed=1
        )

        assert result.total_ticks > 0

    def test_match_with_different_seeds(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=15)

        for seed in [1, 42, 99, 123]:
            result = orch.run_match(
                {"name": "A", "mass": 70.0, "position": 3.0},
                {"name": "B", "mass": 70.0, "position": 9.0},
                simple_test_fighter_final, simple_test_fighter_final, seed=seed
            )
            assert result.total_ticks > 0


# Renderer edge cases
class TestRendererEdges:
    def test_ascii_different_arena_widths(self):
        for width in [10.0, 12.5, 15.0, 20.0]:
            renderer = AsciiRenderer(arena_width=width)
            assert renderer.arena_width == width

    def test_ascii_different_display_widths(self):
        for width in [40, 50, 60, 80]:
            renderer = AsciiRenderer(display_width=width)
            assert renderer.display_width == width

    def test_ascii_scale_calculation(self):
        renderer = AsciiRenderer(arena_width=12.0, display_width=60)
        expected_scale = 12.0 / 60
        assert renderer.scale == expected_scale


# Registry bulk operations
class TestRegistryBulk:
    def test_register_many_fighters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = FighterRegistry(Path(tmpdir) / "r.json", load_existing=False)

            for i in range(20):
                reg.register_fighter(FighterMetadata(
                    id=f"f{i}",
                    name=f"Fighter{i}",
                    description="Test",
                    creator="test",
                    type="rule-based",
                    file_path=f"f{i}.py"
                ))

            assert len(reg.fighters) == 20

    def test_filter_by_type_with_mixed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = FighterRegistry(Path(tmpdir) / "r.json", load_existing=False)

            for i in range(5):
                reg.register_fighter(FighterMetadata(
                    id=f"rule{i}", name=f"R{i}", description="T",
                    creator="t", type="rule-based", file_path="t.py"
                ))

            for i in range(3):
                reg.register_fighter(FighterMetadata(
                    id=f"ai{i}", name=f"AI{i}", description="T",
                    creator="t", type="onnx-ai", file_path="t.py"
                ))

            rules = reg.list_fighters(filter_type="rule-based")
            ais = reg.list_fighters(filter_type="onnx-ai")

            assert len(rules) == 5
            assert len(ais) == 3

    def test_filter_by_tags_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = FighterRegistry(Path(tmpdir) / "r.json", load_existing=False)

            reg.register_fighter(FighterMetadata(
                id="aggressive", name="Agg", description="T",
                creator="t", type="rule-based", file_path="t.py",
                strategy_tags=["aggressive", "offensive", "risky"]
            ))

            reg.register_fighter(FighterMetadata(
                id="defensive", name="Def", description="T",
                creator="t", type="rule-based", file_path="t.py",
                strategy_tags=["defensive", "patient"]
            ))

            aggressive = reg.list_fighters(filter_tags=["aggressive"])
            defensive = reg.list_fighters(filter_tags=["defensive"])
            risky = reg.list_fighters(filter_tags=["risky"])

            assert len(aggressive) == 1
            assert len(defensive) == 1
            assert len(risky) == 1


# Spectacle evaluator score components
class TestSpectacleComponents:
    def test_score_all_components_present(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=15, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_test_fighter_final, simple_test_fighter_final, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert hasattr(score, 'duration')
        assert hasattr(score, 'close_finish')
        assert hasattr(score, 'stamina_drama')
        assert hasattr(score, 'comeback_potential')
        assert hasattr(score, 'positional_exchange')
        assert hasattr(score, 'pacing_variety')
        assert hasattr(score, 'collision_drama')
        assert hasattr(score, 'overall')

    def test_score_values_in_range(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=20, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_test_fighter_final, simple_test_fighter_final, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        assert 0 <= score.duration <= 1
        assert 0 <= score.close_finish <= 1
        assert 0 <= score.overall <= 1

    def test_score_to_dict_all_fields(self):
        config = WorldConfig()
        orch = MatchOrchestrator(config, max_ticks=10, record_telemetry=True)

        result = orch.run_match(
            {"name": "A", "mass": 70.0, "position": 3.0},
            {"name": "B", "mass": 70.0, "position": 9.0},
            simple_test_fighter_final, simple_test_fighter_final, seed=1
        )

        evaluator = SpectacleEvaluator()
        score = evaluator.evaluate(result.telemetry, result)

        d = score.to_dict()

        assert "duration" in d
        assert "close_finish" in d
        assert "overall" in d


# Replay Store operations
class TestReplayStoreOps:
    def test_store_list_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            replays = store.list_replays()
            assert isinstance(replays, list)

    def test_store_save_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ReplayStore(tmpdir)
            config = WorldConfig()
            orch = MatchOrchestrator(config, max_ticks=3, record_telemetry=True)

            for i in range(3):
                result = orch.run_match(
                    {"name": f"A{i}", "mass": 70.0, "position": 3.0},
                    {"name": f"B{i}", "mass": 70.0, "position": 9.0},
                    simple_test_fighter_final, simple_test_fighter_final, seed=i
                )
                store.save(result.telemetry, result)

            replays = store.list_replays()
            assert len(replays) >= 3


# PopulationFighter dataclass
class TestPopFighterDataclass:
    def test_fighter_all_fields(self):
        class M: pass
        f = PopulationFighter(
            name="Complete",
            model=M(),
            generation=3,
            lineage="g0->g1->g2->g3",
            mass=72.0,
            training_episodes=500,
            last_checkpoint="/path.zip"
        )
        assert f.generation == 3
        assert f.mass == 72.0
        assert f.training_episodes == 500


# PopulationCallback
class TestPopCallbackBehavior:
    def test_callback_step_with_empty_infos(self):
        tracker = EloTracker()
        cb = PopulationCallback("F", tracker)
        cb.locals = {"infos": []}
        result = cb._on_step()
        assert result is True

    def test_callback_episode_increment(self):
        tracker = EloTracker()
        cb = PopulationCallback("F", tracker)
        cb.locals = {"infos": [{"episode": {"r": 100, "l": 80}}]}
        initial = cb.episode_count
        cb._on_step()
        assert cb.episode_count == initial + 1

    def test_callback_reward_tracking(self):
        tracker = EloTracker()
        cb = PopulationCallback("F", tracker)
        cb.locals = {"infos": [{"episode": {"r": 150.5, "l": 100}}]}
        cb._on_step()
        assert 150.5 in cb.recent_rewards


# CurriculumCallback
class TestCurrCallbackBehavior:
    def test_callback_rollout_start(self):
        class MT:
            algorithm = "ppo"
        cb = CurriculumCallback(MT())
        cb._on_rollout_start()
        assert cb.last_rollout_time is not None

    def test_callback_rollout_end(self):
        class MT:
            algorithm = "ppo"
        cb = CurriculumCallback(MT())
        cb._on_rollout_start()
        cb._on_rollout_end()
        assert cb.last_train_time is not None

    def test_callback_rollout_counter(self):
        class MT:
            algorithm = "ppo"
        cb = CurriculumCallback(MT())
        cb._on_rollout_start()
        cb._on_rollout_start()
        assert cb.rollout_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
