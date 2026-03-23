#!/usr/bin/env python3
"""
Tests for curriculum structure, JAX registry coverage, holdout integrity,
and Python/JAX parity for the 7 new fighters.
"""

import importlib.util
import math

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from src.atom.training.trainers.curriculum_trainer import CurriculumTrainer, DifficultyLevel
from src.atom.training.opponents_jax import JAX_OPPONENT_REGISTRY

# JAX types needed for parity tests
from src.atom.runtime.arena.arena_1d_jax_jit import (
    FighterStateJAX,
    ArenaStateJAX,
    STANCE_TO_INT,
)
from src.atom.runtime.arena import WorldConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

FIGHTERS_DIR = PROJECT_ROOT / "fighters" / "test_dummies" / "atomic"

NEW_FIGHTER_NAMES = [
    "approach_extended",
    "flee_defending",
    "stamina_burner",
    "hp_adaptive",
    "stamina_punisher",
    "range_switcher",
    "comeback_fighter",
]

EXPECTED_LEVEL_NAMES = [
    "Fundamentals",
    "Basic Skills",
    "Intermediate",
    "Advanced",
    "Adaptive",
    "Expert",
    "Gauntlet",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_curriculum():
    """Instantiate a CurriculumTrainer just enough to get the curriculum list."""
    # CurriculumTrainer.__init__ does heavy work; call _build_curriculum directly
    # via an unbound approach: create a minimal instance.
    trainer = object.__new__(CurriculumTrainer)
    return trainer._build_curriculum()


def _load_python_fighter(name: str):
    """Load a Python fighter module by stem name."""
    path = FIGHTERS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_holdout_suite():
    """Get the holdout suite from a minimal CurriculumTrainer."""
    trainer = object.__new__(CurriculumTrainer)
    return trainer._get_holdout_suite()


def _make_python_state(
    *,
    tick=0,
    position=5.0,
    velocity=0.0,
    hp=93.7,
    max_hp=93.7,
    stamina=8.9,
    max_stamina=8.9,
    stance="neutral",
    opp_distance=4.0,
    opp_direction=1.0,
    opp_velocity=0.0,
    opp_hp=93.7,
    opp_max_hp=93.7,
    opp_stamina=8.9,
    opp_max_stamina=8.9,
    opp_stance_hint="neutral",
    arena_width=12.476,
    opp_position=None,
):
    """Build a Python snapshot dict for decide()."""
    return {
        "tick": tick,
        "you": {
            "position": position,
            "velocity": velocity,
            "hp": hp,
            "max_hp": max_hp,
            "stamina": stamina,
            "max_stamina": max_stamina,
            "stance": stance,
        },
        "opponent": {
            "distance": opp_distance,
            "direction": opp_direction,
            "velocity": opp_velocity,
            "hp": opp_hp,
            "max_hp": opp_max_hp,
            "stamina": opp_stamina,
            "max_stamina": opp_max_stamina,
            "stance_hint": opp_stance_hint,
        },
        "arena": {"width": arena_width},
    }


def _make_jax_state(
    *,
    tick=0,
    # fighter_a is the learner (opponent from the dummy's perspective)
    a_position=1.0,
    a_velocity=0.0,
    a_hp=93.7,
    a_max_hp=93.7,
    a_stamina=8.9,
    a_max_stamina=8.9,
    a_stance=0,
    # fighter_b is the dummy being tested
    b_position=5.0,
    b_velocity=0.0,
    b_hp=93.7,
    b_max_hp=93.7,
    b_stamina=8.9,
    b_max_stamina=8.9,
    b_stance=0,
):
    """Build an ArenaStateJAX for JAX opponent functions.

    In JAX opponents, fighter_b is "self" and fighter_a is the learner.
    """
    fighter_a = FighterStateJAX(
        mass=70.0,
        position=float(a_position),
        velocity=float(a_velocity),
        hp=float(a_hp),
        max_hp=float(a_max_hp),
        stamina=float(a_stamina),
        max_stamina=float(a_max_stamina),
        stance=int(a_stance),
        last_hit_tick=-100,
    )
    fighter_b = FighterStateJAX(
        mass=70.0,
        position=float(b_position),
        velocity=float(b_velocity),
        hp=float(b_hp),
        max_hp=float(b_max_hp),
        stamina=float(b_stamina),
        max_stamina=float(b_max_stamina),
        stance=int(b_stance),
        last_hit_tick=-100,
    )
    return ArenaStateJAX(fighter_a=fighter_a, fighter_b=fighter_b, tick=tick)


def _paired_states(
    *,
    tick=0,
    b_position=5.0,
    a_position=1.0,
    b_hp=93.7,
    b_max_hp=93.7,
    a_hp=93.7,
    a_max_hp=93.7,
    b_stamina=8.9,
    b_max_stamina=8.9,
    a_stamina=8.9,
    a_max_stamina=8.9,
    arena_width=12.476,
):
    """Return (python_state, jax_state, config) that represent the same game state.

    Convention:
        - fighter_a = learner (the "opponent" from the dummy's viewpoint)
        - fighter_b = the dummy being tested
    """
    distance = abs(a_position - b_position)
    direction = 1.0 if a_position > b_position else (-1.0 if a_position < b_position else 0.0)

    py_state = _make_python_state(
        tick=tick,
        position=b_position,
        hp=b_hp,
        max_hp=b_max_hp,
        stamina=b_stamina,
        max_stamina=b_max_stamina,
        opp_distance=distance,
        opp_direction=direction,
        opp_hp=a_hp,
        opp_max_hp=a_max_hp,
        opp_stamina=a_stamina,
        opp_max_stamina=a_max_stamina,
        arena_width=arena_width,
    )

    jax_state = _make_jax_state(
        tick=tick,
        a_position=a_position,
        b_position=b_position,
        a_hp=a_hp,
        a_max_hp=a_max_hp,
        a_stamina=a_stamina,
        a_max_stamina=a_max_stamina,
        b_hp=b_hp,
        b_max_hp=b_max_hp,
        b_stamina=b_stamina,
        b_max_stamina=b_max_stamina,
    )

    config = WorldConfig(arena_width=arena_width)
    return py_state, jax_state, config


# Map Python stance strings to JAX stance ints
_STANCE_STR_TO_INT = {"neutral": 0, "extended": 1, "defending": 2}

# ---------------------------------------------------------------------------
# 1. Curriculum has exactly 7 levels
# ---------------------------------------------------------------------------

class TestCurriculumStructure:

    def test_exactly_seven_levels(self):
        curriculum = _build_curriculum()
        assert len(curriculum) == 7

    # -------------------------------------------------------------------
    # 2. DifficultyLevel enum has ADAPTIVE and GAUNTLET
    # -------------------------------------------------------------------

    def test_difficulty_enum_adaptive(self):
        assert hasattr(DifficultyLevel, "ADAPTIVE")

    def test_difficulty_enum_gauntlet(self):
        assert hasattr(DifficultyLevel, "GAUNTLET")

    # -------------------------------------------------------------------
    # 3. All opponent paths in all 7 levels exist on disk
    # -------------------------------------------------------------------

    def test_all_opponent_paths_exist(self):
        curriculum = _build_curriculum()
        for level in curriculum:
            for opp_path in level.opponents:
                assert (PROJECT_ROOT / opp_path).exists(), (
                    f"Missing opponent file: {opp_path} (level {level.name!r})"
                )

    # -------------------------------------------------------------------
    # 4. Level names match expected
    # -------------------------------------------------------------------

    def test_level_names(self):
        curriculum = _build_curriculum()
        names = [lvl.name for lvl in curriculum]
        assert names == EXPECTED_LEVEL_NAMES

    # -------------------------------------------------------------------
    # 5. Graduation requirements spot checks
    # -------------------------------------------------------------------

    def test_l1_graduation_win_rate(self):
        curriculum = _build_curriculum()
        assert curriculum[0].graduation_win_rate == pytest.approx(0.9)

    def test_l7_graduation_win_rate(self):
        curriculum = _build_curriculum()
        assert curriculum[6].graduation_win_rate == pytest.approx(0.75)

    def test_l5_l6_graduation_episodes(self):
        curriculum = _build_curriculum()
        assert curriculum[4].graduation_episodes == 75  # L5 (Adaptive)
        assert curriculum[5].graduation_episodes == 75  # L6 (Expert)


# ---------------------------------------------------------------------------
# 6. JAX registry coverage: every atomic opponent in L1-L5 + L7 has a JAX entry
# ---------------------------------------------------------------------------

class TestJAXRegistryCoverage:

    def test_all_atomic_opponents_have_jax_entries(self):
        curriculum = _build_curriculum()
        missing = []
        # Levels to check: L1 (idx 0) through L5 (idx 4), and L7 (idx 6)
        check_indices = [0, 1, 2, 3, 4, 6]
        for idx in check_indices:
            level = curriculum[idx]
            for opp_path in level.opponents:
                stem = Path(opp_path).stem
                # Level 7 (Gauntlet) may include fighters/examples/* which
                # don't need JAX entries; only check test_dummies/atomic
                if "test_dummies/atomic" in opp_path:
                    if stem not in JAX_OPPONENT_REGISTRY:
                        missing.append(f"L{idx+1}/{level.name}: {stem}")
        assert missing == [], f"Missing JAX entries: {missing}"


# ---------------------------------------------------------------------------
# 7. Holdout suite integrity
# ---------------------------------------------------------------------------

class TestHoldoutSuiteIntegrity:

    def test_all_holdout_paths_exist(self):
        suite = _get_holdout_suite()
        for entry in suite:
            path = PROJECT_ROOT / entry["opponent_path"]
            assert path.exists(), f"Missing holdout opponent: {entry['opponent_path']}"

    def test_holdout_has_adaptive_category(self):
        suite = _get_holdout_suite()
        categories = {e["category"] for e in suite}
        assert "adaptive" in categories

    def test_holdout_has_12_opponents(self):
        suite = _get_holdout_suite()
        assert len(suite) == 12, f"Expected 12 holdout opponents, got {len(suite)}"

    def test_holdout_six_categories(self):
        suite = _get_holdout_suite()
        categories = {e["category"] for e in suite}
        assert len(categories) == 6, f"Expected 6 categories, got {categories}"

    def test_holdout_two_per_category(self):
        suite = _get_holdout_suite()
        from collections import Counter
        counts = Counter(e["category"] for e in suite)
        for cat, count in counts.items():
            assert count == 2, f"Category {cat!r} has {count} opponents, expected 2"


# ---------------------------------------------------------------------------
# 8. Python/JAX parity
# ---------------------------------------------------------------------------

class TestPythonJAXParity:
    """For each of the 7 new fighters, verify that the Python decide() and the
    corresponding JAX function produce the same stance and similar acceleration
    when given equivalent game states."""

    ACCEL_TOLERANCE = 1.1  # Python and JAX use different spacing heuristics; allow for that

    # Two representative states: close range and far range
    STATE_CONFIGS = [
        {  # Close range: dummy at position 5, learner at 4
            "b_position": 5.0,
            "a_position": 4.0,
            "tick": 0,
            "b_hp": 93.7,
            "b_max_hp": 93.7,
            "a_hp": 93.7,
            "a_max_hp": 93.7,
            "b_stamina": 8.9,
            "b_max_stamina": 8.9,
            "a_stamina": 8.9,
            "a_max_stamina": 8.9,
        },
        {  # Far range: dummy at position 2, learner at 10
            "b_position": 2.0,
            "a_position": 10.0,
            "tick": 50,
            "b_hp": 70.0,
            "b_max_hp": 93.7,
            "a_hp": 60.0,
            "a_max_hp": 93.7,
            "b_stamina": 5.0,
            "b_max_stamina": 8.9,
            "a_stamina": 3.0,
            "a_max_stamina": 8.9,
        },
    ]

    @pytest.mark.parametrize("fighter_name", NEW_FIGHTER_NAMES)
    def test_parity(self, fighter_name):
        py_mod = _load_python_fighter(fighter_name)
        jax_id, jax_fn = JAX_OPPONENT_REGISTRY[fighter_name]
        config = WorldConfig()

        for i, cfg in enumerate(self.STATE_CONFIGS):
            py_state, jax_state, _ = _paired_states(**cfg)
            py_result = py_mod.decide(py_state)
            jax_result = jax_fn(jax_state, config)

            jax_accel = float(jax_result[0])
            jax_stance_int = int(jax_result[1])
            py_stance_int = _STANCE_STR_TO_INT[py_result["stance"]]
            py_accel = float(py_result["acceleration"])

            # Stance must match exactly
            assert py_stance_int == jax_stance_int, (
                f"{fighter_name} state#{i}: Python stance={py_result['stance']} "
                f"({py_stance_int}) != JAX stance={jax_stance_int}"
            )

            # Acceleration must be close (direction match is critical)
            if py_accel != 0.0 or jax_accel != 0.0:
                # At minimum, signs should agree (or one is zero)
                if abs(py_accel) > 0.1 and abs(jax_accel) > 0.1:
                    assert (py_accel > 0) == (jax_accel > 0), (
                        f"{fighter_name} state#{i}: accel sign mismatch: "
                        f"Python={py_accel:.2f}, JAX={jax_accel:.2f}"
                    )
                assert abs(py_accel - jax_accel) < self.ACCEL_TOLERANCE, (
                    f"{fighter_name} state#{i}: accel mismatch: "
                    f"Python={py_accel:.2f}, JAX={jax_accel:.2f}"
                )
