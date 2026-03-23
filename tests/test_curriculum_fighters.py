#!/usr/bin/env python3
"""
Tests for the 7 new curriculum fighter files in fighters/test_dummies/atomic/.

Each fighter exposes a decide(state) function that returns
{"acceleration": float, "stance": str}.
"""

import importlib.util
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGHTERS_DIR = Path(__file__).parent.parent / "fighters" / "test_dummies" / "atomic"

NEW_FIGHTERS = [
    "approach_extended",
    "flee_defending",
    "stamina_burner",
    "hp_adaptive",
    "stamina_punisher",
    "range_switcher",
    "comeback_fighter",
]

VALID_STANCES = {"neutral", "extended", "defending"}


def _load_fighter(name: str):
    """Load a fighter module by stem name using importlib."""
    path = FIGHTERS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_state(
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
):
    """Construct a snapshot dict matching the expected state format."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=NEW_FIGHTERS)
def fighter_module(request):
    """Parametrize over all 7 new fighters."""
    return _load_fighter(request.param)


# ---------------------------------------------------------------------------
# 1. Valid output shape for every fighter
# ---------------------------------------------------------------------------

class TestValidOutput:
    """Every new fighter must return a dict with acceleration (float) and stance (valid string)."""

    def test_returns_valid_dict(self, fighter_module):
        state = _base_state()
        result = fighter_module.decide(state)
        assert isinstance(result, dict), "decide() must return a dict"
        assert "acceleration" in result, "result must contain 'acceleration'"
        assert "stance" in result, "result must contain 'stance'"
        assert isinstance(result["acceleration"], (int, float)), "acceleration must be numeric"
        assert result["stance"] in VALID_STANCES, (
            f"stance must be one of {VALID_STANCES}, got {result['stance']!r}"
        )


# ---------------------------------------------------------------------------
# 2. approach_extended always returns "extended"
# ---------------------------------------------------------------------------

class TestApproachExtended:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("approach_extended")

    def test_always_extended(self):
        for dist, direction in [(0.5, 1.0), (6.0, -1.0), (0.0, 0.0)]:
            result = self.mod.decide(_base_state(opp_distance=dist, opp_direction=direction))
            assert result["stance"] == "extended"


# ---------------------------------------------------------------------------
# 3. flee_defending always returns "defending"
# ---------------------------------------------------------------------------

class TestFleeDefending:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("flee_defending")

    def test_always_defending(self):
        for dist, direction in [(0.5, 1.0), (6.0, -1.0), (0.0, 0.0)]:
            result = self.mod.decide(_base_state(opp_distance=dist, opp_direction=direction))
            assert result["stance"] == "defending"


# ---------------------------------------------------------------------------
# 4. stamina_burner: extended when stamina > 40%, neutral when < 20%
# ---------------------------------------------------------------------------

class TestStaminaBurner:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("stamina_burner")

    def test_high_stamina_extended(self):
        # stamina/max_stamina = 8.0/10.0 = 80% > 40%
        result = self.mod.decide(_base_state(stamina=8.0, max_stamina=10.0))
        assert result["stance"] == "extended"

    def test_low_stamina_neutral(self):
        # stamina/max_stamina = 1.0/10.0 = 10% < 20%
        result = self.mod.decide(_base_state(stamina=1.0, max_stamina=10.0))
        assert result["stance"] == "neutral"

    def test_mid_stamina_neutral(self):
        # stamina/max_stamina = 3.0/10.0 = 30%, between 20%-40% dead zone
        result = self.mod.decide(_base_state(stamina=3.0, max_stamina=10.0))
        assert result["stance"] == "neutral"


# ---------------------------------------------------------------------------
# 5. hp_adaptive: extended when HP lead > 10%, defending when deficit > 10%,
#    neutral in dead zone
# ---------------------------------------------------------------------------

class TestHpAdaptive:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("hp_adaptive")

    def test_hp_lead_extended(self):
        # 90/100 - 70/100 = 0.2 > 0.1 => extended
        result = self.mod.decide(_base_state(hp=90, max_hp=100, opp_hp=70, opp_max_hp=100))
        assert result["stance"] == "extended"

    def test_hp_deficit_defending(self):
        # 60/100 - 90/100 = -0.3 < -0.1 => defending
        result = self.mod.decide(_base_state(hp=60, max_hp=100, opp_hp=90, opp_max_hp=100))
        assert result["stance"] == "defending"

    def test_hp_neutral_dead_zone(self):
        # 85/100 - 80/100 = 0.05, within +-0.1 dead zone => neutral
        result = self.mod.decide(_base_state(hp=85, max_hp=100, opp_hp=80, opp_max_hp=100))
        assert result["stance"] == "neutral"


# ---------------------------------------------------------------------------
# 6. stamina_punisher: extended when opponent stamina < 40%, neutral when > 70%
# ---------------------------------------------------------------------------

class TestStaminaPunisher:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("stamina_punisher")

    def test_opp_low_stamina_extended(self):
        # 2.0/10.0 = 20% < 40% => extended
        result = self.mod.decide(_base_state(opp_stamina=2.0, opp_max_stamina=10.0))
        assert result["stance"] == "extended"

    def test_opp_high_stamina_neutral(self):
        # 8.0/10.0 = 80% > 70% => neutral
        result = self.mod.decide(_base_state(opp_stamina=8.0, opp_max_stamina=10.0))
        assert result["stance"] == "neutral"

    def test_opp_mid_stamina_neutral(self):
        # 5.0/10.0 = 50%, dead zone => neutral
        result = self.mod.decide(_base_state(opp_stamina=5.0, opp_max_stamina=10.0))
        assert result["stance"] == "neutral"


# ---------------------------------------------------------------------------
# 7. range_switcher: neutral in safe phase (tick 0-44), extended in burst (45-59)
# ---------------------------------------------------------------------------

class TestRangeSwitcher:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("range_switcher")

    def test_safe_phase_neutral(self):
        for tick in [0, 10, 30, 44]:
            result = self.mod.decide(_base_state(tick=tick))
            assert result["stance"] == "neutral", f"Expected neutral at tick {tick}"

    def test_burst_phase_extended(self):
        for tick in [45, 50, 59]:
            result = self.mod.decide(_base_state(tick=tick))
            assert result["stance"] == "extended", f"Expected extended at tick {tick}"

    def test_cycle_wraps(self):
        # tick=60 wraps to cycle position 0 => safe phase
        result = self.mod.decide(_base_state(tick=60))
        assert result["stance"] == "neutral"
        # tick=105 => cycle 45 => burst
        result = self.mod.decide(_base_state(tick=105))
        assert result["stance"] == "extended"


# ---------------------------------------------------------------------------
# 8. comeback_fighter: defending early (tick 0), extended late (tick 200+)
# ---------------------------------------------------------------------------

class TestComebackFighter:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_fighter("comeback_fighter")

    def test_early_defending(self):
        # tick=0 => aggression=0.0 < 0.3 => defending
        result = self.mod.decide(_base_state(tick=0))
        assert result["stance"] == "defending"

    def test_late_extended(self):
        # tick=200 => aggression=1.0 >= 0.7 => extended
        result = self.mod.decide(_base_state(tick=200))
        assert result["stance"] == "extended"

    def test_mid_neutral_or_extended(self):
        # tick=100 => aggression=0.5, between 0.3 and 0.7
        # stance depends on distance; at distance 4.0 => distance > 1.0 => "neutral"
        result = self.mod.decide(_base_state(tick=100, opp_distance=4.0))
        assert result["stance"] in {"neutral", "extended"}


# ---------------------------------------------------------------------------
# 9. Dead-zone: hp_adaptive at exactly 0.0 HP diff returns neutral
# ---------------------------------------------------------------------------

class TestDeadZone:
    def test_hp_adaptive_exact_zero(self):
        mod = _load_fighter("hp_adaptive")
        # Identical HP fractions => hp_diff = 0.0, in dead zone => neutral
        result = mod.decide(_base_state(hp=80, max_hp=100, opp_hp=80, opp_max_hp=100))
        assert result["stance"] == "neutral"


# ---------------------------------------------------------------------------
# 10. All fighters handle direction == 0 without crashing
# ---------------------------------------------------------------------------

class TestDirectionZero:
    """direction=0 means the opponent is co-located; no fighter should crash."""

    def test_direction_zero_no_crash(self, fighter_module):
        state = _base_state(opp_direction=0.0, opp_distance=0.0)
        result = fighter_module.decide(state)
        assert isinstance(result, dict)
        assert result["stance"] in VALID_STANCES
        # acceleration should still be a finite number
        assert result["acceleration"] == result["acceleration"]  # not NaN
