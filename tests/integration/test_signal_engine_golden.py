"""Golden-value regression tests for canonical signal engine outputs."""

import json
from pathlib import Path

import numpy as np

from src.atom.training.signal_engine import compute_step_reward_scalar


def test_signal_engine_reward_golden_cases():
    fixture_path = Path(__file__).parent / "fixtures" / "signal_reward_golden.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in payload:
        result = compute_step_reward_scalar(**case["input"])
        expected = case["expected"]

        assert np.isclose(result.reward, expected["reward"], atol=1e-6), case["name"]
        assert np.isclose(result.damage_component, expected["damage_component"], atol=1e-6), case["name"]
        assert np.isclose(result.proximity_component, expected["proximity_component"], atol=1e-6), case["name"]
        assert np.isclose(result.stamina_component, expected["stamina_component"], atol=1e-6), case["name"]
        assert np.isclose(result.stance_component, expected["stance_component"], atol=1e-6), case["name"]
        assert np.isclose(result.inaction_component, expected["inaction_component"], atol=1e-6), case["name"]
        assert np.isclose(result.terminal_component, expected["terminal_component"], atol=1e-6), case["name"]
        assert np.isclose(result.next_last_distance, expected["next_last_distance"], atol=1e-6), case["name"]
