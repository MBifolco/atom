import numpy as np

from src.atom.training.signal_engine import build_observation, build_observation_from_snapshot


def _base_snapshot():
    return {
        "you": {
            "position": 2.0,
            "velocity": 0.5,
            "hp": 80.0,
            "max_hp": 100.0,
            "stamina": 6.0,
            "max_stamina": 10.0,
            "stance": "neutral",
        },
        "opponent": {
            "distance": 5.0,
            "direction": 1.0,
            "velocity": 1.2,
            "hp": 70.0,
            "max_hp": 100.0,
            "stamina": 4.0,
            "max_stamina": 10.0,
            "stance_hint": "extended",
        },
        "arena": {"width": 12.0},
    }


def test_snapshot_adapter_matches_builder_for_opponent_on_right():
    snapshot = _base_snapshot()
    obs_from_snapshot = build_observation_from_snapshot(snapshot)

    expected = build_observation(
        you_position=2.0,
        you_velocity=0.5,
        you_hp=80.0,
        you_max_hp=100.0,
        you_stamina=6.0,
        you_max_stamina=10.0,
        opponent_position=7.0,
        opponent_velocity=1.7,
        opponent_hp=70.0,
        opponent_max_hp=100.0,
        opponent_stamina=4.0,
        opponent_max_stamina=10.0,
        opponent_stance="extended",
        arena_width=12.0,
        opponent_direction=1.0,
        hit_cooldown_fraction=1.0,
    )

    assert obs_from_snapshot.shape == (18,)
    np.testing.assert_allclose(obs_from_snapshot, expected, rtol=1e-6, atol=1e-6)


def test_snapshot_adapter_matches_builder_for_opponent_on_left():
    snapshot = _base_snapshot()
    snapshot["you"]["position"] = 9.0
    snapshot["you"]["velocity"] = -0.4
    snapshot["opponent"]["distance"] = 3.0
    snapshot["opponent"]["direction"] = -1.0
    snapshot["opponent"]["velocity"] = 0.9
    snapshot["opponent"]["stance_hint"] = 2

    obs_from_snapshot = build_observation_from_snapshot(snapshot)

    expected = build_observation(
        you_position=9.0,
        you_velocity=-0.4,
        you_hp=80.0,
        you_max_hp=100.0,
        you_stamina=6.0,
        you_max_stamina=10.0,
        opponent_position=6.0,
        opponent_velocity=-1.3,
        opponent_hp=70.0,
        opponent_max_hp=100.0,
        opponent_stamina=4.0,
        opponent_max_stamina=10.0,
        opponent_stance=2,
        arena_width=12.0,
        opponent_direction=-1.0,
        hit_cooldown_fraction=1.0,
    )

    np.testing.assert_allclose(obs_from_snapshot, expected, rtol=1e-6, atol=1e-6)
    # Opponent stance defending (2) → one-hot [0, 0, 1] at indices 6,7,8
    assert obs_from_snapshot[8] == 1.0
