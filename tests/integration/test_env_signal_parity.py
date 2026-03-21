"""Cross-env parity checks between AtomCombatEnv and VmapEnvWrapper (n_envs=1)."""

import numpy as np

from src.atom.training.gym_env import AtomCombatEnv
from src.atom.training.vmap_env_wrapper import VmapEnvWrapper


def _neutral_opponent(_snapshot):
    return {"acceleration": 0.0, "stance": "neutral"}


def _directional_opponent(snapshot):
    direction = snapshot["opponent"]["direction"]
    distance = snapshot["opponent"]["distance"]
    if distance > 2.0:
        return {"acceleration": 0.7 * direction, "stance": "neutral"}
    return {"acceleration": 0.3 * direction, "stance": "extended"}


def test_initial_observation_parity_single_vs_vmap():
    env_single = AtomCombatEnv(
        opponent_decision_func=_neutral_opponent,
        max_ticks=250,
        seed=42,
    )
    obs_single, _ = env_single.reset(seed=42)

    env_vmap = VmapEnvWrapper(
        n_envs=1,
        opponent_decision_func=_neutral_opponent,
        max_ticks=250,
        seed=42,
    )
    obs_vmap, _ = env_vmap.reset(seed=42)

    assert obs_vmap.shape == (1, 13)
    assert np.allclose(obs_single, obs_vmap[0], atol=1e-5)


def test_first_step_reward_and_observation_parity_single_vs_vmap():
    env_single = AtomCombatEnv(
        opponent_decision_func=_neutral_opponent,
        max_ticks=250,
        seed=123,
    )
    env_single.reset(seed=123)

    env_vmap = VmapEnvWrapper(
        n_envs=1,
        opponent_decision_func=_neutral_opponent,
        max_ticks=250,
        seed=123,
    )
    env_vmap.reset(seed=123)

    action = np.array([0.0, 0.0], dtype=np.float32)

    obs_single, reward_single, done_single, trunc_single, _ = env_single.step(action)
    obs_vmap, rewards_vmap, dones_vmap, trunc_vmap, _ = env_vmap.step(action.reshape(1, 2))

    assert np.allclose(obs_single, obs_vmap[0], atol=1e-5)
    assert np.isclose(float(reward_single), float(rewards_vmap[0]), atol=1e-5)
    assert bool(done_single) == bool(dones_vmap[0])
    assert bool(trunc_single) == bool(trunc_vmap[0])


def test_multistep_parity_with_legacy_opponent_decision_func():
    env_single = AtomCombatEnv(
        opponent_decision_func=_directional_opponent,
        max_ticks=100,
        seed=2026,
    )
    env_single.reset(seed=2026)

    env_vmap = VmapEnvWrapper(
        n_envs=1,
        opponent_decision_func=_directional_opponent,
        max_ticks=100,
        seed=2026,
    )
    env_vmap.reset(seed=2026)

    rng = np.random.default_rng(2026)
    for _ in range(20):
        action = np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.99)],
            dtype=np.float32,
        )
        obs_single, reward_single, done_single, trunc_single, _ = env_single.step(action)
        obs_vmap, rewards_vmap, dones_vmap, trunc_vmap, _ = env_vmap.step(action.reshape(1, 2))

        assert np.allclose(obs_single, obs_vmap[0], atol=1e-4)
        assert np.isclose(float(reward_single), float(rewards_vmap[0]), atol=1e-4)
        assert bool(done_single) == bool(dones_vmap[0])
        assert bool(trunc_single) == bool(trunc_vmap[0])

        if done_single or trunc_single:
            break


def test_episode_end_reward_breakdown_parity():
    env_single = AtomCombatEnv(
        opponent_decision_func=_neutral_opponent,
        max_ticks=1,
        seed=77,
    )
    env_single.reset(seed=77)

    env_vmap = VmapEnvWrapper(
        n_envs=1,
        opponent_decision_func=_neutral_opponent,
        max_ticks=1,
        seed=77,
    )
    env_vmap.reset(seed=77)

    action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, done_single, trunc_single, info_single = env_single.step(action)
    _, _, dones_vmap, trunc_vmap, infos_vmap = env_vmap.step(action.reshape(1, 2))

    assert done_single or trunc_single
    assert bool(dones_vmap[0]) or bool(trunc_vmap[0])

    breakdown_single = info_single.get("reward_breakdown")
    breakdown_vmap = infos_vmap[0].get("reward_breakdown")

    assert breakdown_single is not None
    assert breakdown_vmap is not None

    for key in ["proximity", "damage", "stamina", "stance", "inaction", "terminal", "total"]:
        assert key in breakdown_vmap
        assert np.isclose(
            float(breakdown_single[key]),
            float(breakdown_vmap[key]),
            atol=1e-4,
        )
