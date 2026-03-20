"""Parity tests for shared reward/observation signal engine."""

import numpy as np

from src.training.signal_engine import (
    build_observation,
    build_observation_batch,
    compute_step_reward_scalar,
    compute_step_rewards_batch,
)


def test_observation_batch_matches_scalar():
    n = 4
    you_pos = np.array([2.0, 4.0, 8.0, 12.0], dtype=np.float32)
    you_vel = np.array([0.5, -0.2, 0.0, 1.0], dtype=np.float32)
    you_hp = np.array([80.0, 70.0, 60.0, 50.0], dtype=np.float32)
    you_max_hp = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    you_stamina = np.array([50.0, 40.0, 30.0, 20.0], dtype=np.float32)
    you_max_stamina = np.array([60.0, 60.0, 60.0, 60.0], dtype=np.float32)

    opp_pos = np.array([10.0, 11.0, 7.0, 1.0], dtype=np.float32)
    opp_vel = np.array([-0.3, 0.1, -0.5, 0.2], dtype=np.float32)
    opp_hp = np.array([75.0, 65.0, 55.0, 45.0], dtype=np.float32)
    opp_max_hp = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    opp_stamina = np.array([45.0, 35.0, 25.0, 15.0], dtype=np.float32)
    opp_max_stamina = np.array([60.0, 60.0, 60.0, 60.0], dtype=np.float32)
    opp_stance = np.array([0, 1, 2, 1], dtype=np.int32)
    recent_damage = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)

    batch = build_observation_batch(
        you_position=you_pos,
        you_velocity=you_vel,
        you_hp=you_hp,
        you_max_hp=you_max_hp,
        you_stamina=you_stamina,
        you_max_stamina=you_max_stamina,
        opponent_position=opp_pos,
        opponent_velocity=opp_vel,
        opponent_hp=opp_hp,
        opponent_max_hp=opp_max_hp,
        opponent_stamina=opp_stamina,
        opponent_max_stamina=opp_max_stamina,
        opponent_stance=opp_stance,
        arena_width=15.0,
        recent_damage=recent_damage,
    )

    assert batch.shape == (n, 13)

    for i in range(n):
        scalar = build_observation(
            you_position=float(you_pos[i]),
            you_velocity=float(you_vel[i]),
            you_hp=float(you_hp[i]),
            you_max_hp=float(you_max_hp[i]),
            you_stamina=float(you_stamina[i]),
            you_max_stamina=float(you_max_stamina[i]),
            opponent_position=float(opp_pos[i]),
            opponent_velocity=float(opp_vel[i]),
            opponent_hp=float(opp_hp[i]),
            opponent_max_hp=float(opp_max_hp[i]),
            opponent_stamina=float(opp_stamina[i]),
            opponent_max_stamina=float(opp_max_stamina[i]),
            opponent_stance=int(opp_stance[i]),
            arena_width=15.0,
            recent_damage=float(recent_damage[i]),
        )
        assert np.allclose(scalar, batch[i], atol=1e-6)


def test_reward_batch_matches_scalar():
    dones = np.array([False, True, False, False], dtype=bool)
    truncated = np.array([False, False, True, False], dtype=bool)
    damage_dealt = np.array([1.2, 0.0, 0.0, 0.0], dtype=np.float32)
    damage_taken = np.array([0.2, 2.0, 0.0, 0.0], dtype=np.float32)
    fighter_hp_pct = np.array([0.8, 0.4, 0.5, 0.7], dtype=np.float32)
    opp_hp_pct = np.array([0.7, 0.6, 0.5, 0.2], dtype=np.float32)
    stamina_pct = np.array([0.9, 0.3, 0.1, 0.2], dtype=np.float32)
    opp_stamina_pct = np.array([0.6, 0.4, 0.2, 0.8], dtype=np.float32)
    fighter_stance = np.array([1, 2, 0, 1], dtype=np.int32)
    distance = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)
    last_distance = np.array([1.3, 1.9, 3.8, 0.8], dtype=np.float32)
    tick_counts = np.array([10, 200, 250, 50], dtype=np.int32)
    episode_damage_dealt = np.array([10.0, 50.0, 3.0, 5.0], dtype=np.float32)
    episode_stamina_used = np.array([5.0, 20.0, 4.0, 8.0], dtype=np.float32)

    batch = compute_step_rewards_batch(
        dones=dones,
        truncated=truncated,
        damage_dealt=damage_dealt,
        damage_taken=damage_taken,
        fighter_hp_pct=fighter_hp_pct,
        opponent_hp_pct=opp_hp_pct,
        stamina_pct=stamina_pct,
        opp_stamina_pct=opp_stamina_pct,
        fighter_stance=fighter_stance,
        distance=distance,
        last_distance=last_distance,
        tick_counts=tick_counts,
        max_ticks=250,
        arena_width=15.0,
        episode_damage_dealt=episode_damage_dealt,
        episode_stamina_used=episode_stamina_used,
    )

    for i in range(len(dones)):
        scalar = compute_step_reward_scalar(
            done=bool(dones[i]),
            truncated=bool(truncated[i]),
            damage_dealt=float(damage_dealt[i]),
            damage_taken=float(damage_taken[i]),
            fighter_hp_pct=float(fighter_hp_pct[i]),
            opponent_hp_pct=float(opp_hp_pct[i]),
            stamina_pct=float(stamina_pct[i]),
            opp_stamina_pct=float(opp_stamina_pct[i]),
            fighter_stance=int(fighter_stance[i]),
            distance=float(distance[i]),
            last_distance=float(last_distance[i]),
            tick_count=int(tick_counts[i]),
            max_ticks=250,
            arena_width=15.0,
            episode_damage_dealt=float(episode_damage_dealt[i]),
            episode_stamina_used=float(episode_stamina_used[i]),
        )
        assert np.isclose(scalar.reward, batch.rewards[i], atol=1e-6)
        assert np.isclose(scalar.damage_component, batch.damage_component[i], atol=1e-6)
        assert np.isclose(scalar.proximity_component, batch.proximity_component[i], atol=1e-6)
        assert np.isclose(scalar.stamina_component, batch.stamina_component[i], atol=1e-6)
        assert np.isclose(scalar.stance_component, batch.stance_component[i], atol=1e-6)
        assert np.isclose(scalar.inaction_component, batch.inaction_component[i], atol=1e-6)
        assert np.isclose(scalar.terminal_component, batch.terminal_component[i], atol=1e-6)
