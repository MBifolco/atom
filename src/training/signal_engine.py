"""
Canonical observation and reward signal engine for training environments.

This module centralizes the reward and observation semantics that were
previously duplicated (and drifting) between:
- AtomCombatEnv (single-environment Gym wrapper)
- VmapEnvWrapper (batched JAX/Gym wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


STANCE_NEUTRAL = 0
STANCE_EXTENDED = 1
STANCE_DEFENDING = 2

_STANCE_NAME_TO_INT = {
    "neutral": STANCE_NEUTRAL,
    "extended": STANCE_EXTENDED,
    "defending": STANCE_DEFENDING,
}


def stance_to_int(stance: int | float | str) -> int:
    """Convert stance representation (int/float/str) to canonical int index."""
    if isinstance(stance, str):
        return _STANCE_NAME_TO_INT.get(stance, STANCE_NEUTRAL)
    try:
        return int(np.clip(int(stance), STANCE_NEUTRAL, STANCE_DEFENDING))
    except Exception:
        return STANCE_NEUTRAL


def _to_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _to_bool_array(values) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def _to_stance_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"i", "u", "f"}:
        return np.clip(arr.astype(np.int32), STANCE_NEUTRAL, STANCE_DEFENDING)
    vectorized = np.vectorize(stance_to_int, otypes=[np.int32])
    return vectorized(arr)


def _relative_velocity(you_pos: np.ndarray, you_vel: np.ndarray, opp_pos: np.ndarray, opp_vel: np.ndarray) -> np.ndarray:
    """
    Canonical relative velocity used by AtomCombatEnv.

    If agent is to the left of opponent, positive means opponent moving away.
    If agent is to the right, sign flips accordingly.
    """
    return np.where(you_pos < opp_pos, opp_vel - you_vel, you_vel - opp_vel)


def build_observation(
    *,
    you_position: float,
    you_velocity: float,
    you_hp: float,
    you_max_hp: float,
    you_stamina: float,
    you_max_stamina: float,
    opponent_position: float,
    opponent_velocity: float,
    opponent_hp: float,
    opponent_max_hp: float,
    opponent_stamina: float,
    opponent_max_stamina: float,
    opponent_stance: int | float | str,
    arena_width: float,
    recent_damage: float,
) -> np.ndarray:
    """Build a single 13-dimensional training observation."""
    obs = build_observation_batch(
        you_position=np.array([you_position], dtype=np.float32),
        you_velocity=np.array([you_velocity], dtype=np.float32),
        you_hp=np.array([you_hp], dtype=np.float32),
        you_max_hp=np.array([you_max_hp], dtype=np.float32),
        you_stamina=np.array([you_stamina], dtype=np.float32),
        you_max_stamina=np.array([you_max_stamina], dtype=np.float32),
        opponent_position=np.array([opponent_position], dtype=np.float32),
        opponent_velocity=np.array([opponent_velocity], dtype=np.float32),
        opponent_hp=np.array([opponent_hp], dtype=np.float32),
        opponent_max_hp=np.array([opponent_max_hp], dtype=np.float32),
        opponent_stamina=np.array([opponent_stamina], dtype=np.float32),
        opponent_max_stamina=np.array([opponent_max_stamina], dtype=np.float32),
        opponent_stance=np.array([opponent_stance], dtype=object),
        arena_width=arena_width,
        recent_damage=np.array([recent_damage], dtype=np.float32),
    )
    return obs[0]


def build_observation_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    recent_damage: float = 0.0,
) -> np.ndarray:
    """
    Build canonical observation from protocol snapshot (`generate_snapshot` format).

    This adapter is used in training/evaluation code paths that operate on
    snapshots rather than live arena states, so all observation semantics still
    flow through one canonical builder.
    """
    you = snapshot["you"]
    opponent = snapshot["opponent"]
    arena = snapshot["arena"]

    you_position = float(you["position"])
    you_velocity = float(you["velocity"])

    direction = float(opponent.get("direction", 0.0))
    distance = float(opponent.get("distance", 0.0))
    rel_velocity = float(opponent.get("velocity", 0.0))

    if "position" in opponent:
        opponent_position = float(opponent["position"])
    else:
        opponent_position = you_position + (distance * direction)

    if "absolute_velocity" in opponent:
        opponent_velocity = float(opponent["absolute_velocity"])
    else:
        # Protocol snapshots expose relative velocity, so reconstruct absolute
        # velocity for the canonical observation builder.
        if direction < 0.0:
            opponent_velocity = you_velocity - rel_velocity
        else:
            opponent_velocity = you_velocity + rel_velocity

    snapshot_recent_damage = snapshot.get("recent_damage_dealt", recent_damage)
    opponent_stance = opponent.get("stance_hint", opponent.get("stance", "neutral"))

    return build_observation(
        you_position=you_position,
        you_velocity=you_velocity,
        you_hp=float(you["hp"]),
        you_max_hp=float(you["max_hp"]),
        you_stamina=float(you["stamina"]),
        you_max_stamina=float(you["max_stamina"]),
        opponent_position=opponent_position,
        opponent_velocity=opponent_velocity,
        opponent_hp=float(opponent["hp"]),
        opponent_max_hp=float(opponent["max_hp"]),
        opponent_stamina=float(opponent["stamina"]),
        opponent_max_stamina=float(opponent["max_stamina"]),
        opponent_stance=opponent_stance,
        arena_width=float(arena["width"]),
        recent_damage=float(snapshot_recent_damage),
    )


def build_observation_batch(
    *,
    you_position,
    you_velocity,
    you_hp,
    you_max_hp,
    you_stamina,
    you_max_stamina,
    opponent_position,
    opponent_velocity,
    opponent_hp,
    opponent_max_hp,
    opponent_stamina,
    opponent_max_stamina,
    opponent_stance,
    arena_width: float,
    recent_damage,
) -> np.ndarray:
    """Build batched 13-dimensional observations with canonical semantics."""
    you_position = _to_float_array(you_position)
    you_velocity = _to_float_array(you_velocity)
    you_hp = _to_float_array(you_hp)
    you_max_hp = _to_float_array(you_max_hp)
    you_stamina = _to_float_array(you_stamina)
    you_max_stamina = _to_float_array(you_max_stamina)
    opponent_position = _to_float_array(opponent_position)
    opponent_velocity = _to_float_array(opponent_velocity)
    opponent_hp = _to_float_array(opponent_hp)
    opponent_max_hp = _to_float_array(opponent_max_hp)
    opponent_stamina = _to_float_array(opponent_stamina)
    opponent_max_stamina = _to_float_array(opponent_max_stamina)
    opponent_stance_int = _to_stance_array(opponent_stance).astype(np.float32)
    recent_damage = _to_float_array(recent_damage)

    hp_norm = you_hp / np.maximum(you_max_hp, 1.0)
    stamina_norm = you_stamina / np.maximum(you_max_stamina, 1.0)
    opp_hp_norm = opponent_hp / np.maximum(opponent_max_hp, 1.0)
    opp_stamina_norm = opponent_stamina / np.maximum(opponent_max_stamina, 1.0)

    distance = np.abs(opponent_position - you_position)
    rel_velocity = _relative_velocity(you_position, you_velocity, opponent_position, opponent_velocity)

    wall_dist_left = you_position
    wall_dist_right = float(arena_width) - you_position

    obs = np.stack(
        [
            you_position,
            you_velocity,
            hp_norm,
            stamina_norm,
            distance,
            rel_velocity,
            opp_hp_norm,
            opp_stamina_norm,
            np.full_like(you_position, float(arena_width), dtype=np.float32),
            wall_dist_left,
            wall_dist_right,
            opponent_stance_int,
            recent_damage,
        ],
        axis=1,
    ).astype(np.float32)

    # Keep observation safety behavior aligned with AtomCombatEnv.
    return np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)


@dataclass(frozen=True)
class RewardStepBatchResult:
    rewards: np.ndarray
    damage_component: np.ndarray
    proximity_component: np.ndarray
    stamina_component: np.ndarray
    stance_component: np.ndarray
    inaction_component: np.ndarray
    terminal_component: np.ndarray
    next_last_distance: np.ndarray


@dataclass(frozen=True)
class RewardStepScalarResult:
    reward: float
    damage_component: float
    proximity_component: float
    stamina_component: float
    stance_component: float
    inaction_component: float
    terminal_component: float
    next_last_distance: float


def compute_step_rewards_batch(
    *,
    dones,
    truncated,
    damage_dealt,
    damage_taken,
    fighter_hp_pct,
    opponent_hp_pct,
    stamina_pct,
    opp_stamina_pct,
    fighter_stance,
    distance,
    last_distance: Optional[np.ndarray],
    tick_counts,
    max_ticks: int,
    arena_width: float,
    episode_damage_dealt,
    episode_stamina_used,
) -> RewardStepBatchResult:
    """
    Canonical batched reward computation shared by single and vmap envs.

    Mirrors AtomCombatEnv's reward semantics, including:
    - terminal/timeout reward structure
    - stance-aware low-stamina penalties
    - proximity/engagement logic ordering
    """
    dones = _to_bool_array(dones)
    truncated = _to_bool_array(truncated)
    damage_dealt = _to_float_array(damage_dealt)
    damage_taken = _to_float_array(damage_taken)
    fighter_hp_pct = _to_float_array(fighter_hp_pct)
    opponent_hp_pct = _to_float_array(opponent_hp_pct)
    stamina_pct = _to_float_array(stamina_pct)
    opp_stamina_pct = _to_float_array(opp_stamina_pct)
    fighter_stance = _to_stance_array(fighter_stance)
    distance = _to_float_array(distance)
    tick_counts = _to_float_array(tick_counts)
    episode_damage_dealt = _to_float_array(episode_damage_dealt)
    episode_stamina_used = _to_float_array(episode_stamina_used)

    n = distance.shape[0]
    rewards = np.zeros(n, dtype=np.float32)
    damage_component = np.zeros(n, dtype=np.float32)
    proximity_component = np.zeros(n, dtype=np.float32)
    stamina_component = np.zeros(n, dtype=np.float32)
    stance_component = np.zeros(n, dtype=np.float32)
    inaction_component = np.zeros(n, dtype=np.float32)
    terminal_component = np.zeros(n, dtype=np.float32)

    terminal_mask = dones
    timeout_mask = truncated & ~dones
    mid_mask = ~(dones | truncated)

    # Terminal (death) rewards.
    if np.any(terminal_mask):
        win_mask = terminal_mask & (fighter_hp_pct > opponent_hp_pct)
        tie_mask = terminal_mask & (fighter_hp_pct == opponent_hp_pct)
        loss_mask = terminal_mask & (fighter_hp_pct < opponent_hp_pct)

        time_bonus = np.maximum(0.0, (float(max_ticks) - tick_counts) / 40.0)
        hp_diff = fighter_hp_pct - opponent_hp_pct
        hp_bonus = hp_diff * 50.0
        damage_per_stamina = episode_damage_dealt / np.maximum(episode_stamina_used, 1.0)
        stamina_efficiency = np.minimum(25.0, damage_per_stamina * 5.0)

        win_reward = 100.0 + time_bonus + hp_bonus + stamina_efficiency
        loss_reward = -100.0 - ((opponent_hp_pct - fighter_hp_pct) * 50.0)

        rewards = np.where(win_mask, win_reward, rewards)
        rewards = np.where(tie_mask, -25.0, rewards)
        rewards = np.where(loss_mask, loss_reward, rewards)
        terminal_component = np.where(terminal_mask, rewards, terminal_component)

    # Timeout rewards.
    if np.any(timeout_mask):
        hp_pct_diff = fighter_hp_pct - opponent_hp_pct

        clear_win_mask = timeout_mask & (hp_pct_diff > 0.1)
        slight_win_mask = timeout_mask & (hp_pct_diff > 0.0) & (hp_pct_diff <= 0.1)
        clear_loss_mask = timeout_mask & (hp_pct_diff < -0.1)
        slight_loss_mask = timeout_mask & (hp_pct_diff < 0.0) & (hp_pct_diff >= -0.1)
        exact_tie_mask = timeout_mask & (hp_pct_diff == 0.0)

        rewards = np.where(clear_win_mask, 100.0 + (hp_pct_diff * 50.0), rewards)
        rewards = np.where(slight_win_mask, 0.0, rewards)
        rewards = np.where(clear_loss_mask, -100.0 + (hp_pct_diff * 50.0), rewards)
        rewards = np.where(slight_loss_mask, -50.0, rewards)
        rewards = np.where(exact_tie_mask, -200.0, rewards)
        terminal_component = np.where(timeout_mask, rewards, terminal_component)

    # Mid-episode shaping rewards.
    if np.any(mid_mask):
        # 1) Damage differential and close-range hit bonus.
        damage_component += np.where(mid_mask, (damage_dealt - damage_taken) * 10.0, 0.0)
        close_range_mask = mid_mask & (damage_dealt > 0.0) & (distance < float(arena_width) * 0.3)
        damage_component += np.where(close_range_mask, damage_dealt * 2.0, 0.0)

        # 2) Stamina-aware shaping.
        stamina_adv_mask = mid_mask & (stamina_pct > opp_stamina_pct + 0.2)
        stamina_component += np.where(stamina_adv_mask, 0.02, 0.0)

        low_stamina_penalty_mask = (
            mid_mask & (stamina_pct < 0.2) & (fighter_stance != STANCE_DEFENDING)
        )
        stamina_component += np.where(low_stamina_penalty_mask, -0.05, 0.0)

        # 3) Proximity shaping.
        if last_distance is not None:
            last_distance_arr = _to_float_array(last_distance)
            distance_delta = last_distance_arr - distance

            pursue_cond = (opponent_hp_pct < 0.3) | (opp_stamina_pct < 0.2)
            recover_cond = (~pursue_cond) & (stamina_pct < 0.2)
            engage_cond = (~pursue_cond) & (~recover_cond) & (distance < float(arena_width) * 0.25)

            closing_mask = mid_mask & pursue_cond & (distance_delta > 0.1)
            backing_mask = mid_mask & recover_cond & (distance_delta < -0.1)
            engage_mask = mid_mask & engage_cond

            proximity_component += np.where(closing_mask, 0.2, 0.0)
            proximity_component += np.where(backing_mask, 0.1, 0.0)
            proximity_component += np.where(
                engage_mask,
                0.1 * (1.0 - distance / (float(arena_width) * 0.25)),
                0.0,
            )

        # 4) Stance-appropriate shaping.
        stance_component += np.where(
            mid_mask & (fighter_stance == STANCE_EXTENDED) & (opponent_hp_pct < 0.5),
            0.05,
            0.0,
        )
        stance_component += np.where(
            mid_mask & (fighter_stance == STANCE_DEFENDING) & (stamina_pct < 0.3),
            0.10,
            0.0,
        )

        # 5) Inaction penalty (distance-aware).
        no_action_mask = mid_mask & (damage_dealt == 0.0) & (damage_taken == 0.0)
        close_inaction_mask = no_action_mask & (distance < float(arena_width) * 0.2)
        medium_inaction_mask = no_action_mask & (distance >= float(arena_width) * 0.2) & (
            distance < float(arena_width) * 0.4
        )
        far_inaction_mask = no_action_mask & (distance >= float(arena_width) * 0.4)

        inaction_component += np.where(close_inaction_mask, -0.1, 0.0)
        inaction_component += np.where(medium_inaction_mask, -0.05, 0.0)
        inaction_component += np.where(far_inaction_mask, -0.02, 0.0)

        mid_total = (
            damage_component
            + proximity_component
            + stamina_component
            + stance_component
            + inaction_component
        )
        rewards = np.where(mid_mask, mid_total, rewards)

    rewards = np.nan_to_num(rewards, nan=0.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)

    return RewardStepBatchResult(
        rewards=rewards,
        damage_component=damage_component.astype(np.float32),
        proximity_component=proximity_component.astype(np.float32),
        stamina_component=stamina_component.astype(np.float32),
        stance_component=stance_component.astype(np.float32),
        inaction_component=inaction_component.astype(np.float32),
        terminal_component=terminal_component.astype(np.float32),
        next_last_distance=distance.astype(np.float32),
    )


def compute_step_reward_scalar(
    *,
    done: bool,
    truncated: bool,
    damage_dealt: float,
    damage_taken: float,
    fighter_hp_pct: float,
    opponent_hp_pct: float,
    stamina_pct: float,
    opp_stamina_pct: float,
    fighter_stance: int | float | str,
    distance: float,
    last_distance: Optional[float],
    tick_count: int,
    max_ticks: int,
    arena_width: float,
    episode_damage_dealt: float,
    episode_stamina_used: float,
) -> RewardStepScalarResult:
    """Scalar convenience wrapper around `compute_step_rewards_batch`."""
    last_distance_array = None if last_distance is None else np.array([last_distance], dtype=np.float32)
    batch_result = compute_step_rewards_batch(
        dones=np.array([done], dtype=bool),
        truncated=np.array([truncated], dtype=bool),
        damage_dealt=np.array([damage_dealt], dtype=np.float32),
        damage_taken=np.array([damage_taken], dtype=np.float32),
        fighter_hp_pct=np.array([fighter_hp_pct], dtype=np.float32),
        opponent_hp_pct=np.array([opponent_hp_pct], dtype=np.float32),
        stamina_pct=np.array([stamina_pct], dtype=np.float32),
        opp_stamina_pct=np.array([opp_stamina_pct], dtype=np.float32),
        fighter_stance=np.array([fighter_stance], dtype=object),
        distance=np.array([distance], dtype=np.float32),
        last_distance=last_distance_array,
        tick_counts=np.array([tick_count], dtype=np.float32),
        max_ticks=max_ticks,
        arena_width=arena_width,
        episode_damage_dealt=np.array([episode_damage_dealt], dtype=np.float32),
        episode_stamina_used=np.array([episode_stamina_used], dtype=np.float32),
    )
    return RewardStepScalarResult(
        reward=float(batch_result.rewards[0]),
        damage_component=float(batch_result.damage_component[0]),
        proximity_component=float(batch_result.proximity_component[0]),
        stamina_component=float(batch_result.stamina_component[0]),
        stance_component=float(batch_result.stance_component[0]),
        inaction_component=float(batch_result.inaction_component[0]),
        terminal_component=float(batch_result.terminal_component[0]),
        next_last_distance=float(batch_result.next_last_distance[0]),
    )
