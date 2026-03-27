#!/usr/bin/env python3
"""
Diagnostic script: manually test a curriculum graduate model against holdout opponents.

Answers the question: can the trained policy actually fight, or is the holdout
evaluator broken?

Usage:
    python scripts/analysis/test_curriculum_graduate.py <model_path>
    python scripts/analysis/test_curriculum_graduate.py training_outputs/run2/curriculum/models/curriculum_graduate.zip

Prints per-tick state (position, velocity, stance, actions) for the first few
matches so you can visually confirm whether the fighter moves and attacks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_model(model_path: str):
    """Load an SB3 PPO model from disk."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    print(f"Loaded model from {model_path}")
    print(f"  Policy class: {model.policy.__class__.__name__}")
    print(f"  Observation space: {model.observation_space}")
    print(f"  Action space: {model.action_space}")
    return model


def run_match(model, opponent_path: str, seed: int = 42, verbose: bool = True, deterministic: bool = False) -> dict:
    """Run a single match and return results with per-tick trace."""
    from src.atom.training.gym_env import AtomCombatEnv

    # Load opponent
    import importlib.util

    spec = importlib.util.spec_from_file_location("opponent", opponent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opponent_func = module.decide

    env = AtomCombatEnv(
        opponent_decision_func=opponent_func,
        max_ticks=250,
        fighter_mass=70.0,
        opponent_mass=70.0,
        seed=seed,
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    ticks = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        from src.atom.training.action_codec import extract_stance
        raw_accel = float(action[0])
        raw_stance = extract_stance(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

        tick_data = {
            "tick": info["tick"],
            "fighter_pos": float(env.fighter.position),
            "fighter_vel": float(env.fighter.velocity),
            "fighter_hp": float(env.fighter.hp),
            "fighter_stamina": float(env.fighter.stamina),
            "fighter_stance": int(env.fighter.stance),
            "opponent_pos": float(env.opponent.position),
            "opponent_hp": float(env.opponent.hp),
            "action_accel": raw_accel,
            "action_stance": raw_stance,
            "damage_dealt": info["damage_dealt"],
            "damage_taken": info["damage_taken"],
            "reward": float(reward),
        }
        ticks.append(tick_data)

    fighter_hp = float(info.get("fighter_hp", 0.0))
    opponent_hp = float(info.get("opponent_hp", 0.0))
    won = info.get("won")
    if won is None:
        won = fighter_hp > opponent_hp

    result = {
        "won": bool(won),
        "total_reward": total_reward,
        "fight_length": len(ticks),
        "episode_damage_dealt": float(info.get("episode_damage_dealt", 0.0)),
        "episode_damage_taken": float(info.get("episode_damage_taken", 0.0)),
        "final_fighter_hp": fighter_hp,
        "final_opponent_hp": opponent_hp,
        "ticks": ticks,
    }

    if verbose:
        print_match_trace(result, opponent_path)

    env.close()
    return result


STANCE_NAMES = {0: "neutral", 1: "extended", 2: "defending"}


def print_match_trace(result: dict, opponent_path: str) -> None:
    """Print a readable match trace."""
    opp_name = Path(opponent_path).stem
    ticks = result["ticks"]

    print(f"\n{'='*80}")
    print(f"  vs {opp_name}  |  {'WIN' if result['won'] else 'LOSS'}  |  "
          f"{result['fight_length']} ticks  |  "
          f"dmg dealt: {result['episode_damage_dealt']:.1f}  |  "
          f"dmg taken: {result['episode_damage_taken']:.1f}  |  "
          f"reward: {result['total_reward']:.1f}")
    print(f"  final HP: fighter={result['final_fighter_hp']:.1f}  opponent={result['final_opponent_hp']:.1f}")
    print(f"{'='*80}")

    # Print header
    print(f"{'tick':>4}  {'f_pos':>6}  {'f_vel':>6}  {'stance':>8}  "
          f"{'accel':>6}  {'st_sel':>6}  {'o_pos':>6}  {'dist':>6}  "
          f"{'dmg_dlt':>7}  {'dmg_tkn':>7}  {'f_hp':>6}  {'o_hp':>6}")
    print("-" * 100)

    # Print first 20 ticks, then every 25th, then last 5
    indices = set(range(min(20, len(ticks))))
    indices.update(range(0, len(ticks), 25))
    indices.update(range(max(0, len(ticks) - 5), len(ticks)))

    for i in sorted(indices):
        t = ticks[i]
        dist = abs(t["fighter_pos"] - t["opponent_pos"])
        stance_name = STANCE_NAMES.get(t["fighter_stance"], f"?{t['fighter_stance']}")
        print(
            f"{t['tick']:>4}  "
            f"{t['fighter_pos']:>6.2f}  "
            f"{t['fighter_vel']:>6.2f}  "
            f"{stance_name:>8}  "
            f"{t['action_accel']:>6.3f}  "
            f"{t['action_stance']:>6}  "
            f"{t['opponent_pos']:>6.2f}  "
            f"{dist:>6.2f}  "
            f"{t['damage_dealt']:>7.1f}  "
            f"{t['damage_taken']:>7.1f}  "
            f"{t['fighter_hp']:>6.1f}  "
            f"{t['opponent_hp']:>6.1f}"
        )


def summarize_action_distribution(all_ticks: list[dict]) -> None:
    """Print action distribution summary across all matches."""
    if not all_ticks:
        return

    accels = [t["action_accel"] for t in all_ticks]
    stances = [int(t["action_stance"]) for t in all_ticks]
    velocities = [t["fighter_vel"] for t in all_ticks]

    print(f"\n{'='*80}")
    print("ACTION DISTRIBUTION SUMMARY (all matches)")
    print(f"{'='*80}")
    print(f"  Acceleration: mean={np.mean(accels):.4f}  std={np.std(accels):.4f}  "
          f"min={np.min(accels):.4f}  max={np.max(accels):.4f}")
    print(f"  Velocity:     mean={np.mean(velocities):.4f}  std={np.std(velocities):.4f}  "
          f"min={np.min(velocities):.4f}  max={np.max(velocities):.4f}")

    stance_counts = {0: 0, 1: 0, 2: 0}
    for s in stances:
        stance_counts[min(s, 2)] = stance_counts.get(min(s, 2), 0) + 1
    total = len(stances)
    print(f"  Stance dist:  neutral={stance_counts[0]/total:.1%}  "
          f"extended={stance_counts[1]/total:.1%}  "
          f"defending={stance_counts[2]/total:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Test curriculum graduate against holdout opponents")
    parser.add_argument("model_path", help="Path to curriculum_graduate.zip")
    parser.add_argument("--seed", type=int, default=42, help="Match seed (default: 42)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run with multiple seeds")
    parser.add_argument("--opponent", type=str, help="Single opponent path to test against")
    parser.add_argument("--quiet", action="store_true", help="Only show summary, no per-tick trace")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (not recommended — causes stance collapse)")
    args = parser.parse_args()

    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    model = load_model(model_path)
    mode = "DETERMINISTIC" if args.deterministic else "STOCHASTIC"
    print(f"  Mode: {mode}")

    # Default holdout suite (same as curriculum holdout evaluator)
    if args.opponent:
        opponents = [{"label": Path(args.opponent).stem, "path": args.opponent}]
    else:
        opponents = [
            {"label": "stationary_neutral", "path": "fighters/test_dummies/atomic/stationary_neutral.py"},
            {"label": "stationary_extended", "path": "fighters/test_dummies/atomic/stationary_extended.py"},
            {"label": "approach_slow", "path": "fighters/test_dummies/atomic/approach_slow.py"},
            {"label": "charge_on_approach", "path": "fighters/test_dummies/atomic/charge_on_approach.py"},
            {"label": "aggressive_stance_switcher", "path": "fighters/test_dummies/atomic/aggressive_stance_switcher.py"},
            {"label": "boxer", "path": "fighters/examples/boxer.py"},
            {"label": "slugger", "path": "fighters/examples/slugger.py"},
        ]

    seeds = args.seeds or [args.seed]
    all_ticks = []
    results_summary = []

    for seed in seeds:
        if len(seeds) > 1:
            print(f"\n\n{'#'*80}")
            print(f"  SEED: {seed}")
            print(f"{'#'*80}")

        for opp in opponents:
            if not Path(opp["path"]).exists():
                print(f"  SKIP: {opp['path']} not found")
                continue
            result = run_match(model, opp["path"], seed=seed, verbose=not args.quiet, deterministic=args.deterministic)
            all_ticks.extend(result["ticks"])
            results_summary.append({
                "seed": seed,
                "opponent": opp["label"],
                "won": result["won"],
                "dmg_dealt": result["episode_damage_dealt"],
                "dmg_taken": result["episode_damage_taken"],
                "length": result["fight_length"],
                "reward": result["total_reward"],
            })

    # Print summary table
    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'seed':>6}  {'opponent':<30}  {'result':>6}  {'dmg_dlt':>7}  "
          f"{'dmg_tkn':>7}  {'length':>6}  {'reward':>8}")
    print("-" * 90)
    for r in results_summary:
        print(f"{r['seed']:>6}  {r['opponent']:<30}  "
              f"{'WIN' if r['won'] else 'LOSS':>6}  "
              f"{r['dmg_dealt']:>7.1f}  {r['dmg_taken']:>7.1f}  "
              f"{r['length']:>6}  {r['reward']:>8.1f}")

    wins = sum(1 for r in results_summary if r["won"])
    total = len(results_summary)
    print(f"\nOverall: {wins}/{total} wins ({wins/max(1,total):.0%})")

    summarize_action_distribution(all_ticks)


if __name__ == "__main__":
    main()
