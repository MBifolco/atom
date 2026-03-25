#!/usr/bin/env python3
"""Summarize structured training observability artifacts for a run directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _fmt_seconds(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    total = float(value)
    if total < 60:
        return f"{total:.1f}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _print_manifest(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "analysis" / "run_manifest.json")
    print("=" * 80)
    print("TRAINING OBSERVABILITY SUMMARY")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    if not manifest:
        print("Run manifest: missing")
        print()
        return

    training = manifest.get("training", {})
    runtime = manifest.get("runtime", {})
    git_info = manifest.get("git", {})
    print(f"Phase: {training.get('phase', 'n/a')}")
    print(f"Seed: {training.get('seed', 'n/a')}")
    print(f"Runtime platform: {runtime.get('platform', 'n/a')}")
    print(f"Python: {runtime.get('python_version', 'n/a')}")
    print(f"Git branch: {git_info.get('branch', 'n/a')}")
    print(f"Git commit: {git_info.get('commit', 'n/a')}")
    print()


def _print_curriculum_summary(run_dir: Path) -> None:
    analysis_dir = run_dir / "curriculum" / "analysis"
    level_summaries = _read_jsonl(analysis_dir / "level_summaries.jsonl")
    holdout_records = _read_jsonl(analysis_dir / "holdout_eval.jsonl")
    failure_records = _read_jsonl(analysis_dir / "failure_events.jsonl")

    print("CURRICULUM")
    print("-" * 80)
    if not level_summaries:
        print("No curriculum observability records found.")
        print()
        return

    total_wall = sum(float(record.get("level_wall_clock_seconds", 0.0)) for record in level_summaries)
    print(f"Levels recorded: {len(level_summaries)}")
    print(f"Observed curriculum wall clock: {_fmt_seconds(total_wall)}")
    print()
    print("Level summaries:")
    for record in level_summaries:
        print(
            "  "
            f"L{int(record.get('level_index', 0)) + 1} {record.get('level_name', 'unknown')}: "
            f"{record.get('end_reason', 'n/a')}, "
            f"episodes={record.get('episodes_attempted', 'n/a')}, "
            f"overall={_fmt_pct(record.get('overall_win_rate'))}, "
            f"recent={_fmt_pct(record.get('recent_win_rate'))}, "
            f"time={_fmt_seconds(record.get('level_wall_clock_seconds'))}"
        )

    if holdout_records:
        latest = holdout_records[-1]
        print()
        print(
            f"Latest holdout: {latest.get('checkpoint_label', 'n/a')} | "
            f"overall={_fmt_pct(latest.get('overall_win_rate'))} "
            f"({latest.get('overall_wins', 0)}/{latest.get('overall_matches', 0)})"
        )
        by_category: dict[str, list[float]] = defaultdict(list)
        for result in latest.get("suite_results", []):
            by_category[result.get("category", "unknown")].append(float(result.get("win_rate", 0.0)))
        for category, values in sorted(by_category.items()):
            mean_win_rate = sum(values) / max(1, len(values))
            print(f"  {category:12s} {_fmt_pct(mean_win_rate)} across {len(values)} opponents")

    if failure_records:
        print()
        print("Failure/recovery events:")
        event_counts = Counter(record.get("event_type", "unknown") for record in failure_records)
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")

    print()


def _print_population_summary(run_dir: Path) -> None:
    analysis_dir = run_dir / "population" / "analysis"
    generation_summaries = _read_jsonl(analysis_dir / "generation_summary.jsonl")
    lineage_events = _read_jsonl(analysis_dir / "lineage_events.jsonl")
    leaderboard_records = _read_jsonl(analysis_dir / "current_leaderboard.jsonl")
    export_failures = _read_jsonl(analysis_dir / "export_failures.jsonl")

    print("POPULATION")
    print("-" * 80)
    if not generation_summaries:
        print("No population observability records found.")
        print()
        return

    total_training = sum(float(record.get("training_wall_clock_seconds", 0.0)) for record in generation_summaries)
    total_eval = sum(float(record.get("evaluation_wall_clock_seconds", 0.0)) for record in generation_summaries)
    total_saving = sum(float(record.get("saving_wall_clock_seconds", 0.0)) for record in generation_summaries)
    latest_generation = generation_summaries[-1]
    print(f"Generations recorded: {len(generation_summaries)}")
    print(
        f"Total observed time: training={_fmt_seconds(total_training)}, "
        f"evaluation={_fmt_seconds(total_eval)}, saving={_fmt_seconds(total_saving)}"
    )
    print(
        f"Latest generation: G{latest_generation.get('generation', 'n/a')} | "
        f"champion={latest_generation.get('champion_after_evaluation', 'n/a')} | "
        f"mean_reward={latest_generation.get('mean_reward_overall', 'n/a')}"
    )

    if lineage_events:
        print()
        print(f"Lineage events recorded: {len(lineage_events)}")
        recent_events = lineage_events[-min(5, len(lineage_events)):]
        for event in recent_events:
            print(
                "  "
                f"G{event.get('generation', 'n/a')}: {event.get('child_name', 'n/a')} "
                f"from {event.get('parent_name', 'n/a')} replacing {event.get('replaced_fighter_name', 'n/a')}"
            )

    if leaderboard_records:
        latest_generation_id = max(int(record.get("generation", 0)) for record in leaderboard_records)
        latest_board = [
            record for record in leaderboard_records
            if int(record.get("generation", 0)) == latest_generation_id
        ]
        latest_board.sort(key=lambda record: int(record.get("active_generation_rank", 999999)))
        print()
        print(f"Latest active leaderboard snapshot: G{latest_generation_id}")
        for record in latest_board[:8]:
            marker = "*" if record.get("status_in_generation") == "new_child" else " "
            print(
                f" {marker} #{record.get('active_generation_rank', 'n/a'):<2} "
                f"{record.get('fighter_name', 'n/a'):<24} "
                f"ELO={float(record.get('active_generation_elo', 0.0)):.0f} "
                f"gen={record.get('fighter_generation', 'n/a')} "
                f"status={record.get('status_in_generation', 'n/a')}"
            )

    if export_failures:
        print()
        print(f"Export failures recorded: {len(export_failures)}")
        failure_counts = Counter(record.get("fighter_name", "unknown") for record in export_failures)
        for fighter_name, count in sorted(failure_counts.items()):
            print(f"  {fighter_name}: {count}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run directory, e.g. /content/drive/MyDrive/atom_runs/run1")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        parser.error(f"Run directory not found: {run_dir}")

    _print_manifest(run_dir)
    _print_curriculum_summary(run_dir)
    _print_population_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
