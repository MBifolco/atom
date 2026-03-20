#!/usr/bin/env python3
"""
Validate relative links in markdown files.

Usage:
    python scripts/ops/check_markdown_links.py
    python scripts/ops/check_markdown_links.py --root docs --root fighters
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _normalize_link_target(raw: str) -> str:
    """Strip optional markdown link title while preserving spaces in paths."""
    target = raw.strip()

    # Ignore angle-bracket wrapped links: <path>
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()

    # Strip optional trailing title: path "title"
    m = re.match(r'^(.*?)(?:\s+"[^"]*")?$', target)
    if m:
        target = m.group(1).strip()
    return target


def _iter_markdown_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.md"):
        parts = set(path.parts)
        if ".git" in parts:
            continue
        if "outputs" in parts or "training_outputs" in parts or "htmlcov" in parts:
            continue
        files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Check markdown relative links")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory to scan (default: repository root)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    roots = [repo_root / r for r in args.root] if args.root else [repo_root]

    markdown_files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[warn] root does not exist: {root}")
            continue
        markdown_files.extend(_iter_markdown_files(root))

    broken: list[tuple[Path, str]] = []
    for md_path in markdown_files:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for raw_target in LINK_RE.findall(text):
            target = _normalize_link_target(raw_target)
            if not target:
                continue
            if (
                target.startswith("http://")
                or target.startswith("https://")
                or target.startswith("mailto:")
                or target.startswith("#")
            ):
                continue

            resolved = (md_path.parent / target).resolve()
            if not resolved.exists():
                broken.append((md_path.relative_to(repo_root), target))

    if broken:
        print(f"[fail] broken markdown links: {len(broken)}")
        for md, target in broken:
            print(f"  {md} -> {target}")
        return 1

    print(f"[ok] markdown links valid across {len(markdown_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
