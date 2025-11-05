#!/usr/bin/env python3
"""
Fighter Registry Builder

Scans the fighters/ directory and builds registry.json with metadata
for all discovered fighters.

Usage:
    python build_registry.py
    python build_registry.py --output custom/path/registry.json
"""

import argparse
from pathlib import Path
from src.registry import FighterRegistry


def main():
    parser = argparse.ArgumentParser(
        description="Build fighter registry from fighters/ directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for registry.json (default: fighters/registry.json)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all fighters after building registry"
    )

    args = parser.parse_args()

    # Determine paths
    project_root = Path(__file__).parent
    registry_path = Path(args.output) if args.output else project_root / "fighters" / "registry.json"

    print("="*60)
    print("FIGHTER REGISTRY BUILDER")
    print("="*60)

    # Create registry
    registry = FighterRegistry(registry_path)

    # Scan directories
    fighters_dir = project_root / "fighters"

    print(f"\nScanning {fighters_dir}...")

    # Scan examples directory
    examples_dir = fighters_dir / "examples"
    if examples_dir.exists():
        print(f"\n📁 Scanning {examples_dir}...")
        count = registry.scan_directory(examples_dir, fighter_type="rule-based")
        print(f"   Found {count} rule-based fighters")

    # Scan AIs directory
    ais_dir = fighters_dir / "AIs"
    if ais_dir.exists():
        print(f"\n📁 Scanning {ais_dir}...")
        count = registry.scan_directory(ais_dir, fighter_type="onnx-ai")
        print(f"   Found {count} AI fighters")

    # Scan training_opponents directory
    training_dir = fighters_dir / "training_opponents"
    if training_dir.exists():
        print(f"\n📁 Scanning {training_dir}...")
        count = registry.scan_directory(training_dir, fighter_type="training")
        print(f"   Found {count} training opponents")

    # Save registry
    print(f"\n💾 Saving registry to {registry_path}...")
    registry.save()

    # Print summary
    all_fighters = registry.list_fighters()
    print(f"\n✅ Registry built successfully!")
    print(f"   Total fighters: {len(all_fighters)}")

    # Group by type
    by_type = {}
    for fighter in all_fighters:
        by_type.setdefault(fighter.type, []).append(fighter)

    for fighter_type, fighters in sorted(by_type.items()):
        print(f"   - {fighter_type}: {len(fighters)}")

    # Validate if requested
    if args.validate:
        print("\n🔍 Validating all fighters...")
        results = registry.validate_all()
        valid_count = sum(1 for valid in results.values() if valid)
        invalid_count = len(results) - valid_count

        print(f"   ✅ Valid: {valid_count}")
        if invalid_count > 0:
            print(f"   ❌ Invalid: {invalid_count}")
            for fighter_id, valid in results.items():
                if not valid:
                    print(f"      - {fighter_id}")
        else:
            print("   All fighters are valid!")

    print("\n" + "="*60)
    print(f"Registry saved to: {registry_path}")
    print("="*60)


if __name__ == "__main__":
    main()
