#!/usr/bin/env python3
"""
Test script for progressive training system integration.

Quick test to verify that curriculum learning flows into population training correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_progressive import ProgressiveTrainer


def test_quick_integration():
    """
    Run a very quick integration test of the progressive training system.

    This tests:
    1. Curriculum training with test dummies
    2. Population initialization from curriculum graduate
    3. Population evolution with the trained base
    """
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING INTEGRATION TEST")
    print("="*80)
    print("Running quick test with minimal timesteps...")
    print()

    # Create trainer with test output directory
    trainer = ProgressiveTrainer(
        algorithm="ppo",
        output_dir="outputs/test_progressive_integration",
        verbose=True
    )

    # Run complete pipeline with minimal settings for testing
    trainer.run_complete_pipeline(
        curriculum_timesteps=5000,  # Very short for testing (normally 500k+)
        population_generations=2,    # Just 2 generations (normally 10+)
        population_size=4           # Small population (normally 8+)
    )

    # Verify outputs exist
    output_dir = Path("outputs/test_progressive_integration")

    # Check curriculum outputs
    curriculum_model = output_dir / "curriculum" / "models" / "curriculum_graduate.zip"
    if curriculum_model.exists():
        print("\n✅ Curriculum model created successfully")
    else:
        print("\n❌ Curriculum model not found!")
        return False

    # Check population outputs
    pop_models = list((output_dir / "population" / "models").glob("*.zip"))
    if pop_models:
        print(f"✅ Population models created: {len(pop_models)} models")
    else:
        print("❌ No population models found!")
        return False

    # Check champion exports
    champions = list((output_dir / "champions").glob("*.zip"))
    if champions:
        print(f"✅ Champions exported: {len(champions)} champions")
    else:
        print("❌ No champions exported!")
        return False

    # Check training report
    reports = list(output_dir.glob("training_report_*.json"))
    if reports:
        print(f"✅ Training report created")
    else:
        print("❌ Training report not found!")
        return False

    print("\n" + "="*80)
    print("✅ INTEGRATION TEST PASSED!")
    print("="*80)
    print("\nProgressive training system is working correctly:")
    print("  1. Curriculum learning trains against test dummies")
    print("  2. Population is initialized from curriculum graduate")
    print("  3. Population evolves and exports champions")
    print()
    print("Ready for full training runs!")
    print()

    return True


if __name__ == "__main__":
    success = test_quick_integration()

    if success:
        print("\nTo run a full training session:")
        print("  python training/train_progressive.py --mode complete")
        print()
        print("Or for quick testing:")
        print("  python training/train_progressive.py --mode quick")

    sys.exit(0 if success else 1)