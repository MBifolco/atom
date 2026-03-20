#!/usr/bin/env python3
"""
Test GPU memory management improvements for population training.
"""

import os
import subprocess

def check_gpu_memory():
    """Check current GPU memory usage."""
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Current GPU status:")
            print(result.stdout)
    except:
        print("Unable to check GPU status")

def test_population_training():
    """Test population training with new GPU memory management."""
    print("=" * 80)
    print("Testing GPU Memory Management for Population Training")
    print("=" * 80)

    # Check initial GPU memory
    print("\n1. Initial GPU memory:")
    check_gpu_memory()

    # Test with sequential GPU training (default)
    print("\n2. Testing sequential GPU training (1 fighter at a time)...")
    cmd = [
        "python3", "train_progressive.py",
        "--mode", "population",
        "--use-vmap",
        "--generations", "1",
        "--population", "4",
        "--episodes-per-gen", "100"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if "out of memory" in result.stderr.lower() or "hip error" in result.stderr.lower():
        print("❌ GPU OOM error detected!")
        print("\n3. Trying CPU-only mode for population...")

        # Test with CPU-only mode
        cmd.append("--population-cpu-only")
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ CPU-only mode worked!")
        else:
            print("❌ CPU-only mode also failed")
    else:
        print("✅ Sequential GPU training worked!")

    # Check final GPU memory
    print("\n4. Final GPU memory:")
    check_gpu_memory()

if __name__ == "__main__":
    test_population_training()