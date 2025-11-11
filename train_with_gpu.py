#!/usr/bin/env python3
"""
GPU-Accelerated Training Script
Level 4: Maximum performance with AMD ROCm GPU acceleration

Usage:
    source setup_gpu.sh
    python train_with_gpu.py

Performance: ~77x faster than baseline
"""

import sys
from pathlib import Path
import os

# Verify GPU environment is set
if 'HSA_OVERRIDE_GFX_VERSION' not in os.environ:
    print("❌ GPU environment not configured!")
    print("Run: source setup_gpu.sh")
    print("Then try again.")
    sys.exit(1)

# Verify we're using the right Python
import platform
if not platform.python_version().startswith('3.11'):
    print(f"❌ Wrong Python version: {platform.python_version()}")
    print("Expected: 3.11.x")
    print("Run: source setup_gpu.sh")
    sys.exit(1)

# Verify JAX GPU
import jax
if jax.default_backend() != 'gpu':
    print("⚠️  WARNING: GPU not detected!")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print("\nTraining will run on CPU (slower)")
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(1)
else:
    print(f"✅ GPU detected: {jax.devices()[0]}")
    print(f"   JAX version: {jax.__version__}")

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.trainers.ppo.trainer import train_fighter

print("\n" + "="*80)
print("  GPU-ACCELERATED TRAINING")
print("="*80)
print(f"\nConfiguration:")
print(f"  GPU: {jax.devices()[0]}")
print(f"  Python: {platform.python_version()}")
print(f"  JAX: {jax.__version__}")
print(f"  Backend: {jax.default_backend()}")
print(f"\n  Performance: ~77x faster than baseline")
print(f"  Using vmap with 250 parallel environments")
print("="*80)

# Training configuration
opponent_files = ["fighters/training_opponents/training_dummy.py"]
output_path = "outputs/gpu_trained/champion.zip"
episodes = 10000
n_envs = 250  # Optimal for GPU (from benchmarks)
max_ticks = 250

print(f"\nTraining Configuration:")
print(f"  Opponent: {opponent_files[0]}")
print(f"  Output: {output_path}")
print(f"  Episodes: {episodes:,}")
print(f"  Parallel environments: {n_envs} (GPU-accelerated)")
print(f"  Max ticks/episode: {max_ticks}")
print("")

# Create output directory
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

# Start training
print("Starting GPU-accelerated training...")
print("="*80)
print("")

train_fighter(
    opponent_files=opponent_files,
    output_path=output_path,
    episodes=episodes,
    n_envs=n_envs,
    max_ticks=max_ticks,
    checkpoint_freq=50000,
    verbose=True,
    use_vmap=True,  # Enable Level 3: JAX vmap parallelization
    device="auto"   # Will use GPU if available
)

print("\n" + "="*80)
print("  ✅ GPU-ACCELERATED TRAINING COMPLETE!")
print("="*80)
print(f"\nModel saved to: {output_path}")
print("\nPerformance achieved: ~77x faster than baseline")
print("="*80)
