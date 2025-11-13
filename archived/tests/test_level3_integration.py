#!/usr/bin/env python3
"""Test Level 3 (vmap) Integration with Trainer"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.trainers.ppo.trainer import train_fighter

print("Testing Level 3 (vmap) Integration...")
print("=" * 60)

# Quick test with small timesteps
train_fighter(
    opponent_files=["fighters/training_opponents/training_dummy.py"],
    output_path="outputs/test_vmap/model.zip",
    episodes=20,  # Just 20 episodes for quick test
    n_envs=25,  # Use 25 parallel vmap environments (less than 14 cores)
    max_ticks=250,
    verbose=True,
    use_vmap=True,  # Enable Level 3!
    checkpoint_freq=50000  # Don't save checkpoints
)

print("\n" + "=" * 60)
print("✅ Level 3 Integration Test Complete!")
print("=" * 60)
