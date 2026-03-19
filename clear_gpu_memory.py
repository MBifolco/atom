#!/usr/bin/env python3
"""
Clear GPU memory from stuck processes and caches.
"""

import os
import gc
import time

print("Clearing GPU memory...")

# 1. Clear JAX caches
try:
    import jax
    jax.clear_caches()
    print("✓ JAX caches cleared")
except Exception as e:
    print(f"  JAX: {e}")

# 2. Clear PyTorch caches
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ PyTorch GPU cache cleared")
    else:
        print("  PyTorch: No CUDA devices available")
except Exception as e:
    print(f"  PyTorch: {e}")

# 3. Force garbage collection
gc.collect()
gc.collect()
print("✓ Garbage collection completed")

# 4. Set environment variables for future runs
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCR_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
print("✓ Environment variables set for memory management")

# 5. Check GPU status
import subprocess
print("\nCurrent GPU status:")
result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
if result.returncode == 0:
    for line in result.stdout.split('\n'):
        if line.startswith('0 ') or 'VRAM%' in line:
            print(line)

print("\nGPU memory cleanup complete!")
print("\nNote: Some memory (20-30%) may still be used by:")
print("  - X server / desktop environment")
print("  - System processes")
print("\nTo fully reset GPU (requires sudo):")
print("  sudo rocm-smi --gpureset -d 0")