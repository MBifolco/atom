#!/usr/bin/env python3
"""
Check GPU Setup for JAX

Checks current GPU availability and provides instructions for enabling GPU support.
"""

import subprocess
import sys


def check_gpu_hardware():
    """Check if GPU hardware is present."""
    print("\n" + "="*80)
    print("GPU HARDWARE CHECK")
    print("="*80)

    try:
        result = subprocess.run(["lspci | grep -i vga"], shell=True, capture_output=True, text=True)
        if result.stdout:
            print("\n✅ GPU Found:")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print("\n❌ No GPU detected")
            return False
    except Exception as e:
        print(f"\n❌ Error checking GPU: {e}")
        return False


def check_rocm():
    """Check if ROCm is installed."""
    print("\n" + "="*80)
    print("ROCM CHECK")
    print("="*80)

    try:
        result = subprocess.run(["which", "rocm-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\n✅ ROCm installed: {result.stdout.strip()}")

            # Get ROCm version
            try:
                version_result = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True)
                if version_result.returncode == 0:
                    print(f"   GPU Info:\n{version_result.stdout}")
            except:
                pass

            return True
        else:
            print("\n❌ ROCm not found")
            return False
    except Exception as e:
        print(f"\n❌ Error checking ROCm: {e}")
        return False


def check_jax_gpu():
    """Check if JAX can see GPU."""
    print("\n" + "="*80)
    print("JAX GPU SUPPORT CHECK")
    print("="*80)

    try:
        import jax
        import jaxlib

        print(f"\n✅ JAX installed")
        print(f"   JAX version: {jax.__version__}")
        print(f"   JAXlib version: {jaxlib.__version__}")

        devices = jax.devices()
        print(f"\n   Devices: {devices}")
        print(f"   Default backend: {jax.default_backend()}")

        has_gpu = any('gpu' in str(d).lower() or 'rocm' in str(d).lower() for d in devices)

        if has_gpu:
            print("\n✅ JAX GPU support ENABLED")
            return True
        else:
            print("\n⚠️  JAX GPU support NOT enabled (CPU only)")
            print("\n   Current jaxlib is CPU-only.")
            print("   To enable AMD GPU support, install jax with ROCm:")
            print()
            print("   # Uninstall current jax/jaxlib")
            print("   pip uninstall jax jaxlib -y")
            print()
            print("   # Install JAX with ROCm support")
            print("   pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/jax_releases.html")
            print()
            print("   Note: This requires ROCm 5.x or 6.x")
            return False

    except ImportError as e:
        print(f"\n❌ JAX not installed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error checking JAX: {e}")
        return False


def estimate_gpu_speedup():
    """Estimate potential GPU speedup."""
    print("\n" + "="*80)
    print("POTENTIAL GPU SPEEDUP ESTIMATE")
    print("="*80)

    print("\nBased on JAX benchmarks for similar workloads:")
    print()
    print("  Single Episode (no vectorization):")
    print("    CPU: 10,065 ticks/sec")
    print("    GPU: ~50,000 - 100,000 ticks/sec (estimated)")
    print("    Speedup: ~5-10x")
    print()
    print("  Vectorized (batch=500):")
    print("    CPU: 122,947 ticks/sec")
    print("    GPU: ~1,000,000 - 5,000,000 ticks/sec (estimated)")
    print("    Speedup: ~10-50x")
    print()
    print("  Training (SBX + vmap):")
    print("    CPU: 2,828 steps/sec (Phase 2)")
    print("    GPU: ~10,000 - 50,000 steps/sec (estimated)")
    print("    Speedup: ~5-20x")
    print()
    print("NOTE: These are rough estimates. Actual performance depends on:")
    print("  - GPU model (you have AMD 73ef)")
    print("  - ROCm version")
    print("  - Batch size")
    print("  - Problem complexity")


def main():
    print("\n" + "="*80)
    print("JAX GPU SETUP CHECK")
    print("="*80)

    has_gpu = check_gpu_hardware()
    has_rocm = check_rocm()
    has_jax_gpu = check_jax_gpu()

    estimate_gpu_speedup()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if has_gpu and has_rocm and has_jax_gpu:
        print("\n✅ READY: GPU fully set up and JAX can use it!")
        print("\n   Next step: Run benchmarks with GPU to measure actual speedup")
    elif has_gpu and has_rocm and not has_jax_gpu:
        print("\n⚠️  ALMOST READY: GPU and ROCm installed, but JAX needs ROCm support")
        print("\n   Next step: Install JAX with ROCm support (see instructions above)")
        print("   Warning: This might require some troubleshooting")
    elif has_gpu and not has_rocm:
        print("\n⚠️  GPU detected but ROCm not installed")
        print("\n   AMD GPUs require ROCm for GPU acceleration")
        print("   Install ROCm first: https://rocm.docs.amd.com/")
    elif not has_gpu:
        print("\n❌ No GPU detected - CPU-only training")
    else:
        print("\n❌ GPU setup incomplete")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
