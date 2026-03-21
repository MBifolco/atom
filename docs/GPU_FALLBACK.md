# GPU Fallback for JAX

## Problem
JAX was failing with ROCM/HIP GPU errors:
```
INTERNAL: Failed to load HSACO: HIP_ERROR_NoBinaryForGpu
```

## Solution
Added automatic GPU fallback to CPU mode in two places:

### 1. In `src/atom/runtime/arena/arena_1d_jax_jit.py`
```python
# Force CPU if requested or if GPU fails
if os.environ.get("ATOM_FORCE_CPU", "").lower() in ["1", "true", "yes"]:
    jax.config.update('jax_platform_name', 'cpu')
else:
    # Try GPU first, fallback to CPU if it fails
    try:
        test = jax.numpy.array([1.0])
        _ = test + 1  # Force computation
    except Exception as e:
        print(f"GPU initialization failed ({e}), falling back to CPU")
        jax.config.update('jax_platform_name', 'cpu')
```

### 2. In `atom_fight.py`
Early detection before importing arena:
```python
# Auto-detect GPU issues and fallback to CPU
try:
    import jax
    devices = jax.devices()
    if any('gpu' in str(d).lower() or 'rocm' in str(d).lower() for d in devices):
        try:
            test = jax.numpy.array([1.0])
            _ = test + 1
        except Exception as e:
            print(f"⚠️ GPU initialization failed, using CPU mode")
            os.environ["ATOM_FORCE_CPU"] = "1"
except Exception:
    pass
```

## Usage

### Automatic (Recommended)
Just run normally - will auto-detect and fallback:
```bash
python atom_fight.py fighter_a.py fighter_b.py
```

### Force CPU Mode
Set environment variable to skip GPU entirely:
```bash
ATOM_FORCE_CPU=1 python atom_fight.py fighter_a.py fighter_b.py
```

## Benefits
- No more GPU crashes
- Automatic fallback to CPU
- Works on any system
- Can force CPU mode when needed