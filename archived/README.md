# Archived Files

This directory contains scripts and files that were used during development and testing but are no longer needed for regular use.

## Directory Structure

### `benchmarks/`
Performance benchmarks used during JAX optimization development:
- `benchmark_end_to_end.py` - End-to-end training benchmarks (CPU vs GPU)
- `benchmark_gpu.py` - GPU performance benchmarks (ROCm testing)
- `benchmark_jax_physics.py` - JAX physics engine benchmarks (Phase 1)
- `benchmark_jax_vmap.py` - JAX vmap parallelization benchmarks (Phase 3)
- `benchmark_multi_env.py` - Multi-environment scaling benchmarks (Level 2)
- `benchmark_sbx_training.py` - SBX vs SB3 training benchmarks (Phase 2)
- `demo_jax_scaling.py` - Complete demo showing all optimization levels

**Purpose**: These were used to measure performance improvements and validate optimization levels.

**Status**: Results documented in `docs/FINAL_RESULTS_ALL_LEVELS.md`

### `tests/`
One-time integration tests for JAX optimization:
- `test_gym_jax_jit.py` - Gym environment JAX JIT integration tests
- `test_jax_jit.py` - JAX JIT compilation correctness tests
- `test_level3_integration.py` - Level 3 vmap integration tests
- `test_vmap_wrapper.py` - VmapEnvWrapper unit tests

**Purpose**: Validated correctness of JAX implementations during development.

**Status**: All tests passed, functionality integrated into main codebase.

### `setup_scripts/`
One-time setup and verification scripts:
- `check_gpu_setup.py` - GPU detection and capability checker
- `upgrade_rocm_to_7.sh` - ROCm 6.3 → 7.1 upgrade script

**Purpose**: Used during initial GPU setup and ROCm upgrade.

**Status**: Setup complete, GPU working. Scripts kept for reference.

## Why Archived?

These files served their purpose during development and testing:
- **Benchmarks**: Measured performance, results documented
- **Tests**: Validated correctness, implementations working
- **Setup scripts**: One-time configuration, now complete

They're kept for:
- Historical reference
- Re-running benchmarks if needed
- Understanding optimization development process
- Troubleshooting if issues arise

## Active Files (in root)

For regular use, see these files in the project root:
- `atom_fight.py` - Main battle/testing script
- `train_progressive.py` - Standard training script
- `train_with_gpu.py` - GPU-accelerated training script
- `setup_gpu.sh` - GPU environment setup (source before GPU training)
- `build_registry.py` - Build fighter registry

## Documentation

Complete documentation available in `docs/`:
- `docs/FINAL_RESULTS_ALL_LEVELS.md` - Complete optimization results
- `docs/QUICK_REFERENCE.md` - Quick start guide
- `docs/GPU_SETUP_GUIDE.md` - GPU setup instructions
- `docs/JAX_OPTIMIZATION_ROADMAP.md` - Full optimization guide

---

**Last Updated**: 2025-11-11
**Status**: Development complete, all optimization levels working
