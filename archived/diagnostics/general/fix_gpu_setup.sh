#!/bin/bash
# Fix GPU setup for JAX/ROCm on AMD Radeon RX 6650 XT

echo "Setting up GPU environment for AMD Radeon RX 6650 XT..."

# Set GPU architecture for ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RDNA2 architecture

# Set ROCm paths
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH/hip
export HCC_HOME=$ROCM_PATH/hcc
export ROCM_HOME=$ROCM_PATH

# Set library paths
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/rocprofiler/bin:$ROCM_PATH/opencl/bin:$PATH

# JAX GPU configuration
export JAX_PLATFORMS=rocm  # Force ROCm platform
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"  # Disable triton for stability
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Don't preallocate GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Use up to 75% of GPU memory

# ROCm/HIP memory management
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0

# Disable GPU memory growth to prevent fragmentation
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Force CPU fallback if GPU fails
export JAX_PLATFORM_NAME=rocm

echo "Environment variables set!"
echo ""
echo "GPU Architecture Override: HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "JAX Platform: JAX_PLATFORMS=$JAX_PLATFORMS"
echo "Memory Fraction: XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo ""
echo "To apply these settings, run:"
echo "  source archived/diagnostics/general/fix_gpu_setup.sh"
