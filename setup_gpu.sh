#!/bin/bash
# Setup ROCm GPU environment variables for JAX
# Source this file before running training: source setup_gpu.sh

# Use atom environment (Python 3.11.10 with JAX ROCm)
export PYENV_VERSION=atom

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

echo "✅ GPU environment configured:"
echo "   Python: $(python --version)"
echo "   Environment: $PYENV_VERSION"
echo "   HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "   ROCM_PATH=$ROCM_PATH"
echo "   LD_LIBRARY_PATH set"
echo ""
echo "🎮 Testing GPU detection..."
python -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
