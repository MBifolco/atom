# GPU Setup Guide: JAX + ROCm on AMD GPUs

**Goal**: Enable JAX GPU acceleration on AMD Radeon GPUs using ROCm

**Your Hardware**: AMD GPU gfx1032 (Radeon RX 6000 series)

**Status**: CPU-only JAX currently installed

**Potential Speedup**: 10-100x for vectorized operations

---

## Prerequisites Check

### 1. Verify GPU Hardware

```bash
# Check GPU device
lspci | grep -i vga

# Expected output:
# 09:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] 73ef (Radeon RX 6000 series)
```

### 2. Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi --showproductname

# Or check ROCm directory
ls /opt/rocm

# Expected: ROCm 5.x or 6.x directory structure
```

### 3. Check Current JAX Status

```bash
python -c "import jax; print('Devices:', jax.devices()); print('Default backend:', jax.default_backend())"

# Current output:
# Devices: [CpuDevice(id=0)]
# Default backend: cpu
```

---

## Installation Options

### Option A: Pre-built ROCm Wheels (Recommended First Try)

**Effort**: 15-30 minutes
**Success Rate**: 30-50% (depends on ROCm version compatibility)

**Steps**:

1. **Uninstall CPU-only JAX**:
```bash
pip uninstall jax jaxlib -y
```

2. **Check ROCm Version**:
```bash
# Find your ROCm version
ls /opt/rocm
# Or
cat /opt/rocm/.info/version

# JAX supports ROCm 5.3, 5.4, 5.7, 6.0, 6.1
```

3. **Install JAX with ROCm Support**:
```bash
# For ROCm 6.1 (adjust version as needed)
pip install --upgrade jax[rocm61] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# For ROCm 6.0
pip install --upgrade jax[rocm60] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# For ROCm 5.7
pip install --upgrade jax[rocm57] -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```

4. **Set Environment Variables**:
```bash
# Add to ~/.bashrc or set before running Python
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For gfx1032
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

5. **Verify GPU Detection**:
```bash
python -c "import jax; print('Devices:', jax.devices())"

# Success output:
# Devices: [RocmDevice(id=0)]

# Failure output:
# Devices: [CpuDevice(id=0)]
```

**Common Issues**:
- **No matching distribution found**: Your ROCm version isn't supported by pre-built wheels
- **ImportError: libamdhip64.so**: ROCm libraries not in path, set `LD_LIBRARY_PATH`
- **Unsupported GPU architecture**: gfx1032 may need `HSA_OVERRIDE_GFX_VERSION=10.3.0`

---

### Option B: Build from Source (Advanced)

**Effort**: 4-8 hours
**Success Rate**: 60-70% (if you have build experience)
**Requirements**:
- 20+ GB free disk space
- Bazel build system
- ROCm development tools

**Steps**:

1. **Install Build Dependencies**:
```bash
# Install Bazel
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel

# Install ROCm development tools
sudo apt install rocm-dev rocm-libs
```

2. **Clone JAX Repository**:
```bash
cd ~/projects
git clone https://github.com/google/jax.git
cd jax
```

3. **Configure Build**:
```bash
# Set environment variables
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Configure
python build/build.py --enable_rocm --rocm_path=$ROCM_PATH
```

4. **Build JAXlib** (This takes 2-4 hours):
```bash
# Build with Bazel
bazel build --config=rocm //jaxlib:jaxlib_wheel

# Or use the build script
python build/build.py --enable_rocm --bazel_options=--override_repository=xla=/path/to/xla
```

5. **Install Built Wheel**:
```bash
pip install dist/jaxlib-*.whl
pip install -e .  # Install JAX from source
```

6. **Verify**:
```bash
python -c "import jax; print(jax.devices())"
```

**Common Build Issues**:
- **Bazel OOM**: Add `--local_ram_resources=4096` to limit memory usage
- **XLA compatibility**: Ensure XLA version matches JAX version
- **Compiler errors**: May need specific GCC/Clang versions for ROCm

---

### Option C: Use Docker (Safest)

**Effort**: 1-2 hours
**Success Rate**: 80-90%
**Note**: Isolates JAX GPU setup from system

**Steps**:

1. **Install Docker with ROCm Support**:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

2. **Pull ROCm + PyTorch Base Image**:
```bash
docker pull rocm/pytorch:latest
```

3. **Create Dockerfile with JAX**:
```dockerfile
FROM rocm/pytorch:latest

# Install JAX with ROCm
RUN pip install --upgrade "jax[rocm61]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Set environment
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV ROCM_PATH=/opt/rocm

# Install your project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace
```

4. **Build and Run**:
```bash
docker build -t atom-jax-rocm .

# Run with GPU access
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it atom-jax-rocm

# Inside container, test JAX
python -c "import jax; print(jax.devices())"
```

---

## Verification Tests

Once JAX GPU is installed, run these tests to verify performance:

### Test 1: Basic GPU Detection

```python
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Should show: [RocmDevice(id=0)]
```

### Test 2: Simple Computation

```python
import jax.numpy as jnp
from jax import jit
import time

@jit
def matrix_multiply(a, b):
    return jnp.dot(a, b)

# Create large matrices
size = 4096
a = jnp.ones((size, size))
b = jnp.ones((size, size))

# Warm up JIT
_ = matrix_multiply(a, b).block_until_ready()

# Benchmark
start = time.time()
for _ in range(10):
    c = matrix_multiply(a, b).block_until_ready()
elapsed = time.time() - start

print(f"10 x {size}x{size} matrix multiplies: {elapsed:.3f}s")
print(f"Average: {elapsed/10*1000:.1f}ms per multiply")

# Expected:
# CPU: ~500-1000ms per multiply
# GPU: ~10-50ms per multiply (10-50x faster)
```

### Test 3: JAX Physics Benchmark

```bash
# Run existing benchmark with GPU
cd /home/biff/eng/atom
python benchmark_jax_vmap.py --batch_sizes 100 500 1000

# Compare CPU vs GPU performance
```

Expected results:
- **CPU** (current): 122,947 tps @ batch=500
- **GPU** (estimated): 1,000,000+ tps @ batch=500 (8-10x faster)

---

## Performance Expectations

### Physics Simulation

| Workload | CPU (Current) | GPU (Expected) | Speedup |
|----------|---------------|----------------|---------|
| Single episode | 10,065 tps | 50,000-100,000 tps | 5-10x |
| Batch=100 | 50,000 tps | 500,000 tps | 10x |
| Batch=500 | 122,947 tps | 1,000,000+ tps | 8-10x |
| Batch=1000 | 150,000 tps | 2,000,000+ tps | 13x |

### Training (SBX + vmap)

| Configuration | CPU | GPU | Speedup |
|--------------|-----|-----|---------|
| 1 env | 2,828 steps/sec | 5,000-10,000 | 2-3x |
| 10 envs | 7,000 steps/sec | 20,000-40,000 | 3-6x |
| 100 envs (vmap) | 10,000 steps/sec | 100,000+ | 10x |

---

## Troubleshooting

### Issue: JAX still using CPU after installation

**Symptoms**:
```python
>>> jax.devices()
[CpuDevice(id=0)]
```

**Solutions**:

1. **Check environment variables**:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

2. **Force GPU platform**:
```python
import jax
jax.config.update('jax_platform_name', 'rocm')
print(jax.devices())
```

3. **Verify ROCm libraries**:
```bash
# Check ROCm is accessible
ls $ROCM_PATH/lib/libamdhip64.so

# Test ROCm directly
rocm-smi
```

### Issue: ImportError: libamdhip64.so not found

**Solution**:
```bash
# Add ROCm to library path
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Unsupported GPU architecture (gfx1032)

**Solution**:
```bash
# Override GFX version to compatible target
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# gfx1032 is RDNA2, closest supported is gfx1030
```

### Issue: Out of memory errors

**Symptoms**:
```
RuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

**Solutions**:

1. **Reduce batch size**:
```python
# Instead of batch=1000
env = VmapEnvWrapper(n_envs=100)  # Smaller batch
```

2. **Enable memory preallocation**:
```python
import jax
jax.config.update('jax_platform_name', 'rocm')
# Allocate 90% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

3. **Check GPU memory**:
```bash
rocm-smi
# Look for "GPU Memory Usage"
```

---

## Decision Matrix

### Should You Install GPU JAX?

**Install if**:
- ✅ You need >10x speedup for training
- ✅ You're comfortable with 1-2 days of setup/debugging
- ✅ You have ROCm 5.7+ or 6.x installed
- ✅ You're okay with experimental features
- ✅ Training time is a bottleneck (>1 hour per run)

**Skip if**:
- ❌ Current 2.6x speedup (SBX) is sufficient
- ❌ Limited time for debugging (< 1 day)
- ❌ Training already fast enough (< 10 minutes)
- ❌ You prefer stability over performance
- ❌ ROCm version incompatible

---

## Alternative: Multi-Environment Scaling (Easier)

Instead of GPU setup, consider **Level 1 optimization** (much easier):

```python
# Use 16-32 parallel environments instead of GPU
from stable_baselines3.common.vec_env import SubprocVecEnv

envs = SubprocVecEnv([make_env() for _ in range(16)])

# Expected: 2-3x additional speedup
# Total: 2.6x (SBX) × 2.5x (multi-env) = 6.5x speedup
# Time: < 1 hour vs 1-2 days for GPU
```

See `docs/JAX_OPTIMIZATION_ROADMAP.md` for full comparison.

---

## Recommended Path

### Conservative (Recommended):
1. ✅ Use SBX (2.6x speedup) - Already done
2. ✅ Add 16 parallel environments (2.5x additional) - 1 hour
3. Total: **6.5x speedup** with minimal risk

### Aggressive:
1. ✅ Use SBX (2.6x speedup) - Already done
2. 🔄 Attempt GPU setup (10x additional) - 1-2 days
3. Total: **26x speedup** if successful

### Expert:
1. ✅ Use SBX (2.6x speedup) - Already done
2. 🔄 Setup GPU - 1-2 days
3. 🔄 Full JAX pipeline (PureJaxRL) - 3-5 days
4. Total: **100-500x speedup** if all successful

---

## Resources

### Official Documentation
- JAX ROCm: https://github.com/google/jax#pip-installation-gpu-rocm
- ROCm: https://rocm.docs.amd.com/

### Community Support
- JAX GitHub Issues: https://github.com/google/jax/issues
- ROCm Support: https://github.com/RadeonOpenCompute/ROCm/issues

### Benchmarks
- JAX Performance: https://github.com/google/jax/discussions/categories/performance

---

## Quick Start Script

For the impatient, here's a one-shot attempt:

```bash
#!/bin/bash
# quick_gpu_setup.sh

# Backup current JAX
pip freeze | grep jax > jax_cpu_backup.txt

# Uninstall CPU JAX
pip uninstall jax jaxlib -y

# Set environment (adjust ROCm version)
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Try installing JAX with ROCm 6.1 (adjust as needed)
pip install --upgrade "jax[rocm61]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

# Test
python -c "import jax; print('Devices:', jax.devices())"

# If it shows CpuDevice, rollback:
# pip install $(cat jax_cpu_backup.txt)
```

**Run**:
```bash
chmod +x quick_gpu_setup.sh
./quick_gpu_setup.sh
```

**If successful**: Proceed to benchmark GPU performance
**If failed**: Revert to CPU JAX and consider multi-env scaling instead

---

**Bottom Line**: GPU setup is high-effort, high-reward, experimental. Multi-env scaling is low-effort, medium-reward, stable. Choose based on your time budget and risk tolerance!
