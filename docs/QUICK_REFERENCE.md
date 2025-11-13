# Quick Reference: JAX GPU Training

## 🚀 GPU Training (77x speedup)

### Setup (every time)
```bash
cd /home/biff/eng/atom
source setup_gpu.sh
```

### Verify GPU
```bash
python -c "import jax; print(jax.devices())"
# Should show: [RocmDevice(id=0)]
```

### Run Training

**Progressive Training (Recommended)**:
```bash
# GPU-accelerated progressive training
source setup_gpu.sh
python train_progressive.py --use-vmap --mode complete
```

**Direct Training (Advanced)**:
```python
from src.training.trainers.ppo.trainer import train_fighter

train_fighter(
    opponent_files=["fighters/training_opponents/training_dummy.py"],
    output_path="outputs/model.zip",
    episodes=10000,
    n_envs=250,      # 250 parallel environments on GPU
    use_vmap=True,   # Enable GPU acceleration
    verbose=True
)
```

---

## Performance Summary

| Level | Configuration | Speedup | Command |
|-------|--------------|---------|---------|
| 1+2 | SBX + 8 envs (CPU) | 3.2x | `python train_progressive.py` |
| 3 | vmap 100 (CPU) | 10-15x | `python train_progressive.py --use-vmap` |
| 4 | vmap 250 (GPU) | **77x** | `source setup_gpu.sh && python train_progressive.py --use-vmap` |

### Progressive Training Options

```bash
# CPU training (default)
python train_progressive.py --mode complete --timesteps 500000

# GPU training (77x faster)
source setup_gpu.sh
python train_progressive.py --use-vmap --mode complete --timesteps 500000

# Quick test (GPU)
source setup_gpu.sh
python train_progressive.py --use-vmap --mode quick
```

---

## Environment

**Python**: 3.11.10 (atom environment)
**JAX**: 0.7.1 with ROCm 7.1
**GPU**: AMD Radeon RX 6000 (gfx1032)
**ROCm**: 7.1.0

---

## Benchmark Results

**Physics Throughput** (GPU):
- Batch 50: 63,620 tps
- Batch 100: 167,112 tps
- Batch 250: 435,244 tps
- Batch 500: 842,488 tps (83.7x vs baseline)

**Training Time** (100M timesteps):
- Baseline: 25.4 hours
- GPU: **20 minutes** (99% faster)

---

## Troubleshooting

**GPU not detected**:
```bash
# Did you source the script?
source setup_gpu.sh

# Check environment
echo $PYENV_VERSION  # Should be: atom
python --version      # Should be: 3.11.10
```

**Import errors**:
```bash
# Ensure atom environment is active
pyenv activate atom
pip list | grep jax  # Should show jax 0.7.1
```

---

## Files

- `setup_gpu.sh` - GPU setup script (source before training)
- `benchmark_gpu.py` - GPU performance benchmarks
- `docs/FINAL_RESULTS_ALL_LEVELS.md` - Complete documentation

---

**Quick Start**: `source setup_gpu.sh && python train.py`
