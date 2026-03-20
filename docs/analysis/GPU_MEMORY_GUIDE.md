# GPU Memory Management Guide for Population Training

## Problem
Population training can cause GPU out-of-memory (OOM) errors when multiple fighters try to train in parallel, each using GPU resources. This is especially problematic with ROCm/AMD GPUs.

## Solutions Implemented

### 1. **Sequential GPU Training (Default)**
By default, population training now uses sequential training (1 fighter at a time) when GPU mode is enabled:
```bash
python train_progressive.py --use-vmap  # Automatically uses sequential training
```

Benefits:
- Prevents GPU OOM errors
- Each fighter gets full GPU resources
- Still uses JAX vmap for fast parallel environments (250 envs)

### 2. **CPU-Only Mode for Population**
If you still experience GPU issues, force CPU-only mode for population training:
```bash
python train_progressive.py --use-vmap --population-cpu-only
```

This will:
- Use GPU for curriculum training (fast)
- Switch to CPU for population training (stable)
- Avoid all GPU memory issues in population phase

### 3. **Memory Cleanup Between Phases**
The system now performs comprehensive cleanup between curriculum and population phases:
- Closes all JAX/vmap environments
- Deletes PPO models and buffers
- Clears PyTorch GPU cache
- Clears JAX compilation cache
- Runs garbage collection

### 4. **Per-Process Memory Limits**
Each training subprocess now sets:
- PyTorch memory fraction limit (75% per process)
- JAX memory growth limits
- ROCm/HIP environment variables

## Troubleshooting

### If you see "Hip error: 'out of memory'"
1. First try with default settings (sequential GPU):
   ```bash
   python train_progressive.py --use-vmap
   ```

2. If that fails, use CPU-only for population:
   ```bash
   python train_progressive.py --use-vmap --population-cpu-only
   ```

3. Or use full CPU mode (slower but most stable):
   ```bash
   python train_progressive.py  # No --use-vmap flag
   ```

### Monitor GPU Memory
Check GPU memory usage:
```bash
rocm-smi
```

### Custom Parallel Settings
If you have lots of GPU memory and want parallel training:
```bash
python train_progressive.py --use-vmap --n-parallel-fighters 2
```

⚠️ **Warning**: This may cause OOM errors. Only use if you have 16GB+ VRAM.

## Performance Comparison

| Mode | Curriculum Speed | Population Speed | Stability |
|------|-----------------|------------------|-----------|
| Full GPU | 77x faster | Fast | May OOM |
| GPU + CPU Population | 77x faster | Normal | Stable |
| Full CPU | Normal | Normal | Most Stable |

## Recommended Settings

### For 8GB GPUs (RX 7600, etc):
```bash
python train_progressive.py --use-vmap --population-cpu-only
```

### For 16GB+ GPUs:
```bash
python train_progressive.py --use-vmap
```

### For CPU-only systems:
```bash
python train_progressive.py
```