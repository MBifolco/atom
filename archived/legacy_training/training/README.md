## 🧠 Atom Combat - Fighter Training Guide

**Train AI fighters using reinforcement learning (PPO).**

**📖 New to ML training? Start with [../../../docs/PROGRESSIVE_TRAINING.md](../../../docs/PROGRESSIVE_TRAINING.md).**
*Explains how AI training works with simple language and diagrams.*

---

## Directory Structure

```
training/
├── train_fighter.py       # Training CLI
├── requirements.txt       # Python dependencies
├── src/                   # Training infrastructure
│   ├── gym_env.py        # Gymnasium wrapper
│   ├── trainer.py        # PPO trainer with curriculum support
│   └── onnx_fighter.py   # ONNX export/inference
├── outputs/               # Trained models (auto-created)
│   ├── *.zip             # Stable-Baselines3 models
│   ├── *.onnx            # ONNX exported models
│   ├── *.py              # Python wrapper files
│   └── checkpoints/      # Training checkpoints
└── logs/                  # Training logs (auto-created)
    └── *_training_*.log  # Detailed episode logs
```

**All trained models automatically save to `outputs/`**
**All training logs automatically save to `logs/`**

---

## Quick Start

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

**What this installs:**
- `stable-baselines3` - PPO reinforcement learning
- `gymnasium` - OpenAI Gym interface
- `torch` - PyTorch (neural networks)
- `onnx` + `onnxruntime` - Model export/inference

---

### 2. Train Your First Fighter

```bash
# Train against Training Dummy (easiest opponent)
python train_fighter.py --opponent ../fighters/training_opponents/training_dummy.py --output my_first_fighter --episodes 1000 --create-wrapper

# This will:
# - Train using 10 CPU cores in parallel
# - Use same mass (70kg) for both fighter and opponent
# - Auto-stop when performance plateaus
# - Save to outputs/:
#   - my_first_fighter.zip (model)
#   - my_first_fighter.onnx (ONNX export)
#   - my_first_fighter.py (Python wrapper)
# - Save log to logs/my_first_fighter_training_TIMESTAMP.log
```

**Expected output:**
```
============================================================
           ATOM COMBAT - FIGHTER TRAINING
============================================================

Loading 1 opponent(s)...
  ✓ tank.py

Training Configuration:
  Fighter mass: 70.0kg
  Opponent mass: 70.0kg (same as fighter)
  Target episodes: ~1,000
  Parallel environments: 10
  Max ticks/episode: 1000
  Estimated timesteps: ~500,000

Creating 10 parallel environments...
  ✓ Environments ready

Initializing PPO model...
  ✓ Model created

Starting training...
------------------------------------------------------------
Step 1,000 | Episodes: 23 | Mean Reward: -45.2 | Mean Length: 234 ticks
Step 2,000 | Episodes: 47 | Mean Reward: -12.8 ⬆️ | Mean Length: 198 ticks
...
  📈 Improvement detected: 23.5
...
  🛑 Training stopped: Performance plateaued at 45.8
------------------------------------------------------------

Saving trained model to: my_first_fighter.zip
  ✓ Model saved

============================================================
                    TRAINING COMPLETE
============================================================
```

---

### 3. Test Your Fighter

```bash
# Test against Tank (from parent directory)
cd ..
python atom_fight.py training/outputs/my_first_fighter.py fighters/examples/tank.py --html test_fight.html

# Fight against other opponents
python atom_fight.py training/outputs/my_first_fighter.py fighters/examples/rusher.py --watch
```

---

## Training Options

### Basic Usage

```bash
python train_fighter.py --opponent ../fighters/tank.py --output my_fighter
```

### Train Against Multiple Opponents (Recommended!)

```bash
# Train against all hardcoded fighters
python train_fighter.py --opponents ../fighters/*.py --output versatile_fighter

# Train against specific opponents
python train_fighter.py --opponents ../fighters/tank.py ../fighters/rusher.py --output my_fighter
```

**Why multiple opponents?**
- Prevents overfitting to one strategy
- Creates more robust fighters
- Learns to adapt to different styles

---

### Full Options

```bash
python train_fighter.py \\
    --opponents ../fighters/*.py \\    # Train against all fighters
    --output champion \\                # Output name
    --episodes 50000 \\                 # Target episodes (~50K is good)
    --cores 10 \\                       # CPU cores (default: 10)
    --mass 70 \\                        # Your fighter's mass
    --opponent-mass 70 \\               # Opponent mass (default: same as --mass)
    --patience 10 \\                    # Plateau patience
    --create-wrapper                    # Create .py wrapper file
```

**Parameters:**

| Option | Default | Description |
|--------|---------|-------------|
| `--opponent` | required | Single opponent file |
| `--opponents` | required | Multiple opponents (or wildcard) |
| `--output` | required | Output name (creates .zip and .onnx) |
| `--episodes` | 10000 | Target number of training episodes |
| `--cores` | 10 | CPU cores for parallel training |
| `--mass` | 70.0 | Your fighter's mass (kg) |
| `--opponent-mass` | 75.0 | Opponent mass (kg) |
| `--max-ticks` | 1000 | Max ticks per episode |
| `--patience` | 5 | Checks without improvement before stopping |
| `--checkpoint-freq` | 10000 | Save checkpoint every N steps |
| `--tensorboard` | none | TensorBoard log directory |
| `--quiet` | false | Minimal output |
| `--create-wrapper` | false | Create standalone .py wrapper |

---

## How It Works

### Architecture

**Observation Space (9 values):**
```python
[
    your_position,          # 0-12.5m
    your_velocity,          # -3 to +3 m/s
    your_hp_normalized,     # 0-1
    your_stamina_normalized,# 0-1
    opponent_distance,      # 0-12.5m
    opponent_rel_velocity,  # -5 to +5 m/s
    opponent_hp_normalized, # 0-1
    opponent_stamina_norm,  # 0-1
    arena_width            # 12.48m
]
```

**Action Space:**
- **Acceleration**: Continuous [-1, 1] (scaled to ±4.4 m/s²)
- **Stance**: Discrete [0-3] (neutral, extended, retracted, defending)

**Neural Network:**
- Small MLP (multi-layer perceptron)
- 2-3 hidden layers, 64 units each
- Fast inference (<1ms per decision)

---

### Reward Function

```python
reward = (damage_dealt - damage_taken) + win_bonus

# Per-step:
reward = damage_to_opponent - damage_to_self

# Episode end:
if won:
    reward += 100
else:
    reward -= 100
```

**Why this reward?**
- Encourages dealing damage
- Penalizes taking damage
- Big bonus for winning
- Dense signal (every collision gives feedback)

---

### Training Process

1. **Parallel Environments (10 cores)**
   - 10 matches running simultaneously
   - Each core runs independent arena
   - Dramatically faster than sequential

2. **PPO Algorithm**
   - Proven RL method for continuous control
   - Stable training (won't diverge)
   - Sample-efficient

3. **Auto-Stopping**
   - Monitors mean reward over last 100 episodes
   - Stops when no improvement for `patience` checks
   - Prevents overtraining

4. **Checkpointing**
   - Saves model every 10K steps
   - Resume from checkpoints if training crashes
   - Located in `checkpoints/` directory

---

## Training Strategies

### Strategy 1: Quick Test (5 min)

```bash
python train_fighter.py --opponent ../fighters/tank.py --output test --episodes 1000
```

**Use for:**
- Testing the system works
- Quick experiments
- Debugging

---

### Strategy 2: Single Opponent Specialist (30 min)

```bash
python train_fighter.py --opponent ../fighters/tank.py --output tank_killer --episodes 20000
```

**Results:**
- Very good against that specific opponent
- Might struggle against others
- Good for tournaments where you know opponent

---

### Strategy 3: Generalist (2-4 hours)

```bash
python train_fighter.py --opponents ../fighters/*.py --output generalist --episodes 50000
```

**Results:**
- Robust against multiple styles
- Doesn't overfit
- Recommended for tournaments

---

### Strategy 4: Mass-Optimized

```bash
# Train lightweight (65kg)
python train_fighter.py --opponents ../fighters/*.py --output lightweight --mass 65 --episodes 30000

# Train heavyweight (85kg)
python train_fighter.py --opponents ../fighters/*.py --output heavyweight --mass 85 --episodes 30000
```

**Use for:**
- Exploring mass tradeoffs
- Finding optimal weight class
- Creating specialized builds

---

## Monitoring Training

### Progress Output

```
Step 5,000 | Episodes: 115 | Mean Reward: 12.3 ⬆️ | Mean Length: 187 ticks
```

**What this means:**
- **Step**: Total environment steps taken
- **Episodes**: Matches completed
- **Mean Reward**: Average reward (higher = better)
  - Negative = losing badly
  - ~0 = breaking even
  - Positive = winning
- **Mean Length**: Average match duration
  - Short (~100 ticks) = quick knockouts
  - Long (~500 ticks) = careful fights
- **⬆️** = Improvement detected (new best!)

---

### TensorBoard (Advanced)

```bash
# Train with TensorBoard logging
python train_fighter.py --opponent ../fighters/tank.py --output my_fighter --tensorboard logs/

# In another terminal:
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

**TensorBoard shows:**
- Reward curves over time
- Episode lengths
- Policy loss
- Value loss
- Learning rate
- etc.

---

## Using Trained Fighters

### Option 1: Use ONNX Directly (Python API)

```python
from training.src.onnx_fighter import ONNXFighter

fighter = ONNXFighter("my_fighter.onnx")
action = fighter.decide(snapshot)
```

---

### Option 2: Create Wrapper (Standalone .py)

```bash
python train_fighter.py --opponent ../fighters/tank.py --output my_fighter --create-wrapper
```

This creates `my_fighter.py`:
```python
def decide(snapshot):
    # Loads my_fighter.onnx automatically
    # Compatible with atom_fight.py
    ...
```

**Use it:**
```bash
cd ..
python atom_fight.py training/my_fighter.py fighters/tank.py
```

---

## Troubleshooting

### Training is Slow

**Solution 1: Use more cores**
```bash
python train_fighter.py ... --cores 16
```

**Solution 2: Reduce episodes**
```bash
python train_fighter.py ... --episodes 5000
```

---

### Fighter Isn't Learning

**Check mean reward trend:**
- Still negative after 5K steps? Might need longer
- Stuck at same value? Try different opponent or mass
- Oscillating wildly? Normal early in training

**Solutions:**
- Train longer (more episodes)
- Try different opponent
- Adjust fighter mass
- Check if opponent is too hard

---

### Out of Memory

**Solution: Reduce cores**
```bash
python train_fighter.py ... --cores 4
```

Each environment uses ~100-200MB RAM.

---

### ONNX Export Fails

**Common issue:** PyTorch/ONNX version mismatch

**Solution:**
- Use the .zip model directly with Stable-Baselines3
- Or update: `pip install --upgrade torch onnx onnxruntime`

---

## Next Steps

After training:

1. **Test vs all opponents**
   ```bash
   cd ..
   for opponent in fighters/*.py; do
       python atom_fight.py training/my_fighter.py $opponent --html "replay_$(basename $opponent .py).html"
   done
   ```

2. **Calculate win rate**
   - Run 100 matches vs each opponent
   - Track wins/losses
   - Adjust if needed

3. **Train v2**
   - Use lessons learned
   - Try different mass
   - Train against your v1!

4. **Share your fighter**
   - Export to ONNX
   - Include metadata (mass, training opponents, win rates)
   - Submit to tournaments

---

## Advanced: Self-Play

*Coming soon: Train fighters against themselves to discover novel strategies.*

---

## Files Created

After training with `--output my_fighter`:

```
my_fighter.zip              # Stable-Baselines3 model (for retraining)
my_fighter.onnx             # ONNX model (for deployment)
my_fighter.py               # Wrapper (if --create-wrapper)
checkpoints/                # Training checkpoints
  fighter_checkpoint_10000_steps.zip
  fighter_checkpoint_20000_steps.zip
  ...
```

---

**Ready to train?**

```bash
cd training
pip install -r requirements.txt
python train_fighter.py --opponent ../fighters/tank.py --output my_first_ai --episodes 5000 --create-wrapper
cd ..
python atom_fight.py training/my_first_ai.py fighters/tank.py --html victory.html
```

Let's see if your AI can beat the hand-coded fighters! 🥊
