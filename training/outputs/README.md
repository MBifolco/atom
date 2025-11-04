# Training Outputs

This directory contains trained fighter models, organized by model name.

## File Types

- **`.zip`** - Stable-Baselines3 model files (can be loaded for continued training)
- **`.onnx`** - ONNX exported models (platform-independent, inference only)
- **`.py`** - Python wrapper files (can be used directly with `atom_fight.py`)

## Organization

Each trained model gets its own directory to keep files organized.

### Single Training
When training with `--opponent` or `--opponents`:
```
outputs/
└── my_fighter/
    ├── model.zip
    ├── model.onnx
    └── fighter.py
```

### Curriculum Training
When training with `--curriculum`:
```
outputs/
└── parzival/
    ├── level1.zip      # Level 1: Training Dummy
    ├── level1.onnx
    ├── level1.py
    ├── level2.zip      # Level 2: Wanderer (continues from level1)
    ├── level2.onnx
    ├── level2.py
    ├── level3.zip      # Level 3: Bumbler (continues from level2)
    ├── level3.onnx
    └── level3.py
```

**Note:** Curriculum training uses continual learning - each level continues training from the previous level to prevent catastrophic forgetting.

### Checkpoints and Logs
Periodic checkpoints and training logs are stored with the model:
```
outputs/
└── parzival_1.0.12/
    ├── level1.zip
    ├── level1.onnx
    ├── level1.py
    ├── checkpoints/
    │   ├── fighter_checkpoint_100000_steps.zip
    │   ├── fighter_checkpoint_200000_steps.zip
    │   └── ...
    └── logs/                              # NEW: Logs live with their models
        ├── parzival_1.0.12_level_1_timestamp.log
        ├── parzival_1.0.12_level_2_timestamp.log
        └── ...
```

**Benefits:**
- Clear association: logs are with their models
- Easy cleanup: delete the entire fighter directory
- No shared/mixed log directories

## Using Trained Fighters

```bash
# Single training
python atom_fight.py training/outputs/my_fighter/fighter.py fighters/examples/tank.py --html replay.html

# Curriculum training - test a specific level
python atom_fight.py training/outputs/parzival/level3.py fighters/examples/tank.py --html replay.html

# Test all levels from curriculum
for level in training/outputs/parzival/level*.py; do
  echo "Testing $level..."
  python atom_fight.py "$level" fighters/examples/tank.py
done
```
