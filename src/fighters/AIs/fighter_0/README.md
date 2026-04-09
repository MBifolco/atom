# fighter_0

Trained AI Fighter from Population-Based Training

## Stats

- **Generation**: 0
- **Lineage**: founder
- **Mass**: 70.0kg
- **Training Episodes**: 0

### Performance Metrics

- **ELO Rating**: 1610
- **Win Rate**: 100.0%
- **Record**: 10W - 0L - 0D
- **Total Matches**: 10

## Usage

This fighter is compatible with `atom_fight.py`:

```bash
# Fight against another AI
python atom_fight.py fighters/AIs/fighter_0/fighter_0.py fighters/examples/boxer.py

# Watch the fight in terminal
python atom_fight.py fighters/AIs/fighter_0/fighter_0.py fighters/examples/slugger.py --watch

# Generate HTML replay
python atom_fight.py fighters/AIs/fighter_0/fighter_0.py fighters/examples/counter_puncher.py --html replay.html

# Custom mass (if different from trained mass)
python atom_fight.py fighters/AIs/fighter_0/fighter_0.py fighters/examples/boxer.py --mass-a 70
```

## Files

- `fighter_0.py` - Python wrapper with decide() function
- `fighter_0.onnx` - ONNX model (neural network weights)
- `README.md` - This file

## Requirements

```bash
pip install onnxruntime numpy
```

## Strategy

This fighter learned its strategy through population-based training, competing against
other evolving AI fighters. Its behavior emerged from reinforcement learning rather than
being hand-coded.

**Training Algorithm**: PPO
**Population Size**: 2
**Generation**: 0

## Notes

- The fighter was trained at 70.0kg mass. Performance may vary with different masses.
- Win rate of 100.0% was achieved against the training population.
- The ONNX model requires `onnxruntime` to run inference.
