# Progressive Training System

## Overview

The progressive training system combines curriculum learning with population-based training to create strong, diverse AI fighters for Atom Combat. This two-phase approach ensures fighters learn fundamental skills before engaging in complex strategic evolution.

## Architecture

```
Progressive Training Pipeline
├── Phase 1: Curriculum Learning
│   ├── Level 1: Stationary Targets (fundamentals)
│   ├── Level 2: Simple Movement (basic skills)
│   ├── Level 3: Distance Management (intermediate)
│   ├── Level 4: Behavioral Fighters (advanced)
│   └── Level 5: Hardcoded Experts (mastery)
│
└── Phase 2: Population Training
    ├── Initialize from curriculum graduate
    ├── Create diverse population variants
    ├── Train through self-play
    └── Evolve and select champions
```

## Quick Start

### Basic Usage

```bash
# Navigate to project root
cd /path/to/atom

# Run complete progressive training pipeline (CPU)
python train_progressive.py --mode complete

# GPU-accelerated training (77x faster!)
python train_progressive.py --mode complete --use-vmap

# Quick test run (minimal settings, ~5 minutes)
python train_progressive.py --mode quick
```

### Advanced Options

```bash
# Curriculum learning only
python train_progressive.py --mode curriculum --timesteps 4000000

# GPU-accelerated curriculum
python train_progressive.py --mode curriculum --use-vmap --timesteps 4000000

# Population training only (requires existing curriculum graduate)
python train_progressive.py --mode population --population 16 --generations 20 --episodes-per-gen 4000

# Full pipeline with custom settings and GPU support
python train_progressive.py \
    --mode complete \
    --algorithm ppo \
    --timesteps 4000000 \
    --population 8 \
    --generations 20 \
    --episodes-per-gen 4000 \
    --cores 8 \
    --use-vmap \
    --output-dir outputs/my_run
```

## Phase 1: Curriculum Learning

### Overview

Fighters progress through 5 levels of increasing difficulty, each with specific learning objectives and graduation requirements.

### Curriculum Levels

**Hardcore Graduation System**: Fighters must maintain elite standards (80-90% recent win rates) throughout all levels. Each level requires BOTH recent win rate and overall win rate thresholds to prevent lucky streaks.

| Level | Name | Opponents | Min Episodes | Learning Objectives | Graduation (Recent / Overall) |
|-------|------|-----------|--------------|---------------------|----------------------|
| 1 | Fundamentals | 3 stationary atomic dummies | 200 | Basic attacking, stance usage | 90% / 50% over 50 episodes |
| 2 | Basic Skills | 6 simple movement dummies | 300 | Pursuit, evasion, predictive movement | 88% / 50% over 50 episodes |
| 3 | Intermediate | 6 distance/stamina/movement dummies | 400 | Spacing, resource management, wall combat | 85% / 50% over 50 episodes |
| 4 | Advanced | 6 stance/behavior dummies | 500 | Complex strategies, counter-strategies | 83% / 50% over 50 episodes |
| 5 | Expert | 5 example fighters | 600 | Mastery against expert opponents | 80% / 50% over 50 episodes |

**Dual-Requirement System**:
- **Recent Win Rate**: Performance over last 50 episodes (measures current skill)
- **Overall Win Rate**: Performance across ALL episodes at level (prevents lucky streaks)
- **Both Must Pass**: Ensures consistent, sustained performance before graduation

### Test Dummies

The curriculum currently draws opponents from `fighters/test_dummies/atomic/`, then
uses hand-authored experts from `fighters/examples/` at Level 5.

#### Atomic Dummies
Representative opponents:
- **Fundamentals**: `stationary_neutral`, `stationary_extended`, `stationary_defending`
- **Basic Skills**: `approach_slow`, `flee_always`, `shuttle_slow`, `shuttle_medium`, `circle_left`, `circle_right`
- **Intermediate**: `distance_keeper_1m`, `stamina_efficient`, `charge_on_approach`, `forward_mover`, `backward_mover`, `sideways_mover_smooth`
- **Advanced**: `aggressive_stance_switcher`, `balanced_stance_switcher`, `defensive_stance_switcher`, `forward_charger`, `oscillator`, `retreater`

#### Hardcoded Experts (Level 5)
Expert opponents in `fighters/examples/`:
- `boxer`, `counter_puncher`, `out_fighter`, `slugger`, `swarmer`

### Implementation

```python
from src.training.trainers.curriculum_trainer import CurriculumTrainer

trainer = CurriculumTrainer(
    algorithm="ppo",
    output_dir="outputs/curriculum",
    n_envs=4,
    verbose=True
)

# Train through all levels
trainer.train(total_timesteps=500_000)

# Access the trained model
model_path = "outputs/curriculum/models/curriculum_graduate.zip"
```

## Phase 2: Population Training

### Overview

Creates a diverse population of fighters initialized from the curriculum graduate, then evolves them through self-play and selection.

### Population Initialization

1. Load curriculum graduate model
2. Create population with variations (±10% parameter noise)
3. Assign unique masses (60-85kg range)
4. Track lineage and generation

### Evolution Process

```
Generation Loop:
1. Matchmaking (balanced pairs based on ELO)
2. Training (self-play episodes)
3. Evaluation (round-robin matches)
4. Selection (keep top 50%)
5. Reproduction (mutate winners)
```

### Implementation

```python
from src.training.trainers.population.population_trainer import PopulationTrainer

trainer = PopulationTrainer(
    population_size=8,
    algorithm="ppo",
    output_dir="outputs/population",
    verbose=True
)

# Initialize from curriculum graduate
trainer.initialize_population(
    base_model_path="outputs/curriculum/models/curriculum_graduate.zip",
    variation_factor=0.1
)

# Run evolution
trainer.train(
    generations=10,
    episodes_per_generation=500,
    keep_top=0.5
)
```

## Complete Pipeline

The `ProgressiveTrainer` class orchestrates both phases:

```python
from src.training.pipelines import ProgressiveTrainer

trainer = ProgressiveTrainer(
    algorithm="ppo",
    output_dir="outputs/progressive",
    verbose=True
)

# Run complete pipeline
trainer.run_complete_pipeline(
    curriculum_timesteps=500_000,
    population_generations=10,
    population_size=8
)

# Export best fighters
trainer.export_best_fighters(top_n=3)
```

## Output Structure

```
outputs/progressive_TIMESTAMP/
├── curriculum/
│   ├── models/
│   │   ├── level_1_graduate.zip
│   │   ├── level_2_graduate.zip
│   │   └── curriculum_graduate.zip
│   └── logs/
│       └── training_progress.json
│
├── population/
│   ├── models/
│   │   ├── generation_0/
│   │   └── generation_N/
│   ├── logs/
│   │   └── population_training_*.log
│   └── rankings.txt
│
├── champions/
│   ├── champion_1_*.zip
│   ├── champion_2_*.zip
│   └── champion_3_*.zip
│
└── training_report_*.json
```

## Configuration Options

### Algorithm Selection

- **PPO** (default): Best for curriculum learning, stable training
- **SAC**: Better exploration, more sample efficient

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| curriculum_timesteps | 4,000,000 | Total steps for curriculum (recommended for full training) |
| population_size | 8 | Number of fighters in population |
| generations | 20 | Evolution iterations |
| episodes_per_generation | 4000 | Training episodes per generation |
| cores | 1 | CPU cores for parallel training (set to 8+ for speed) |
| keep_top | 0.5 | Selection pressure (keep top 50%) |
| mass_range | (60, 85) | Fighter mass variation in kg |
| variation_factor | 0.1 | Initial population diversity (±10% parameter noise) |

## Performance Tracking

### Curriculum Metrics
- Win rate per level
- Episodes to graduation
- Damage dealt/taken ratios
- Stamina efficiency

### Population Metrics
- ELO ratings
- Win/loss/draw records
- Population diversity (ELO spread)
- Lineage tracking

## Exporting Champions

Trained fighters are automatically exported in formats compatible with `atom_fight.py`:

```bash
# Use exported champion
python atom_fight.py \
    outputs/progressive/champions/champion_1.py \
    fighters/examples/boxer.py
```

## Output Structure

After training completes, you'll find:

```
outputs/progressive_TIMESTAMP/
├── curriculum/
│   ├── models/
│   │   └── curriculum_graduate.zip     # Trained curriculum model
│   └── logs/
│       ├── curriculum_training_*.log   # Training progress
│       └── tensorboard/                # TensorBoard logs
│
├── population/
│   ├── models/
│   │   ├── generation_0/               # Initial population
│   │   ├── generation_1/               # Evolved fighters
│   │   └── generation_N/
│   └── logs/
│       └── population_training_*.log   # Evolution logs
│
└── training_report_*.json              # Summary statistics
```

**Exported Champions:**
- Automatically exported to `fighters/AIs/`
- Each fighter has: `.zip` model, `.py` wrapper, `README.md`
- Ready to use with `atom_fight.py`

**Monitoring Progress:**
```bash
# Watch training in real-time
tail -f outputs/progressive_*/curriculum/logs/*.log
tail -f outputs/progressive_*/population/logs/*.log

# View TensorBoard metrics
tensorboard --logdir outputs/progressive_*/curriculum/logs/tensorboard
```

## Troubleshooting

### Import Errors

All imports are now properly structured in `src/training/`. Simply run from project root:

```bash
python train_progressive.py --mode quick
```

### Memory Issues

For large populations or long training:
- Reduce `population_size`
- Lower `n_envs_per_fighter`
- Use `--algorithm sac` (more memory efficient)

### Training Stagnation

If fighters stop improving:
- Increase `variation_factor` for more diversity
- Adjust `keep_top` for different selection pressure
- Add more curriculum levels

## Best Practices

1. **Start Small**: Test with `--mode quick` before full runs
2. **Monitor Progress**: Check logs regularly for convergence
3. **Save Checkpoints**: Models are saved each generation
4. **Tune Gradually**: Adjust one parameter at a time
5. **Validate Results**: Test champions against various opponents

## Advanced Usage

### Custom Curriculum Levels

```python
from src.training.trainers.curriculum_trainer import CurriculumLevel

custom_level = CurriculumLevel(
    name="Custom Challenge",
    opponents=["fighters/custom/opponent.py"],
    graduation_win_rate=0.8,
    graduation_episodes=25,
    timesteps_per_episode=1000
)

trainer.curriculum.append(custom_level)
```

### Population Seeding

```python
# Seed population with multiple base models
for model_path in ["model1.zip", "model2.zip"]:
    trainer.initialize_population(
        base_model_path=model_path,
        variation_factor=0.05
    )
```

## Results and Expectations

### Typical Training Timeline

- **Curriculum Phase**: 2-4 hours (500k timesteps)
- **Population Phase**: 4-8 hours (10 generations)
- **Total Pipeline**: 6-12 hours

### Expected Performance

After complete training:
- Champions achieve 70%+ win rate vs hardcoded experts
- Diverse strategies emerge (aggressive, defensive, balanced)
- Consistent performance across different opponent types

## Troubleshooting

### Common Issues

#### Monitor File Handle Crash
**Error**: `ValueError: I/O operation on closed file`

**Cause**: Previous versions attempted to close and recreate VecEnv during level transitions, which closed Monitor file handlers mid-episode.

**Fix (Nov 2024)**: The system now uses `set_opponent()` to dynamically change opponents without recreating environments, keeping Monitor file handlers open throughout training.

#### Test Dummy Snapshot Format Errors
**Error**: `KeyError: 'position'` or `KeyError: 'stamina_max'`

**Cause**: Test dummies using incorrect snapshot field names. The protocol provides:
- `snapshot["opponent"]["distance"]` (not `"position"`)
- `snapshot["you"]["max_stamina"]` (not `"stamina_max"`)
- `snapshot["you"]["max_hp"]` (not `"hp_max"`)

**Fix (Nov 2024)**: All 32 test dummies updated to use correct snapshot format with position-based heuristics for determining opponent direction.

#### Infinite Expert Graduation Loop
**Error**: Training completes all 5 levels but keeps graduating from Expert repeatedly

**Cause**: `should_graduate()` didn't check if curriculum was already complete, allowing infinite re-graduation from the final level.

**Fix (Nov 2024)**: Added early return in `should_graduate()` when `current_level >= len(curriculum)`.

#### Graduation Check Log Spam
**Error**: Logs flooded with graduation checks every single episode when recent win rate is high

**Cause**: Graduation check logged on every episode when recent threshold was met, even if overall threshold wasn't met yet.

**Fix (Nov 2024)**: Now only logs graduation checks when actually graduating OR every 100 episodes (prevents spam while maintaining visibility).

#### Lucky Streak Graduations
**Error**: Fighter graduates after a few lucky wins despite poor overall performance

**Cause**: Only checking recent win rate (last 50 episodes) allowed statistical flukes to trigger graduation.

**Fix (Nov 2024)**: Added dual-requirement system that checks BOTH recent AND overall win rates. Overall must be within 15% of graduation requirement to ensure sustained performance.

#### Low Win Rates
**Issue**: Fighter not graduating after many timesteps

**Solutions**:
- Increase `--timesteps` (recommended: 4M for full curriculum)
- Check reward breakdown in logs - should see positive damage rewards
- Ensure test dummies are using correct snapshot format
- Verify Monitor is reporting rollout stats correctly

#### Training Slowdown
**Issue**: Training taking too long

**Solutions**:
- Use `--cores 8` or higher for multicore parallel training
- Increase `--episodes-per-gen` (recommended: 4000) for population training
- Consider using `--mode curriculum` only if you don't need population diversity

## GPU Acceleration

### Performance

**CPU Training** (8 parallel environments):
- Curriculum: ~2,800 steps/sec
- Population (8 fighters): ~400 steps/sec per fighter

**GPU Training** (with --use-vmap):
- Curriculum: ~16,000 steps/sec (250 parallel environments) - **77x speedup**
- Population (8 fighters): ~2,000 steps/sec per fighter (45 envs each)

### GPU Configuration

**Curriculum Training**:
- 250 parallel GPU environments
- Single fighter training
- ~7-8 GB VRAM usage

**Population Training**:
- 45 parallel GPU environments per fighter
- 8 fighters training simultaneously
- Opponent models load on CPU to save GPU memory
- Total: 360 parallel environments (~7-8 GB VRAM)

### Setup Requirements

**AMD GPU** (ROCm):
```bash
# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Run training
python train_progressive.py --mode complete --use-vmap
```

**NVIDIA GPU** (CUDA):
```bash
# JAX should detect CUDA automatically
python train_progressive.py --mode complete --use-vmap
```

See `docs/GPU_SETUP_GUIDE.md` for detailed GPU setup instructions.

## Future Enhancements

Planned improvements:
- [ ] Multi-objective optimization (damage, efficiency, style)
- [ ] Transfer learning between weight classes
- [ ] Adversarial training against specific opponents
- [ ] Neural architecture search
- [ ] Distributed training support
- [x] GPU acceleration for model training (COMPLETED)

## Conclusion

The progressive training system provides a robust framework for creating skilled AI fighters. By combining curriculum learning's structured progression with population training's evolutionary diversity, it produces fighters that are both technically proficient and strategically sophisticated.

For questions or contributions, see the main project README.
