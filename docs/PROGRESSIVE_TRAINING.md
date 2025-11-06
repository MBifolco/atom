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

# Run complete progressive training pipeline
python train_progressive.py --mode complete

# Quick test run (minimal settings, ~5 minutes)
python train_progressive.py --mode quick
```

### Advanced Options

```bash
# Curriculum learning only
python training/train_progressive.py --mode curriculum --timesteps 1000000

# Population training only (requires existing curriculum graduate)
python training/train_progressive.py --mode population --population 16 --generations 20

# Full pipeline with custom settings
python training/train_progressive.py \
    --mode complete \
    --algorithm ppo \
    --timesteps 2000000 \
    --population 8 \
    --generations 10 \
    --output-dir outputs/my_run
```

## Phase 1: Curriculum Learning

### Overview

Fighters progress through 5 levels of increasing difficulty, each with specific learning objectives and graduation requirements.

### Curriculum Levels

| Level | Name | Opponents | Learning Objectives | Graduation Requirement |
|-------|------|-----------|---------------------|----------------------|
| 1 | Fundamentals | Stationary targets | Basic combat, stances, damage | 90% win rate over 10 episodes |
| 2 | Basic Skills | Simple movers | Movement, positioning | 85% win rate over 20 episodes |
| 3 | Intermediate | Distance keepers | Stamina, spacing, timing | 80% win rate over 30 episodes |
| 4 | Advanced | Behavioral fighters | Strategy, adaptation | 75% win rate over 40 episodes |
| 5 | Expert | Hardcoded experts | Mastery, consistency | 70% win rate over 50 episodes |

### Test Dummies

The curriculum uses 23+ specialized test dummies organized in three categories:

#### Atomic Dummies
- **Stationary**: neutral, extended, defending, retracted stances
- **Movement**: approach, flee, shuttle, circle, wall_hugger
- **Distance**: maintain 1m, 3m, 5m spacing
- **Stamina**: waster, cycler, efficient patterns
- **Reactive**: mirror, counter, charge_on_approach

#### Behavioral Fighters
- Tank, Rusher, Dodger, Sniper
- Bumbler (random actions)
- Defender (pure defense)

#### Scenario Dummies
- Cornered fighter, exhausted fighter
- Combo sequences, bait patterns

### Implementation

```python
from src.trainers.curriculum_trainer import CurriculumTrainer

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
from src.trainers.population.population_trainer import PopulationTrainer

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
from training.train_progressive import ProgressiveTrainer

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
| curriculum_timesteps | 500,000 | Total steps for curriculum |
| population_size | 8 | Number of fighters |
| generations | 10 | Evolution iterations |
| episodes_per_generation | 500 | Training episodes per gen |
| keep_top | 0.5 | Selection pressure |
| mass_range | (60, 85) | Fighter mass variation |
| variation_factor | 0.1 | Initial population diversity |

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
    fighters/examples/tank.py
```

## Troubleshooting

### Import Errors

If you encounter module import errors:

```bash
# Run from project root
cd /home/biff/eng/atom
python training/train_progressive.py
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
from src.trainers.curriculum_trainer import CurriculumLevel

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

## Future Enhancements

Planned improvements:
- [ ] Multi-objective optimization (damage, efficiency, style)
- [ ] Transfer learning between weight classes
- [ ] Adversarial training against specific opponents
- [ ] Neural architecture search
- [ ] Distributed training support

## Conclusion

The progressive training system provides a robust framework for creating skilled AI fighters. By combining curriculum learning's structured progression with population training's evolutionary diversity, it produces fighters that are both technically proficient and strategically sophisticated.

For questions or contributions, see the main project README.