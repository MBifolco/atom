# Population-Based Training for Atom Combat

## Overview

Population-based training is a major advancement for Atom Combat that trains multiple fighters simultaneously, allowing them to learn from each other and evolve diverse strategies through competition and selection.

## Key Concepts

### 1. Population Management
- **Multiple Learning Agents**: Instead of training a single fighter, we maintain a population of 8-16 fighters
- **Concurrent Training**: Each fighter trains against others in the population
- **Diverse Strategies**: Different fighters develop different approaches naturally

### 2. ELO Rating System
- **Performance Tracking**: Each fighter has an ELO rating (starting at 1500)
- **Relative Skill Measurement**: Ratings update based on match outcomes
- **Matchmaking**: Use ELO to create balanced matches or teaching pairs

### 3. Evolution Mechanics
- **Selection**: Keep top performers based on ELO rankings
- **Mutation**: Create variations of successful fighters
- **Replacement**: Replace weak fighters with mutated versions of strong ones
- **Generational Progress**: Population evolves over multiple generations

## Implementation Architecture

### Core Components

#### `EloTracker` (`training/src/trainers/population/elo_tracker.py`)
- Manages fighter statistics and ratings
- Calculates expected outcomes and rating updates
- Tracks match history and performance metrics
- Provides diversity metrics for population health

Key features:
- K-factor of 32 for rating volatility
- Win/loss/draw tracking
- Damage ratio statistics
- Matchup predictions

#### `PopulationTrainer` (`training/src/trainers/population/population_trainer.py`)
- Orchestrates the entire population training process
- Manages multiple PPO/SAC models simultaneously
- Implements matchmaking strategies
- Handles evolution and selection

Key features:
- Supports 8-16 fighters per population
- Parallel training with SubprocVecEnv
- Multiple matchmaking strategies (balanced, random, teaching)
- Generation-based evolution with configurable keep rate

#### CLI Interface (`training/train_population.py`)
```bash
python train_population.py \
  --population 8 \
  --generations 10 \
  --episodes 500 \
  --algorithm ppo \
  --mass-range 60 85 \
  --evolution-freq 2 \
  --keep-top 0.5
```

## Training Process

### 1. Initialization
```python
# Create diverse initial population
for i in range(population_size):
    mass = np.random.uniform(60, 85)
    model = PPO("MlpPolicy", env, learning_rate=1e-4)
    fighter = PopulationFighter(name, model, mass)
```

### 2. Training Loop
Each generation consists of:
1. **Matchmaking**: Create fighter pairs using multiple strategies
2. **Training**: Each fighter trains against assigned opponents
3. **Evaluation**: Run matches to update ELO ratings
4. **Evolution**: Replace weak fighters with mutations of strong ones

### 3. Matchmaking Strategies
- **Balanced Matches** (50%): Pair fighters with similar ELO
- **Random Matches** (30%): Maintain diversity
- **Teaching Matches** (20%): Strong vs weak for knowledge transfer

### 4. Evolution Process
```python
# Keep top 50% of population
survivors = top_fighters_by_elo[:keep_count]

# Replace bottom 50%
for weak_fighter in bottom_fighters:
    parent = weighted_random_choice(survivors)  # Weight by ELO
    new_fighter = mutate(parent)  # Vary mass, learning rate
    replace(weak_fighter, new_fighter)
```

## Benefits Over Single Fighter Training

### 1. Prevents Overfitting
- Fighters must succeed against multiple opponents
- Can't exploit single opponent's weakness
- Robust strategies emerge naturally

### 2. Diverse Strategies
- Different fighters develop different styles
- Population maintains strategy diversity
- Natural rock-paper-scissors dynamics

### 3. Faster Learning
- Parallel training across population
- Knowledge transfer through evolution
- Best strategies propagate quickly

### 4. Better Evaluation
- ELO provides continuous skill measurement
- Can track improvement over time
- Identifies truly strong fighters

## Example Results

From the demonstration:
```
Generation 1:
  Alpha (aggression: 0.52) → ELO: 1513
  Gamma (aggression: 0.75) → ELO: 1567  [TOP]

Generation 3 (after evolution):
  Gamma → ELO: 1587 (75% win rate)
  Echo_v2 (mutant) → ELO: 1547 (62% win rate)

Population Diversity:
  ELO Range: 181
  ELO Std Dev: 40.3
```

## Usage Examples

### Quick Test
```bash
# Small population, fast training
python train_population.py --population 4 --generations 2 --episodes 100
```

### Standard Training
```bash
# Balanced configuration
python train_population.py --population 8 --generations 10 --episodes 500
```

### Large Diverse Population
```bash
# Maximum diversity
python train_population.py \
  --population 16 \
  --generations 20 \
  --episodes 1000 \
  --mass-range 50 90
```

### Fast Evolution with SAC
```bash
# Aggressive evolution
python train_population.py \
  --algorithm sac \
  --evolution-freq 1 \
  --keep-top 0.3 \
  --mutation-rate 0.2
```

## Future Enhancements

### Near Term
1. **Import Fix**: Resolve module import issues for full integration
2. **Tensorboard Integration**: Track population metrics in real-time
3. **Save/Load Populations**: Checkpoint entire populations
4. **Tournament Mode**: Run elimination tournaments

### Long Term
1. **Neural Architecture Search**: Evolve network architectures too
2. **Multi-Objective Optimization**: Balance win rate, spectacle, efficiency
3. **Cross-Population Battles**: Mix populations from different runs
4. **Online Learning**: Continuous population updates

## Technical Notes

### Import Issues
Currently there are circular import issues when running from the training directory. The workaround is to:
1. Use the demonstration script (`test_population_demo.py`)
2. Fix gym_env.py imports to handle different execution contexts
3. Use lazy imports in __init__ files

### Performance Considerations
- Each fighter needs ~100MB RAM for PPO model
- 16 fighters × 2 envs = 32 parallel environments
- Recommend 8+ CPU cores for large populations
- GPU acceleration helps but not required

### Diversity Metrics
Key indicators of healthy population:
- ELO Std Dev > 30 (variety in skill)
- ELO Range > 150 (clear skill hierarchy)
- No single fighter > 80% win rate (no dominant strategy)
- Win rate variance > 0.1 (different success levels)

## Conclusion

Population-based training represents a significant advancement for Atom Combat, enabling:
- **Diverse fighter strategies** through co-evolution
- **Robust performance** against multiple opponents
- **Natural selection** of effective techniques
- **Measurable progress** via ELO ratings

The implementation provides a solid foundation for training competitive AI fighters that can adapt to different opponents and develop unique fighting styles.

## Files Created

- `/training/src/trainers/population/__init__.py` - Package initialization
- `/training/src/trainers/population/elo_tracker.py` - ELO rating system
- `/training/src/trainers/population/population_trainer.py` - Main trainer class
- `/training/train_population.py` - CLI interface
- `/test_population_demo.py` - Working demonstration
- `/docs/POPULATION_TRAINING.md` - This documentation