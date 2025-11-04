# Training Logs

This directory contains detailed training logs.

## Log Format

Each training session creates a timestamped log file:
```
{model_name}_training_{YYYYMMDD_HHMMSS}.log
```

Example:
```
my_fighter_training_20251103_230148.log
```

## Log Contents

Each log includes:

- **Training start info**: Opponents, number of environments, configuration
- **Episode details**: For every episode across all environments
  - Episode number
  - Environment ID
  - Opponent name
  - Reward received
  - Episode length (ticks)
  - Final HP for both fighters
  - Damage dealt/taken
  - Win/Loss result

## Example Log Entry

```
2025-11-03 23:06:13,591 - DEBUG - Episode 2253 | Env 0 | Opponent: bumbler | Reward: 392.0 | Length: 86 ticks
2025-11-03 23:06:13,591 - DEBUG -   Final HP: Fighter=2.8, Opponent=0.0
2025-11-03 23:06:13,591 - DEBUG -   Damage: Dealt=101.3, Taken=90.9
2025-11-03 23:06:13,591 - DEBUG -   Result: WIN
```

## Analyzing Logs

```bash
# Count wins vs losses
grep "Result: WIN" my_fighter_training_*.log | wc -l
grep "Result: LOSS" my_fighter_training_*.log | wc -l

# Find episodes against specific opponent
grep "Opponent: training_dummy" my_fighter_training_*.log

# Check average rewards
grep "Reward:" my_fighter_training_*.log | awk '{sum+=$9; count++} END {print sum/count}'

# Find longest episodes
grep "Length:" my_fighter_training_*.log | awk '{print $11}' | sort -n | tail -20
```

## Retention

Logs can be large (1-2MB per training session). Feel free to delete old logs to save space.
