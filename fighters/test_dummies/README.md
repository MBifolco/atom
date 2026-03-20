# Curriculum Test Dummies

This directory contains scripted opponents used by progressive training.

## Layout

```text
fighters/test_dummies/
└── atomic/
    ├── stationary_*.py
    ├── approach_*.py / flee_*.py / circle_*.py / shuttle_*.py
    ├── distance_keeper_*.py / stamina_*.py / charge_*.py
    └── stance-switch / retreater / oscillator variants
```

## How They Are Used

- **Level 1 (fundamentals):** stationary stance opponents
- **Level 2 (basic skills):** simple movement patterns
- **Level 3 (intermediate):** spacing and stamina patterns
- **Level 4 (advanced):** mixed strategy patterns
- **Level 5 (expert):** hand-authored expert fighters from `fighters/examples/`

## Quick Command

```bash
python atom_fight.py fighters/examples/boxer.py fighters/test_dummies/atomic/stationary_neutral.py --html dummy_match.html
```
