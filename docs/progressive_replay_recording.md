# Progressive Replay Recording System

## Overview
The progressive replay recording system captures training fights throughout the learning process to show progression from novice to expert, replacing the old spectacle-based sampling.

## Components

### 1. ProgressiveReplayRecorder (`src/training/progressive_replay_recorder.py`)
- Records fights at strategic intervals during training
- Recording strategy:
  - **Early phase (0-20%)**: Every 10 episodes
  - **Mid phase (20-80%)**: Every 50 episodes
  - **Late phase (80-100%)**: Every 100 episodes
  - Always records first and last episodes

### 2. CurriculumCallback Integration (`src/training/trainers/curriculum_trainer.py`)
- `_on_step()`: Checks if episode should be recorded
- `_record_evaluation_replay()`: Runs evaluation match with telemetry
- Records when `should_record()` returns True

### 3. HTML Montage Generator (`create_html_montage.py`)
- Loads progressive replays from training run
- Creates single HTML file with embedded replay data
- Interactive playback controls (play/pause, speed, navigation)
- Shows progression metrics (level, episode, win rate)

## Usage

### Training with Recording
```bash
# MUST use --record-replays flag
python train_progressive.py --record-replays --timesteps 1000000
```

### Generate HTML Montage
```bash
# After training completes
python create_html_montage.py --run-dir outputs/progressive_YYYYMMDD_HHMMSS
```

### Check Recording Status
```bash
python test_recording_simple.py
```

## Current Issues

### Recording Not Triggering
The progressive recording is not saving files during training because:
1. The `_record_evaluation_replay` method may not be called properly
2. The episode detection in `_on_step` might not trigger correctly
3. The model prediction might fail during evaluation

### Workaround
Currently using `create_html_montage_from_existing.py` to generate montages from old spectacle-based replays with simulated progressive metadata.

## Testing

Run progressive replay tests:
```bash
python -m pytest tests/test_progressive_replay_integration.py -v
```

All 6 tests should pass:
- `test_callback_initializes_recorder`
- `test_callback_checks_recording_intervals`
- `test_record_evaluation_replay_runs`
- `test_episode_detection_triggers_recording`
- `test_recording_handles_errors_gracefully`
- `test_progressive_index_saved_on_completion`

## Next Steps

1. Debug why `_record_evaluation_replay` isn't being called during actual training
2. Ensure episode detection triggers recording at proper intervals
3. Verify model can predict actions during evaluation matches
4. Test with full training run to generate real progressive replays
5. Remove old spectacle-based code once confirmed working