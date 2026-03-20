# Creating Training Montages

This guide explains how to create video montages from recorded training replays.

## Quick Start

### 1. Train with Replay Recording

```bash
# Train with replay recording enabled
python train_progressive.py \
  --mode complete \
  --timesteps 500000 \
  --generations 20 \
  --record-replays \
  --replay-frequency 5 \
  --use-vmap
```

This creates:
- `outputs/progressive_TIMESTAMP/replays/` - Compressed replay files
- `outputs/progressive_TIMESTAMP/replay_index.json` - Index of all replays

### 2. Create Montage Video

#### Option A: Automated (Recommended)

Install Playwright for automated video recording:

```bash
# Install dependencies
pip install playwright
playwright install chromium

# Create montage
python create_montage.py --run-dir outputs/progressive_20251114_120000 --speed 3.0
```

This automatically:
1. Renders replays to HTML
2. Records HTML animations to video using headless browser
3. Creates title cards
4. Concatenates everything into final montage

#### Option B: Manual (Simpler, no extra dependencies)

```bash
# Just render HTML files
python render_replays.py --run-dir outputs/progressive_20251114_120000
```

Then manually:
1. Open each HTML file in browser
2. Use screen recording software (OBS, QuickTime, etc.)
3. Edit together in video editor

## What Gets Recorded

### Curriculum Training (Part 1: "From Zero to Graduate")

Records 3 fights per level after graduation:
- **Bottom spectacle** - Early struggles, showing the learning process
- **Middle spectacle** - Developing competence
- **Top spectacle** - Mastery-level fights

Example for 5 levels: 15 curriculum replays

### Population Training (Part 2: "Natural Selection")

Records every 5th generation (configurable):
- **Bottom spectacle** - Awkward early fights
- **Middle spectacle** - Strategic development
- **Top spectacle** - Elite evolved combat

Example for 20 generations @ frequency=5: 12 population replays (gen 5, 10, 15, 20)

### Total Montage Length

With default settings (3x speed, 250 max ticks):
- Average fight: ~7 seconds at 3x speed
- 27 replays × 7 seconds = ~3 minutes of action
- Plus title cards: ~3.5 minute final montage

## Customization

### Playback Speed

```bash
# 5x speed for shorter montage
python create_montage.py --run-dir outputs/progressive_20251114_120000 --speed 5.0

# 1x speed for full detail
python create_montage.py --run-dir outputs/progressive_20251114_120000 --speed 1.0
```

### Resolution

```bash
# 4K
python create_montage.py --run-dir outputs/progressive_20251114_120000 --resolution 3840x2160

# 720p (smaller file)
python create_montage.py --run-dir outputs/progressive_20251114_120000 --resolution 1280x720
```

### Recording Frequency

```bash
# Record every generation (more replays)
python train_progressive.py --record-replays --replay-frequency 1

# Record every 10 generations (fewer replays)
python train_progressive.py --record-replays --replay-frequency 10
```

## Replay Data Structure

Each replay contains:
- Full tick-by-tick telemetry (positions, HP, stamina, actions)
- Spectacle scores (drama, pacing, close finishes, etc.)
- Metadata (stage, fighters, generation, etc.)

Files are saved as compressed JSON (`.json.gz`) for efficiency.

## Spectacle Scoring

Replays are ranked by entertainment value using:
- **Duration** - Ideal fight length (100-400 ticks)
- **Close finish** - Nail-biter endings
- **Stamina drama** - Exhaustion moments
- **Comeback potential** - Lead changes
- **Positional exchange** - Arena movement variety
- **Pacing variety** - Mix of speeds
- **Collision drama** - Impactful exchanges

This ensures the montage shows the most interesting fights!

## Troubleshooting

### "Playwright not found"
```bash
pip install playwright
playwright install chromium
```

### "ffmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Video recording fails
Try manual method:
```bash
python render_replays.py --run-dir outputs/progressive_20251114_120000
```

Then use OBS or similar to record the HTML animations.

### Out of disk space
Replays are compressed but a full 20-generation run creates:
- ~27 replay files × ~500KB each = ~13MB replays
- Temp video files during creation: ~500MB
- Final montage: ~50-100MB

Clean up temp files after:
```bash
rm -rf outputs/progressive_TIMESTAMP/montage_temp/
```

## Advanced: Custom Montage Edits

For full creative control:

1. Render individual HTML files:
```bash
python render_replays.py --run-dir outputs/progressive_20251114_120000
```

2. Open HTML files in browser and record with screen capture

3. Edit in your favorite video editor:
   - Add music
   - Add commentary
   - Adjust pacing
   - Add transitions
   - Color grading

4. Export final montage

## Sharing Your Montage

Recommended settings for YouTube:
- Resolution: 1920x1080 (1080p)
- Framerate: 30 fps
- Speed: 3-5x
- Include title cards explaining what's happening

Add description:
```
AI learns to fight in Atom Combat through:
1. Curriculum Learning (5 difficulty levels)
2. Population-Based Evolution (20 generations)

Replay system automatically records bottom/middle/top spectacle
fights showing the full learning journey from beginner to expert.

Code: https://github.com/...
```
