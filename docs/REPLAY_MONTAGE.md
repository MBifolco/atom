# Replay Recording & Montage Creation

Automatically record and create video montages from AI training sessions, showcasing the complete learning journey from beginner to expert.

## Overview

The replay system captures fights during training and uses **spectacle-based sampling** to automatically select the most interesting matches. This creates a compelling narrative showing:

- **Part 1: Curriculum Learning** - Watch the AI progress from struggling with stationary targets to mastering complex strategies
- **Part 2: Population Evolution** - See diverse fighting styles emerge through natural selection over 40+ generations

### What Makes This Special?

Instead of recording every fight (which would be thousands of hours), the system:

1. **Runs evaluation matches** at key training milestones
2. **Calculates spectacle scores** for entertainment value (close finishes, comebacks, drama)
3. **Samples bottom, middle, and top** spectacle fights from each stage
4. **Creates a 3-5 minute montage** showing the complete learning arc

## Quick Start

### 1. Record Replays During Training

```bash
python train_progressive.py \
  --mode complete \
  --timesteps 20000000 \
  --generations 40 \
  --use-vmap \
  --record-replays \
  --replay-frequency 2
```

**New flags:**
- `--record-replays` - Enable replay recording
- `--replay-frequency 2` - Record every 2nd generation (default: 5)

### 2. Create Montage Video

#### Option A: Fully Automated (Recommended)

```bash
# Install Playwright
pip install playwright
playwright install chromium

# Create montage
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_20251114_120000 \
  --speed 3.0
```

#### Option B: Manual (No Extra Dependencies)

```bash
# Render HTML files
python scripts/montage/render_replays.py \
  --run-dir outputs/progressive_20251114_120000

# Open outputs/progressive_TIMESTAMP/rendered_html/playlist.html
# Record with OBS, QuickTime, or other screen capture software
```

### 3. Watch Your Montage!

```bash
mpv outputs/progressive_20251114_120000/training_montage.mp4
```

## Recording Configuration

### Basic Options

```bash
python train_progressive.py \
  --record-replays \              # Enable recording
  --replay-frequency 5 \          # Record every 5th generation
  --mode complete \
  --timesteps 10000000 \
  --generations 20
```

### What Gets Recorded

#### Curriculum Training (Part 1)
Records 3 fights **after each level graduation**:

| Level | Description | Replays Recorded |
|-------|-------------|------------------|
| Level 1 | Fundamentals (Stationary targets) | 3 (bottom/middle/top spectacle) |
| Level 2 | Basic Skills (Simple movement) | 3 |
| Level 3 | Intermediate (Distance/stamina) | 3 |
| Level 4 | Advanced (Behavioral fighters) | 3 |
| Level 5 | Expert (Hardcoded fighters) | 3 |
| **Total** | | **15 curriculum replays** |

#### Population Training (Part 2)
Records 3 fights **every Nth generation** (configurable):

With `--replay-frequency 5` and `--generations 40`:
- Records at: Gen 5, 10, 15, 20, 25, 30, 35, 40
- Total: 8 generations × 3 replays = **24 population replays**

**Grand Total: ~39 replays** (15 curriculum + 24 population)

### Advanced Configuration

```bash
python train_progressive.py \
  --record-replays \
  --replay-frequency 2 \          # Record more often
  --keep-top 0.3 \                # More selective evolution (70% replaced)
  --mutation-rate 0.15 \          # Higher mutation = more diversity
  --evolution-frequency 1 \       # Evolve every generation
  --episodes-per-gen 3000         # More training per generation
```

**Tips:**
- **Lower `--replay-frequency`** = More replays (e.g., `2` records every other generation)
- **Higher evolution pressure** (`--keep-top 0.3`) = Faster skill progression in montage
- **More episodes** (`--episodes-per-gen 3000`) = Better skill between recordings

## Spectacle Scoring

Replays are ranked by **entertainment value**, not just who wins. The spectacle evaluator scores fights on:

### Spectacle Metrics

| Metric | Ideal Range | What It Measures |
|--------|-------------|------------------|
| **Duration** | 100-400 ticks | Not too quick, not endless |
| **Close Finish** | <20% HP remaining | Nail-biter endings |
| **Stamina Drama** | 10-30% time at low stamina | Exhaustion moments |
| **Comeback Potential** | 3+ lead changes | Back-and-forth action |
| **Positional Exchange** | 5-20% position swaps | Arena movement variety |
| **Pacing Variety** | Speed variance | Mix of fast/slow moments |
| **Collision Drama** | 8-25 impactful collisions | Not grinding, not passive |

### Sampling Strategy

From each stage (level/generation), the system:

1. **Runs evaluation matches** (e.g., 3 matches per opponent pair)
2. **Calculates spectacle score** for each match (0.0 to 1.0)
3. **Sorts by spectacle** (lowest to highest)
4. **Samples 3 fights**:
   - **Bottom spectacle** (~17th percentile) - Shows early struggles
   - **Middle spectacle** (50th percentile/median) - Typical performance
   - **Top spectacle** (~83rd percentile) - Best fights

This ensures the montage shows a **progression from struggle to mastery** at each stage!

## Montage Creation

### Automated Method (Playwright)

**Prerequisites:**
```bash
pip install playwright
playwright install chromium
sudo apt-get install ffmpeg  # Ubuntu/Debian
# OR: brew install ffmpeg     # macOS
```

**Create montage:**
```bash
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_20251114_120000 \
  --speed 3.0 \
  --fps 30 \
  --resolution 1920x1080
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | (required) | Training output directory |
| `--output` | `training_montage.mp4` | Output video filename |
| `--speed` | `3.0` | Playback speed multiplier (3x faster) |
| `--fps` | `30` | Video framerate |
| `--resolution` | `1920x1080` | Video resolution (WxH) |
| `--use-chrome` | (flag) | Use Chrome instead of Playwright |

**Process:**
1. ✅ Renders each replay to HTML with animation
2. ✅ Records HTML animations to video (headless browser)
3. ✅ Adds title cards ("Part 1: From Zero to Graduate", etc.)
4. ✅ Concatenates all clips into final montage
5. ✅ Outputs `training_montage.mp4`

**Expected duration:**
- Processing: ~2-5 minutes for 39 replays
- Final montage: ~3-5 minutes at 3x speed

### Manual Method (No Dependencies)

**Render HTML files:**
```bash
python scripts/montage/render_replays.py \
  --run-dir outputs/progressive_20251114_120000 \
  --speed 3.0
```

**Then:**
1. Open `outputs/progressive_TIMESTAMP/rendered_html/playlist.html` in browser
2. Click through each replay
3. Use screen recording software:
   - **OBS Studio** (Free, cross-platform)
   - **QuickTime** (macOS - File → New Screen Recording)
   - **SimpleScreenRecorder** (Linux)
   - **Xbox Game Bar** (Windows - Win+G)
4. Edit clips together in video editor
5. Export final montage

**Advantages:**
- No extra dependencies
- Full creative control (add music, transitions, commentary)
- Can cherry-pick specific fights

**Disadvantages:**
- Manual effort required
- Longer to create

## File Structure

After training with `--record-replays`:

```
outputs/progressive_20251114_120000/
├── curriculum/                                    # Curriculum training
│   ├── models/
│   │   └── curriculum_graduate.zip
│   └── logs/
├── population/                                    # Population training
│   ├── models/
│   │   ├── generation_0/
│   │   ├── generation_1/
│   │   └── ...
│   └── logs/
├── replays/                                       # 🎬 REPLAY DATA
│   ├── curriculum_level_1_fundamentals_bottom_spectacle_0.234.json.gz
│   ├── curriculum_level_1_fundamentals_middle_spectacle_0.567.json.gz
│   ├── curriculum_level_1_fundamentals_top_spectacle_0.891.json.gz
│   ├── curriculum_level_2_basic_skills_bottom_spectacle_0.345.json.gz
│   ├── ...
│   ├── population_gen_5_bottom_spectacle_0.456.json.gz
│   ├── population_gen_5_middle_spectacle_0.678.json.gz
│   ├── population_gen_5_top_spectacle_0.823.json.gz
│   └── ...
├── replay_index.json                              # 📋 INDEX OF ALL REPLAYS
├── rendered_html/                                 # HTML replays (if rendered)
│   ├── playlist.html
│   ├── curriculum_01_level_1_bottom.html
│   └── ...
├── montage_temp/                                  # Temp video clips
│   ├── title_curriculum.mp4
│   ├── curriculum_1.mp4
│   └── ...
└── training_montage.mp4                           # 🎥 FINAL MONTAGE!
```

### Replay Index Format

`replay_index.json` contains metadata for all recorded replays:

```json
{
  "total_replays": 39,
  "replays": [
    {
      "stage": "curriculum_level_1_fundamentals",
      "stage_type": "curriculum",
      "spectacle_score": 0.891,
      "spectacle_rank": "top",
      "fighter_a": "AI_Fighter",
      "fighter_b": "stationary_neutral",
      "winner": "AI_Fighter",
      "notes": "Sampled top spectacle from 12 matches"
    },
    {
      "stage": "population_gen_5",
      "stage_type": "population",
      "spectacle_score": 0.823,
      "spectacle_rank": "top",
      "fighter_a": "sleepy_boyd",
      "fighter_b": "compassionate_payne",
      "winner": "sleepy_boyd",
      "notes": "Sampled top spectacle from 28 matches"
    }
  ]
}
```

### Replay Data Format

Each `.json.gz` file contains complete fight telemetry:

```json
{
  "version": "1.0",
  "timestamp": "2025-11-14T12:34:56",
  "result": {
    "winner": "AI_Fighter",
    "total_ticks": 187,
    "final_hp_a": 87.3,
    "final_hp_b": 0.0
  },
  "telemetry": {
    "fighter_a_name": "AI_Fighter",
    "fighter_b_name": "stationary_neutral",
    "ticks": [
      {
        "tick": 0,
        "fighter_a": {
          "position": 3.0,
          "velocity": 0.0,
          "hp": 100.0,
          "stamina": 100.0,
          "stance": "neutral"
        },
        "fighter_b": { ... },
        "action_a": {
          "acceleration": 0.5,
          "stance": "extended"
        },
        "action_b": { ... }
      }
    ]
  },
  "events": [ ... ],
  "metadata": {
    "stage": "curriculum_level_1_fundamentals",
    "spectacle_score": 0.891,
    "spectacle_breakdown": {
      "duration": 0.95,
      "close_finish": 0.87,
      "stamina_drama": 0.92,
      ...
    }
  }
}
```

## Examples

### Example 1: Quick Test Run

```bash
# Short training with frequent recording
python train_progressive.py \
  --mode complete \
  --timesteps 1000000 \
  --generations 10 \
  --record-replays \
  --replay-frequency 2 \
  --use-vmap

# Creates ~20 replays (5 curriculum + 15 population)
# Generate montage
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_TIMESTAMP \
  --speed 5.0
```

### Example 2: Production Run

```bash
# Full training with selective recording
python train_progressive.py \
  --mode complete \
  --timesteps 20000000 \
  --generations 40 \
  --keep-top 0.3 \
  --mutation-rate 0.15 \
  --evolution-frequency 1 \
  --episodes-per-gen 3000 \
  --record-replays \
  --replay-frequency 5 \
  --use-vmap

# Creates ~39 replays
# High-quality 1080p montage at 3x speed
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_TIMESTAMP \
  --speed 3.0 \
  --resolution 1920x1080
```

### Example 3: 4K Montage for YouTube

```bash
# After training, create 4K montage
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_20251114_120000 \
  --speed 3.0 \
  --resolution 3840x2160 \
  --fps 60 \
  --output youtube_montage_4k.mp4
```

### Example 4: Resume Training with Recording

```bash
# Resume from generation 20, continue recording
python scripts/training/resume_population_training.py \
  --checkpoint-dir outputs/progressive_20251114_120000 \
  --start-gen 20 \
  --total-gens 60 \
  --record-replays \
  --replay-frequency 5 \
  --use-vmap

# Replays append to existing replay_index.json
# Creates montage with all generations (0-60)
python scripts/montage/create_montage.py \
  --run-dir outputs/progressive_20251114_120000 \
  --speed 3.0
```

## Customization

### Playback Speed

```bash
# Very fast (5x)
python scripts/montage/create_montage.py --run-dir ... --speed 5.0
# Result: ~2 minute montage, fast-paced

# Moderate (3x) - Recommended
python scripts/montage/create_montage.py --run-dir ... --speed 3.0
# Result: ~3.5 minute montage, clear action

# Slow (1x) - Real-time
python scripts/montage/create_montage.py --run-dir ... --speed 1.0
# Result: ~10+ minute montage, every detail visible
```

### Recording Frequency

```bash
# Record every generation (lots of replays)
python train_progressive.py --record-replays --replay-frequency 1

# Record every 10 generations (fewer replays)
python train_progressive.py --record-replays --replay-frequency 10

# Curriculum only (no population recording)
python train_progressive.py --mode curriculum --record-replays
```

### Resolution Presets

```bash
# 4K (YouTube, high quality)
--resolution 3840x2160

# 1080p (Standard HD, recommended)
--resolution 1920x1080

# 720p (Smaller file size)
--resolution 1280x720

# Vertical video (TikTok, Instagram Stories)
--resolution 1080x1920
```

## Sharing Your Montage

### YouTube Upload Settings

**Recommended:**
- Resolution: 1920x1080 (1080p)
- Framerate: 30 fps
- Speed: 3-5x
- Duration: 3-5 minutes

**Title ideas:**
- "AI Learns to Fight: Complete Training Journey (Curriculum → Evolution)"
- "From Zero to Expert: AI Population-Based Training Montage"
- "40 Generations of AI Evolution - Atom Combat"

**Description template:**
```
AI learns to fight through:
1. Curriculum Learning (5 difficulty levels)
2. Population-Based Evolution (40 generations)

Replay system automatically records bottom/middle/top spectacle
fights showing the full learning journey from beginner to expert.

Training parameters:
- Algorithm: PPO
- Timesteps: 20M (curriculum)
- Generations: 40
- Population: 8 fighters
- Evolution: Keep top 30%, 15% mutation, evolve every generation

Code: https://github.com/...
Method: Population-Based Training with spectacle-based sampling
```

### File Size Estimates

| Resolution | FPS | Duration | Approximate Size |
|------------|-----|----------|------------------|
| 720p | 30 | 3 min | 20-40 MB |
| 1080p | 30 | 3 min | 50-100 MB |
| 1080p | 60 | 3 min | 80-150 MB |
| 4K | 30 | 3 min | 200-400 MB |
| 4K | 60 | 3 min | 350-700 MB |

## Troubleshooting

### "Replay index not found"

**Problem:** No `replay_index.json` file

**Solution:**
```bash
# Make sure you trained with --record-replays
python train_progressive.py --record-replays ...

# Check if file exists
ls outputs/progressive_TIMESTAMP/replay_index.json
```

### "Playwright not found"

**Problem:** Automated montage creation requires Playwright

**Solution:**
```bash
# Install Playwright
pip install playwright
playwright install chromium

# Or use manual method
python scripts/montage/render_replays.py --run-dir ...
```

### "ffmpeg not found"

**Problem:** Video creation requires ffmpeg

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

### Video recording fails

**Problem:** Headless browser issues

**Solutions:**
```bash
# 1. Try manual method instead
python scripts/montage/render_replays.py --run-dir ...

# 2. Check Chrome/Chromium availability
python scripts/montage/create_montage.py --run-dir ... --use-chrome

# 3. Reduce resolution
python scripts/montage/create_montage.py --run-dir ... --resolution 1280x720
```

### Replays look choppy

**Problem:** Low FPS or too fast playback

**Solutions:**
```bash
# Increase framerate
python scripts/montage/create_montage.py --run-dir ... --fps 60

# Reduce playback speed
python scripts/montage/create_montage.py --run-dir ... --speed 2.0

# Or use realtime speed for smooth viewing
python scripts/montage/create_montage.py --run-dir ... --speed 1.0
```

### Out of disk space

**Problem:** Temp files taking space

**Solution:**
```bash
# Clean up after montage creation
rm -rf outputs/progressive_TIMESTAMP/montage_temp/

# Compressed replays are small (~500KB each)
# HTML files are ~1MB each
# Video clips are ~10-50MB each (temp)
# Final montage is ~50-100MB
```

### Not enough replays recorded

**Problem:** `--replay-frequency` too high or training stopped early

**Solutions:**
```bash
# Record more often
python train_progressive.py --record-replays --replay-frequency 1

# Or resume training
python scripts/training/resume_population_training.py \
  --checkpoint-dir outputs/... \
  --start-gen 20 \
  --total-gens 40 \
  --record-replays \
  --replay-frequency 2
```

## Performance Notes

### Recording Overhead

Replay recording has **minimal impact** on training:
- Evaluation matches run **after** training episodes
- ~30-60 seconds per generation (vs hours of training)
- Replays compressed to ~500KB each

**Total overhead:** <1% of training time

### Disk Usage

Example for 40-generation run with `--replay-frequency 5`:
- Replays: 39 files × 500KB = ~20 MB
- Index: 1 file × 100KB = ~0.1 MB
- HTML (if rendered): 39 files × 1MB = ~40 MB
- Temp videos (deleted after): ~2 GB
- Final montage: ~50-100 MB

**Total persistent:** ~60-160 MB

### Montage Creation Time

With Playwright (automated):
- Rendering: ~2-5 seconds per replay
- Recording: ~5-10 seconds per replay
- Concatenation: ~10-30 seconds
- **Total: ~5-10 minutes** for 39 replays

Manual method:
- Depends on your recording speed
- **~20-60 minutes** including editing

## Advanced Usage

### Custom Spectacle Weights

The spectacle evaluator can be customized by modifying `src/evaluator/spectacle_evaluator.py`:

```python
# Default weights (all equal)
self.weights = {
    "duration": 1.0,
    "close_finish": 1.0,
    "stamina_drama": 1.0,
    "comeback_potential": 1.0,
    "positional_exchange": 1.0,
    "pacing_variety": 1.0,
    "collision_drama": 1.0
}

# Prefer close finishes and comebacks
self.weights = {
    "duration": 0.5,
    "close_finish": 2.0,      # ← Double weight
    "stamina_drama": 1.0,
    "comeback_potential": 2.0, # ← Double weight
    "positional_exchange": 0.5,
    "pacing_variety": 0.5,
    "collision_drama": 1.0
}
```

### Programmatic Access

```python
from src.telemetry.replay_store import load_replay
from src.evaluator.spectacle_evaluator import SpectacleEvaluator
import json

# Load replay index
with open('outputs/progressive_TIMESTAMP/replay_index.json') as f:
    index = json.load(f)

# Find highest spectacle replay
best_replay = max(index['replays'], key=lambda r: r['spectacle_score'])
print(f"Best fight: {best_replay['fighter_a']} vs {best_replay['fighter_b']}")
print(f"Spectacle: {best_replay['spectacle_score']:.3f}")

# Load full replay data
replay_path = f"outputs/.../replays/{best_replay['stage']}_{best_replay['spectacle_rank']}_*.json.gz"
replay_data = load_replay(replay_path)

# Analyze
print(f"Duration: {replay_data['result']['total_ticks']} ticks")
print(f"Winner: {replay_data['result']['winner']}")
```

## FAQ

**Q: Can I record replays from an existing training run?**

A: No, replays must be recorded during training with `--record-replays`. However, you can resume training with recording enabled.

**Q: How much does recording slow down training?**

A: <1% overhead. Evaluation matches run after training episodes, taking ~30-60 seconds per generation.

**Q: Can I change which fights get recorded?**

A: Yes, modify the sampling in `src/training/replay_recorder.py`. Current strategy samples bottom/middle/top spectacle, but you could sample differently (e.g., only top, random selection, etc.).

**Q: Can I add music to the montage?**

A: Use the manual method and edit in a video editor (iMovie, DaVinci Resolve, Premiere, etc.), or use ffmpeg to add audio track:

```bash
ffmpeg -i training_montage.mp4 -i music.mp3 \
  -c:v copy -c:a aac -shortest \
  montage_with_music.mp4
```

**Q: Can I export individual replays as videos?**

A: Yes! The `scripts/montage/create_montage.py` script creates individual clips in `montage_temp/`. Keep them instead of deleting:

```bash
# After creating montage
cp outputs/.../montage_temp/*.mp4 my_replays/
```

**Q: What's the difference between curriculum and population replays?**

A:
- **Curriculum**: Shows progression through difficulty levels (vs pre-scripted opponents)
- **Population**: Shows evolution of diverse strategies (AI vs AI self-play)

Both are included in the montage to show the complete learning journey.

---

## See Also

- [MONTAGE_CREATION.md](MONTAGE_CREATION.md) - Quick start guide
- [POPULATION_TRAINING.md](POPULATION_TRAINING.md) - Population training details
- [Spectacle Evaluator](../src/evaluator/spectacle_evaluator.py) - Scoring implementation
- [Replay Recorder](../src/training/replay_recorder.py) - Recording implementation
