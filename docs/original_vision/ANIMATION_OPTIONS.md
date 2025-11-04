# Atom Combat - Animation Options

## Available Animation Formats

### 1. ✅ ASCII Terminal Animation (Already Implemented)

**Location:** `src/renderer/ascii_renderer.py`

**Features:**
- Real-time terminal playback
- Works anywhere with Python
- Configurable playback speed
- Shows HP/stamina bars as text
- Stance indicators (●▶◀■)
- Collision highlights (💥)

**Usage:**
```python
from src.renderer import AsciiRenderer

renderer = AsciiRenderer(arena_width=config.arena_width)
renderer.play_replay(
    result.telemetry,
    result,
    spectacle_score=score,
    playback_speed=5.0,
    skip_ticks=5  # Show every 5th tick for highlights
)
```

**Pros:**
- ✅ Zero dependencies
- ✅ Fast and lightweight
- ✅ Good for debugging/testing
- ✅ Works over SSH

**Cons:**
- ❌ Limited visual appeal
- ❌ Hard to share
- ❌ Not smooth (discrete ticks)

---

### 2. ✅ HTML5 Canvas Animation (Just Implemented!)

**Location:** `src/renderer/html_renderer.py`

**Features:**
- Standalone .html file (no server needed!)
- Smooth 60fps animation
- Interactive controls (play/pause/step/speed)
- Real-time stats display
- Visual stance shapes (circle/triangle/square/hexagon)
- Velocity arrows
- HP/Stamina bars
- Collision highlighting
- **Winner/Loser end screen with full match metrics**

**Usage:**
```python
from src.renderer import HtmlRenderer

renderer = HtmlRenderer()
output_file = renderer.generate_replay_html(
    result.telemetry,
    result,
    "replay.html",
    spectacle_score=score,
    playback_speed=1.0
)
# Open replay.html in any browser!
```

**Quick Generate:**
```bash
python generate_html_replay.py
# Opens replay_viewer.html
```

**Pros:**
- ✅ Easy to share (single .html file)
- ✅ Beautiful, smooth animations
- ✅ Interactive controls
- ✅ Works in any browser
- ✅ No dependencies
- ✅ Can be embedded in websites
- ✅ Shows all match stats

**Cons:**
- ❌ Requires browser to view

**Best for:** Sharing replays, tournament broadcasts, web embedding

---

### 3. 🔧 Matplotlib Animation (Can Implement)

**Features:**
- Line plots showing fighter positions over time
- Can save as MP4/GIF
- Multiple subplots (position, HP, stamina)
- Good for analysis

**Example Implementation:**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_matplotlib_animation(telemetry, output_file="replay.mp4"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # Position plot
    # HP plot
    # Stamina plot

    anim = animation.FuncAnimation(fig, update, frames=len(ticks), interval=50)
    anim.save(output_file, writer='ffmpeg')
```

**Pros:**
- ✅ Can save as video files
- ✅ Good for analysis/presentations
- ✅ Multiple views at once
- ✅ Python-native

**Cons:**
- ❌ Requires matplotlib + ffmpeg
- ❌ Less interactive than HTML
- ❌ Larger file sizes

**Best for:** Research papers, video presentations, data analysis

---

### 4. 🔧 Animated GIF Generator (Can Implement)

**Features:**
- Frame-by-frame rendering using PIL/Pillow
- Embeddable anywhere (Discord, GitHub, etc.)
- No playback controls needed

**Example Implementation:**
```python
from PIL import Image, ImageDraw

def generate_gif(telemetry, output_file="replay.gif"):
    frames = []

    for tick in telemetry["ticks"]:
        # Create frame
        img = Image.new('RGB', (800, 400), color='black')
        draw = ImageDraw.Draw(img)

        # Draw arena, fighters, stats
        # ...

        frames.append(img)

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=84,  # ms per frame (matches dt)
        loop=0
    )
```

**Pros:**
- ✅ Embeddable everywhere
- ✅ No player needed
- ✅ Small file sizes (for short matches)
- ✅ Auto-plays in most viewers

**Cons:**
- ❌ No playback controls
- ❌ Large files for long matches
- ❌ Limited color palette (256 colors)

**Best for:** Sharing on social media, GitHub READMEs, Discord

---

### 5. 🔧 Pygame Desktop Viewer (Can Implement)

**Features:**
- Native desktop application
- Real-time rendering
- Keyboard controls
- Better graphics than ASCII
- Can add sound effects

**Example Implementation:**
```python
import pygame

def run_pygame_viewer(telemetry):
    pygame.init()
    screen = pygame.display.set_mode((1200, 600))

    running = True
    tick = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill((20, 30, 50))

        # Draw fighters, arena, stats
        # ...

        pygame.display.flip()
        pygame.time.wait(int(dt * 1000))
```

**Pros:**
- ✅ Smooth real-time playback
- ✅ Good graphics
- ✅ Can add sound/particle effects
- ✅ Keyboard/mouse controls

**Cons:**
- ❌ Requires pygame dependency
- ❌ Not easily shareable
- ❌ Desktop-only

**Best for:** Tournament viewing, live events, polish

---

### 6. 🔧 Plotly Interactive Visualization (Can Implement)

**Features:**
- Interactive web-based plots
- Zoom/pan functionality
- Multiple synchronized charts
- Hover tooltips

**Example:**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_plotly_viz(telemetry):
    fig = make_subplots(rows=3, cols=1,
        subplot_titles=('Position', 'HP', 'Stamina'))

    # Add traces for both fighters
    # ...

    fig.write_html("replay_interactive.html")
```

**Pros:**
- ✅ Highly interactive
- ✅ Professional-looking
- ✅ Multiple synchronized views
- ✅ Good for analysis

**Cons:**
- ❌ Requires plotly dependency
- ❌ Larger file sizes
- ❌ More complex than simple animation

**Best for:** Data analysis, research, presentations

---

## Recommendation Summary

| Use Case | Best Option | Why |
|----------|-------------|-----|
| **Sharing replays** | HTML5 Canvas | Single file, beautiful, interactive |
| **Quick testing** | ASCII Terminal | Fast, no setup |
| **Social media** | Animated GIF | Embeds everywhere |
| **Tournament broadcast** | Pygame Desktop | Polished, real-time |
| **Research/Analysis** | Matplotlib/Plotly | Multiple views, data-focused |
| **Web embedding** | HTML5 Canvas | Clean integration |

## Current Status

✅ **Implemented:**
1. ASCII Terminal Animation
2. HTML5 Canvas Animation

🔧 **Can Implement on Request:**
3. Matplotlib Animation
4. Animated GIF Generator
5. Pygame Desktop Viewer
6. Plotly Interactive Visualization

---

## Quick Start Examples

### Generate HTML Replay (Recommended)
```bash
python generate_html_replay.py
# Opens replay_viewer.html in browser
```

### ASCII Playback
```python
from src.renderer import AsciiRenderer
from src.orchestrator import MatchOrchestrator

# ... run match ...

renderer = AsciiRenderer()
renderer.play_replay(result.telemetry, result, playback_speed=2.0)
```

### Custom HTML with Specific Match
```python
from src.renderer import HtmlRenderer
from src.telemetry import load_replay

# Load saved replay
replay_data = load_replay("replays/match_001.json.gz")

renderer = HtmlRenderer()
renderer.generate_replay_html(
    replay_data["telemetry"],
    replay_data["result"],
    "custom_replay.html"
)
```

---

## HTML5 Canvas Features

The HTML5 renderer includes:

**Visual Elements:**
- 🔴 Fighter A (red circle/shapes)
- 🟢 Fighter B (green circle/shapes)
- ➡️ Velocity arrows
- 📊 HP bars (red)
- 🔋 Stamina bars (green)
- 💥 Collision effects

**Stance Shapes:**
- ⚪ Neutral → Circle
- ▶️ Extended → Triangle (pointing forward)
- ⬛ Retracted → Square
- ⬢ Defending → Hexagon

**Controls:**
- ▶️ Play/Pause button
- 🔄 Restart button
- ⏭️ Step forward button
- 🎚️ Speed slider (0.25x - 5x)

**Stats Display:**
- Fighter stats (HP, stamina, velocity, position, mass)
- Match info (winner, duration, collisions)
- Spectacle score (if provided)

**Winner/Loser End Screen:**
- 🏆 Winner announcement with colored banner
- Winner vs Defeated comparison boxes
- Final HP percentages
- 📊 Match statistics:
  - Duration (ticks and seconds)
  - Total collisions (count and percentage)
  - Damage dealt to loser
  - Damage taken by winner
- ⭐ Spectacle score with color-coded rating
- Star rating (⭐⭐ to ⭐⭐⭐⭐⭐)
- Top 3 spectacle metrics with progress bars
- Beautiful gradient background

---

## Future Enhancements

Possible additions for HTML5 renderer:
- [ ] Seek bar for jumping to specific ticks
- [ ] Slow-motion for collision moments
- [ ] Heatmap of position density
- [ ] Damage history graph
- [ ] Export current frame as image
- [ ] Multiple camera angles/zoom levels
- [ ] Particle effects for collisions
- [ ] Sound effects (whoosh, impact, etc.)
- [ ] Animated confetti/celebration on winner screen
- [ ] Share button to export match summary
