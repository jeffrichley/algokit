# BFS Timing Configuration Guide

## Overview
The BFS scene now uses a centralized timing configuration system that allows you to easily adjust animation speeds and durations without modifying code.

## Configuration File
**Location**: `data/examples/scenarios/bfs_timing_config.yaml`

## Quick Start

### 1. Change Timing Mode
```python
# In your BFS scene
scene = BreadthFirstSearchScene()
scene.set_timing_mode("quick_demo")  # Very fast
scene.set_timing_mode("development") # Fast
scene.set_timing_mode("cinematic")   # Normal
```

### 2. Reduce Video Length (Biggest Impact)
Edit `bfs_timing_config.yaml`:
```yaml
event_limits:
  max_events_displayed: 20  # Instead of 999
  skip_non_essential: true  # Skip events not on path
```

### 3. Adjust Specific Animations
Edit `bfs_timing_config.yaml`:
```yaml
base_timings:
  title_write: 0.5        # Faster title (was 1.8)
  confetti_effects: 0.5   # Faster celebration (was 2.0)

wait_times:
  final_study_pause: 1.0  # Shorter final pause (was 4.0)
```

### 4. Create Custom Speed Mode
Add to `bfs_timing_config.yaml`:
```yaml
speed_multipliers:
  ultra_fast:
    setup: 20.0
    bfs_events: 40.0
    path_drawing: 30.0
    celebrations: 15.0
    educational: 10.0
    waits: 50.0
```

## Available Timing Modes

| Mode | Description | Speed Multiplier |
|------|-------------|------------------|
| `cinematic` | Normal viewing speeds | 1.0x |
| `development` | Fast development speeds | 4-10x faster |
| `quick_demo` | Very fast demo speeds | 8-20x faster |

## Key Settings for Video Length Reduction

### High Impact Changes:
1. **Reduce BFS Events**: `max_events_displayed: 15-20`
2. **Skip Non-Essential Events**: `skip_non_essential: true`
3. **Reduce Wait Times**: Lower all `wait_times` values by 50-75%
4. **Use Quick Demo Mode**: Set mode to `quick_demo`

### Medium Impact Changes:
1. **Faster Celebrations**: Reduce `confetti_effects` and `goal_text_reveal`
2. **Streamline Path Reconstruction**: Reduce `tracer_movement` and `final_study_pause`
3. **Faster Setup**: Reduce `title_write`, `subtitle_write`, `grid_wave`

## Example: Ultra-Fast Configuration
```yaml
event_limits:
  max_events_displayed: 15
  skip_non_essential: true

base_timings:
  title_write: 0.3
  confetti_effects: 0.2
  tracer_movement: 1.0
  goal_text_reveal: 0.3

wait_times:
  initial_setup: 0.5
  final_study_pause: 0.5
  educational_pause: 0.5

speed_multipliers:
  ultra_fast:
    setup: 20.0
    bfs_events: 40.0
    path_drawing: 30.0
    celebrations: 20.0
    educational: 15.0
    waits: 50.0
```

## Testing Changes
```bash
# Test with demo script
uv run python examples/bfs_timing_demo.py

# Test with actual scene
uv run manim -pql src/agloviz/scenes/planning/bfs/render_bfs_scene.py RenderBfsScene
```

## Tips
- Start with reducing `max_events_displayed` for biggest impact
- Use `quick_demo` mode as starting point for fast videos
- Adjust `wait_times` values to reduce pauses
- Test changes incrementally to find optimal balance
