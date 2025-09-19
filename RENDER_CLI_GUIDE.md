# 🎬 AlgoKit Render CLI Guide

A comprehensive CLI for rendering algorithm visualizations with Manim.

> **✅ CLI Refactored**: The CLI has been updated to eliminate the confusing double `render` command structure. 
> Use `algokit render bfs` instead of `algokit render render bfs`. See [CLI_REFACTOR_GUIDE.md](CLI_REFACTOR_GUIDE.md) for details.

## 🚀 Quick Start

```bash
# List available scenarios
algokit render scenarios

# List available algorithms  
algokit render algorithms

# Quick render for testing
algokit render quick

# Demo render (high quality)
algokit render demo

# Custom render
algokit render bfs --scenario "Tiny Test" --quality high
```

## 📋 Commands

### `algokit render scenarios`
List all available scenarios with optional details.

```bash
# Basic list
algokit render scenarios

# With details (grid size, description)
algokit render scenarios --details
```

### `algokit render algorithms`
List all available algorithms and their descriptions.

```bash
algokit render algorithms
```

### `algokit render render`
Main render command with full customization.

```bash
# Basic render with default scenario
algokit render bfs

# With specific scenario
algokit render bfs --scenario "Tiny Test"
algokit render bfs --scenario "Harbor Flood Small"

# Quality options: low, medium, high, ultra
algokit render bfs --quality high

# Timing modes: cinematic, development, quick_demo
algokit render bfs --timing development

# Limit events for faster rendering
algokit render bfs --max-events 20

# Custom output directory
algokit render bfs --output ./my_videos/

# Preview mode (low quality, fast)
algokit render bfs --preview
```

### `algokit render quick`
Quick render for testing (low quality, fast timing, limited events).

```bash
# Default quick render
algokit render quick

# Quick render with specific scenario
algokit render quick --scenario "Tiny Test"
```

### `algokit render demo`
High-quality demo render with cinematic timing.

```bash
algokit render demo
```

## 🎯 Examples

### Development Workflow
```bash
# Quick test during development
algokit render quick

# Medium quality for review
algokit render bfs --quality medium --timing development

# High quality for final output
algokit render bfs --quality high --timing cinematic
```

### Scenario Testing
```bash
# Test all scenarios quickly
algokit render quick --scenario "Tiny Test"
algokit render quick --scenario "Harbor Flood Small"  
algokit render quick --scenario "Harbor Storm Medium"
```

### Performance Optimization
```bash
# Very fast rendering (10 events only)
algokit render bfs --timing quick_demo --max-events 10

# Balanced quality and speed
algokit render bfs --quality medium --timing development --max-events 30
```

## ⚙️ Configuration

The CLI automatically manages:
- **Timing configuration**: Updates `data/examples/scenarios/bfs_timing_config.yaml`
- **Scenario loading**: Sets environment variables for Manim
- **Output directories**: Creates and manages video output locations

## 📁 Output

Videos are saved to:
- **Default**: `media/videos/render_bfs_scene/`
- **Custom**: Use `--output` option to specify directory

## 🔧 Advanced Usage

### Custom Scenario Files
```bash
# Use any YAML scenario file
algokit render bfs --scenario /path/to/my_scenario.yaml
```

### Environment Variables
The CLI automatically sets:
- `AGLOVIZ_SCENARIO_FILE`: Path to scenario file
- `MANIM_OUTPUT_DIR`: Output directory (if specified)

### Integration with Timing Config
The CLI updates the timing configuration file automatically based on your options:
- `--timing` sets the speed mode
- `--max-events` limits BFS events
- `--preview` enables quick demo mode

## 🎨 Quality Settings

| Quality | Resolution | FPS | Use Case |
|---------|------------|-----|----------|
| `low` | 480p | 15 | Quick testing |
| `medium` | 720p | 30 | Development |
| `high` | 1080p | 60 | Final output |
| `ultra` | 4K | 60 | Professional |

## ⏱️ Timing Modes

| Mode | Speed | Use Case |
|------|-------|----------|
| `cinematic` | 1x (normal) | Final videos |
| `development` | 4-10x faster | Development |
| `quick_demo` | 8-20x faster | Quick testing |

## 🚨 Troubleshooting

### Common Issues

**"No scenarios found"**
```bash
# Check if scenarios exist
ls data/examples/scenarios/*.yaml
```

**"Scene file not found"**
```bash
# Make sure you're in the project root
pwd
# Should show: /path/to/algokit
```

**"Render failed"**
```bash
# Try with preview mode first
algokit render bfs --preview

# Check Manim installation
uv run manim --version
```

### Getting Help
```bash
# General help
algokit render --help

# Command-specific help
algokit render render --help
algokit render scenarios --help
```

## 🎯 Pro Tips

1. **Start with preview**: Always test with `--preview` first
2. **Use quick mode**: For development, use `algokit render quick`
3. **Limit events**: Use `--max-events` for faster iteration
4. **Check scenarios**: Use `algokit render scenarios --details` to see options
5. **Custom timing**: Edit `data/examples/scenarios/bfs_timing_config.yaml` for fine-tuning
