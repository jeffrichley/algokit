# AlgoKit CLI Refactor Guide

## Overview

The AlgoKit CLI has been successfully refactored to eliminate the confusing double `render` command structure. Users can now run algorithm visualizations with a clean, intuitive command structure.

## ✅ What Changed

### Before (Confusing)
```bash
# ❌ Old confusing structure
algokit render render bfs --quality low --timing cinematic
```

### After (Clean)
```bash
# ✅ New clean structure  
algokit render bfs --quality low --timing cinematic
```

## 🚀 New Command Structure

### Basic Algorithm Rendering
```bash
# Render BFS with default settings
algokit render bfs

# Render with specific quality and timing
algokit render bfs --quality low --timing cinematic

# Render with scenario
algokit render bfs --scenario "Tiny Test"

# Render with custom output directory
algokit render bfs --output ./my_videos --quality high
```

### Preset System
```bash
# Quick preset (low quality, development timing)
algokit render bfs --preset quick

# Demo preset (high quality, cinematic timing)  
algokit render bfs --preset demo

# Override preset settings with CLI flags
algokit render bfs --preset demo --timing development
```

### Shortcut Commands
```bash
# Quick render shortcut
algokit render quick [algorithm]

# Demo render shortcut
algokit render demo [algorithm]

# Preview mode (same as quick preset)
algokit render bfs --preview
```

### Environment Variable Support
```bash
# Use scenario from environment variable
export AGLOVIZ_SCENARIO_FILE=data/examples/scenarios/harbor_flood_small.yaml
algokit render bfs

# Or inline
AGLOVIZ_SCENARIO_FILE=data/examples/tiny.yaml algokit render bfs
```

## 📋 Complete Command Reference

### Main Render Command
```bash
algokit render <algorithm> [OPTIONS]
```

**Arguments:**
- `algorithm`: Algorithm to render (e.g., 'bfs', 'breadth-first-search')

**Options:**
- `--preset`: Preset configuration (none, quick, demo) [default: none]
- `--scenario, -s`: Scenario name or file path
- `--quality, -q`: Video quality (low, medium, high, ultra) [default: medium]
- `--timing, -t`: Animation timing (cinematic, development, quick_demo) [default: cinematic]
- `--max-events, -e`: Maximum BFS events to display
- `--output, -o`: Output directory for videos
- `--preview, -p`: Preview mode (low quality, fast timing)
- `--progress`: Show rendering progress [default: True]

### Preset Configurations

| Preset | Quality | Timing | Use Case |
|--------|---------|--------|----------|
| `none` | medium | cinematic | Default balanced settings |
| `quick` | low | development | Fast iteration during development |
| `demo` | high | cinematic | High-quality presentations |

### Subcommands
```bash
# List available scenarios
algokit render scenarios

# List available algorithms  
algokit render algorithms

# Quick render shortcut
algokit render quick [algorithm]

# Demo render shortcut
algokit render demo [algorithm]
```

## 🔧 Algorithm Registry System

### Currently Registered Algorithms
- `bfs` or `breadth-first-search` - Breadth-First Search visualization

### Adding New Algorithms
The new system uses a decorator-based registry for easy algorithm addition:

```python
from algokit.cli.render import register_algorithm

@register_algorithm("dijkstra")
@register_algorithm("shortest-path")  # Alternative name
def render_dijkstra(*, scenario: str | None, quality: str, timing: str, output: str | None) -> None:
    """Render Dijkstra's algorithm visualization."""
    # Implementation here
    pass
```

## 🔄 Backward Compatibility

The old double `render` command structure still works but is hidden from help:

```bash
# ✅ Still works (hidden legacy support)
algokit render render bfs --quality low

# ✅ New preferred syntax
algokit render bfs --quality low
```

## 📊 Timing Tracker Integration

The timing tracker automatically works with all command variations:

```bash
# All of these generate timing reports
algokit render bfs --quality low --timing cinematic
algokit render bfs --preset quick  
algokit render quick bfs
algokit render demo bfs
```

Reports are saved to `output/bfs_timing_report_TIMESTAMP.txt` with comprehensive statistics.

## 🎯 Common Usage Examples

### Development Workflow
```bash
# Quick test render during development
algokit render bfs --preset quick

# Test with specific scenario
algokit render bfs --preset quick --scenario "Tiny Test"

# Limit events for faster testing
algokit render bfs --preset quick --max-events 10
```

### Production/Demo Workflow
```bash
# High-quality demo video
algokit render bfs --preset demo

# Custom high-quality with specific scenario
algokit render bfs --quality high --timing cinematic --scenario "Harbor Storm"

# Save to specific directory
algokit render bfs --preset demo --output ./presentation_videos
```

### Comparison/Analysis Workflow
```bash
# Compare different timing modes
algokit render bfs --timing cinematic --output ./cinematic
algokit render bfs --timing development --output ./development
algokit render bfs --timing quick_demo --output ./quick

# Compare different qualities
algokit render bfs --quality low --output ./low_quality
algokit render bfs --quality high --output ./high_quality
```

## 🧪 Testing the New CLI

### Verification Commands
```bash
# Test help shows no double render
algokit render --help

# Test basic algorithm rendering
algokit render bfs

# Test with options
algokit render bfs --quality low --timing cinematic

# Test presets
algokit render bfs --preset quick
algokit render bfs --preset demo

# Test shortcuts
algokit render quick
algokit render demo

# Test legacy compatibility (should work but hidden)
algokit render render bfs

# Test error handling
algokit render nonexistent_algorithm  # Should show clear error
```

### Expected Behaviors
- ✅ `algokit render --help` shows no "render render" in examples
- ✅ `algokit render bfs` works without double render
- ✅ `algokit render bfs --quality low --timing cinematic` works with options
- ✅ `algokit render nonexistent` shows clear error with available algorithms
- ✅ `algokit render` with no args shows help
- ✅ Legacy `algokit render render bfs` still works (hidden)

## 🔍 Technical Implementation

### Key Changes Made

1. **Callback-Based Dispatch**: Used `@app.callback(invoke_without_command=True)` to handle algorithm arguments directly
2. **Algorithm Registry**: Decorator-based system for registering algorithm renderers
3. **Preset System**: Built-in presets with CLI flag override capability
4. **Environment Variable Support**: Automatic fallback to `AGLOVIZ_SCENARIO_FILE`
5. **Legacy Compatibility**: Hidden `render` subcommand for backward compatibility

### Architecture Benefits

- **Extensible**: Adding new algorithms requires only a decorator
- **Clean UX**: No more confusing double commands
- **Flexible**: Presets + CLI overrides provide power and convenience
- **Compatible**: Old commands still work during transition
- **Type-Safe**: Full typing with Protocol-based renderer interface

## 📈 Performance Impact

The refactor has **no performance impact** on rendering:
- Same Manim execution path
- Same timing tracker integration  
- Same scenario loading system
- Same output generation

The changes are purely CLI interface improvements.

## 🐛 Troubleshooting

### Common Issues

**Command not recognized:**
```bash
# ❌ Old way (might be confusing)
algokit render render bfs

# ✅ New way
algokit render bfs
```

**Algorithm not found:**
```bash
# Check available algorithms
algokit render algorithms

# Use exact algorithm name
algokit render bfs  # not "BFS" or "breadth_first_search"
```

**Options not working:**
```bash
# ✅ Correct syntax
algokit render bfs --quality low --timing cinematic

# ❌ Wrong order
algokit render --quality low bfs --timing cinematic
```

### Debug Commands
```bash
# List available algorithms
algokit render algorithms

# List available scenarios  
algokit render scenarios

# Check help for specific command
algokit render --help
algokit render quick --help
algokit render demo --help
```

## 🚀 Migration Guide

### For Users
1. **Update commands**: Remove the double `render` from existing commands
2. **Use presets**: Try `--preset quick` and `--preset demo` for common use cases
3. **Environment variables**: Set `AGLOVIZ_SCENARIO_FILE` for default scenarios

### For Developers
1. **Add algorithms**: Use `@register_algorithm("name")` decorator
2. **Follow interface**: Implement the `Renderer` protocol
3. **Test thoroughly**: Verify both new and legacy command structures work

## 📚 Related Documentation

- [Timing Tracker Guide](TIMING_TRACKER_GUIDE.md) - Comprehensive timing analysis
- [Render CLI Guide](RENDER_CLI_GUIDE.md) - Detailed rendering options
- [Algorithm Development Guide](docs/contributing.md) - Adding new algorithms

## ✅ Success Metrics

The CLI refactor successfully achieved:

- ✅ **Eliminated double render**: `algokit render bfs` vs `algokit render render bfs`
- ✅ **Improved UX**: Clean, intuitive command structure
- ✅ **Maintained compatibility**: Legacy commands still work (hidden)
- ✅ **Added flexibility**: Preset system with CLI overrides
- ✅ **Enhanced extensibility**: Easy algorithm registration
- ✅ **Preserved functionality**: All existing features work unchanged
- ✅ **Comprehensive testing**: All command variations verified

The refactor transforms the CLI from confusing to intuitive while maintaining full backward compatibility and adding powerful new features.
