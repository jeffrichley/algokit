# BFS Scene Timing Tracker Guide

## Overview

The timing tracker system provides comprehensive monitoring and analysis of all timing requests in the BFS scene animations. It tracks how many times each animation and wait is requested, what duration is used, and generates detailed reports for performance analysis.

## Features

✅ **Comprehensive Tracking**: Monitors all animation timings, wait times, and legacy timing requests  
✅ **Detailed Statistics**: Count, base time, average time, and total time for each timing request  
✅ **Multiple Report Formats**: Console output and detailed file reports  
✅ **Mode Comparison**: Compare timing usage across cinematic, development, and quick_demo modes  
✅ **Automatic Integration**: Seamlessly integrated into BreadthFirstSearchScene  
✅ **Performance Analysis**: Identify timing bottlenecks and optimization opportunities  

## How It Works

The timing tracker intercepts all calls to timing methods in the BFS scene:

- `get_animation_time(animation_name)` - Tracks animation timing requests
- `get_wait_time(wait_name)` - Tracks wait timing requests  
- `_get_timing(stage, base_time)` - Tracks legacy timing requests

Each request is logged with the timing name, requested duration, and current timing mode.

## Report Format

### Comprehensive Report

```
================================================================================
🎬 COMPREHENSIVE TIMING USAGE REPORT
================================================================================
Total Timing Requests: 106
  - Animation Requests: 73
  - Wait Requests: 33
  - Legacy Requests: 0

🎭 ANIMATION TIMINGS
--------------------------------------------------
Animation Name            Count    Base     Avg      Total     
--------------------------------------------------
path_dot_grow             25       0.400    0.400    10.000    
path_line_create          25       0.300    0.300    7.500     
layer_complete_pulse      5        0.800    0.800    4.000     
tracer_movement           1        3.000    3.000    3.000     
token_entrance            3        0.800    0.800    2.400     
title_write               1        1.800    1.800    1.800     
--------------------------------------------------
TOTAL ANIMATION TIME                                 39.300    

⏳ WAIT TIMINGS
--------------------------------------------------
Wait Name                 Count    Base     Avg      Total     
--------------------------------------------------
dequeue_delay             25       0.300    0.300    7.500     
layer_complete_wait       5        1.000    1.000    5.000     
final_study_pause         1        4.000    4.000    4.000     
initial_setup             1        3.000    3.000    3.000     
celebration_pause         1        0.500    0.500    0.500     
--------------------------------------------------
TOTAL WAIT TIME                                      20.000    

🏁 GRAND TOTALS
------------------------------
Total Requests: 106
Total Time: 59.300 seconds
Average Time per Request: 0.559 seconds
```

### Key Metrics Explained

- **Count**: Number of times this timing was requested
- **Base**: Base timing value from configuration (cinematic mode)
- **Avg**: Average timing value across all requests  
- **Total**: Sum of all timing requests (Count × Average)

## Usage Examples

### Basic Usage (Automatic)

The timing tracker is automatically enabled when you create a BreadthFirstSearchScene:

```python
from agloviz.scenes.planning.bfs.breadth_first_search_scene import BreadthFirstSearchScene

# Create scene - timing tracker is automatically initialized
scene = BreadthFirstSearchScene(scenario=your_scenario)

# Run the scene - timings are automatically tracked
scene.construct()  # This will generate a report at the end
```

### Manual Report Generation

```python
# Generate report without running full scene
scene = BreadthFirstSearchScene(scenario=your_scenario)

# Simulate some timing requests
scene.get_animation_time("title_write")
scene.get_wait_time("initial_setup")

# Generate and save report
report = scene.generate_timing_report(
    save_to_file=True,      # Save to output/bfs_timing_report_TIMESTAMP.txt
    print_to_console=True   # Print to console
)

# Print quick summary
scene.print_timing_summary()

# Get raw statistics
stats = scene.get_timing_stats()
```

### Advanced Analysis

```python
from agloviz.core.timing_tracker import get_timing_tracker

# Get the global timing tracker
tracker = get_timing_tracker()

# Get detailed stats for specific animations
animation_stats = tracker.get_animation_stats("path_dot_grow")
print(f"Path dot grow: {animation_stats['count']} requests, {animation_stats['total_time']}s total")

# Get detailed stats for specific waits
wait_stats = tracker.get_wait_stats("dequeue_delay")
print(f"Dequeue delay: {wait_stats['count']} requests, {wait_stats['total_time']}s total")

# Generate detailed breakdown
detailed_report = tracker.generate_detailed_breakdown()
print(detailed_report)
```

### Mode Comparison Analysis

```python
# Compare timing across different modes
modes = ["cinematic", "development", "quick_demo"]
results = {}

for mode in modes:
    # Reset tracker
    from agloviz.core.timing_tracker import reset_timing_tracker
    reset_timing_tracker()
    
    # Create scene with specific mode
    scene = BreadthFirstSearchScene(scenario=scenario)
    scene.set_timing_mode(mode)
    
    # Simulate typical requests
    scene.get_animation_time("title_write")
    scene.get_wait_time("initial_setup")
    # ... more requests
    
    # Get results
    tracker = get_timing_tracker()
    total_time = sum(tracker.get_animation_stats(name)['total_time'] 
                    for name in tracker.animation_requests.keys())
    results[mode] = total_time

# Compare results
for mode, time in results.items():
    speedup = results["cinematic"] / time if time > 0 else 0
    print(f"{mode}: {time:.1f}s ({speedup:.1f}x speedup)")
```

## Sample Analysis Results

### Timing Insights from Real BFS Scene

Based on analysis of a typical BFS scene with 25 BFS events:

**Most Frequently Requested Animations:**
- `path_dot_grow`: 25 requests, 10.000s total
- `path_line_create`: 25 requests, 7.500s total  
- `layer_complete_pulse`: 5 requests, 4.000s total

**Most Time-Consuming Waits:**
- `dequeue_delay`: 7.500s total (25 requests)
- `layer_complete_wait`: 5.000s total (5 requests)
- `final_study_pause`: 4.000s total (1 request)

**Total Scene Time:** 59.3 seconds
- Animation Time: 39.3s (66.3%)
- Wait Time: 20.0s (33.7%)

### Mode Comparison Results

| Mode | Total Time | Speedup |
|------|------------|---------|
| cinematic | 22.8s | 1.0x |
| development | 4.4s | 5.2x |
| quick_demo | 2.2s | 10.4x |

## Performance Optimization

### Identifying Bottlenecks

The timing tracker helps identify performance bottlenecks:

1. **High-frequency animations**: Look for animations with high request counts
2. **Long-duration operations**: Look for animations/waits with high total times
3. **Inefficient patterns**: Look for unexpected timing patterns

### Optimization Strategies

Based on timing analysis, you can:

1. **Reduce wait times** in development mode for faster iteration
2. **Optimize high-frequency animations** like `path_dot_grow` and `path_line_create`
3. **Adjust timing modes** based on use case (presentation vs development)
4. **Identify unnecessary delays** in the animation sequence

## File Output

Reports are automatically saved to:
- `output/bfs_timing_report_TIMESTAMP.txt` - Comprehensive report
- `output/timing_tracker_demo_report.txt` - Demo report (from examples)

Each report includes:
- **Comprehensive summary** with totals and breakdowns
- **Detailed breakdown** showing all individual requests
- **Timing configuration** information
- **Performance analysis** insights

## Integration with CLI

The timing tracker works seamlessly with the CLI timing modes:

```bash
# Run with timing tracking in different modes
algokit render bfs --timing-mode cinematic    # Full timing analysis
algokit render bfs --timing-mode development  # Fast mode analysis  
algokit render bfs --timing-mode quick_demo   # Ultra-fast mode analysis
```

The timing mode is automatically detected from the `AGLOVIZ_TIMING_MODE` environment variable set by the CLI.

## API Reference

### TimingTracker Class

```python
class TimingTracker:
    def track_animation_time(self, animation_name: str, requested_time: float) -> float
    def track_wait_time(self, wait_name: str, requested_time: float) -> float
    def track_legacy_timing(self, stage: str, base_time: float, returned_time: float) -> float
    def get_animation_stats(self, animation_name: str) -> dict[str, Any]
    def get_wait_stats(self, wait_name: str) -> dict[str, Any]
    def generate_comprehensive_report(self) -> str
    def generate_detailed_breakdown(self) -> str
    def save_report_to_file(self, filename: str, include_detailed: bool = True) -> None
    def reset(self) -> None
```

### BreadthFirstSearchScene Methods

```python
def generate_timing_report(self, save_to_file: bool = True, print_to_console: bool = True) -> str
def print_timing_summary(self) -> None  
def get_timing_stats(self) -> dict[str, Any]
```

### Global Functions

```python
def get_timing_tracker() -> TimingTracker
def reset_timing_tracker() -> None
```

## Examples

See the following example files for complete usage demonstrations:

- `examples/timing_tracker_demo.py` - Basic timing tracker functionality
- `examples/bfs_timing_analysis.py` - BFS scene timing analysis
- `examples/bfs_custom_timing.py` - Custom timing configuration

## Troubleshooting

### Common Issues

**No timing data collected:**
- Ensure the scene is properly initialized
- Check that timing methods are being called
- Verify timing tracker is properly integrated

**Missing report file:**
- Check that `output/` directory exists
- Verify write permissions
- Check for file path errors

**Unexpected timing values:**
- Verify timing configuration file is loaded correctly
- Check current timing mode setting
- Ensure proper timing mode transitions

### Debug Information

Enable debug output:

```python
# Print current timing configuration
scene.timing_config.print_current_settings()

# Print timing tracker summary  
scene.print_timing_summary()

# Check timing tracker state
tracker = get_timing_tracker()
print(f"Total requests: {tracker.total_animation_requests + tracker.total_wait_requests}")
```

## Conclusion

The timing tracker provides powerful insights into BFS scene performance and timing usage. Use it to:

- **Optimize animation sequences** for better performance
- **Compare timing modes** for different use cases  
- **Identify bottlenecks** in scene construction
- **Generate comprehensive reports** for analysis
- **Monitor timing changes** during development

The system is designed to be transparent and automatic, requiring no changes to existing scene code while providing detailed timing analytics.
